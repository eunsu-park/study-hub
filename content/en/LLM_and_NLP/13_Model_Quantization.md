# 13. Model Quantization

## Learning Objectives

- Understand the concept and necessity of quantization
- INT8/INT4 quantization techniques
- Practice with GPTQ, AWQ, bitsandbytes
- Efficient fine-tuning with QLoRA

---

## 1. Quantization Overview

### Why is Quantization Needed?

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Memory Requirements                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Model Size   │  FP32      │  FP16      │  INT8    │  INT4   │
│  ─────────────┼────────────┼────────────┼──────────┼─────────│
│  7B params    │  28GB      │  14GB      │  7GB     │  3.5GB  │
│  13B params   │  52GB      │  26GB      │  13GB    │  6.5GB  │
│  70B params   │  280GB     │  140GB     │  70GB    │  35GB   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Quantization Types

| Type | Description | Advantages | Disadvantages |
|------|-------------|------------|---------------|
| Post-Training Quantization (PTQ) | Quantize after training | Fast, simple | Possible accuracy loss |
| Quantization-Aware Training (QAT) | Simulate quantization during training | High accuracy | Increased training time |
| Dynamic Quantization | Runtime quantization | Flexible | Inference overhead |
| Static Quantization | Calibration-based | Fast inference | Calibration required |

### Bit Precision Comparison

```python
# FP32 (32-bit floating point)
# Sign 1bit + Exponent 8bit + Mantissa 23bit
# Range: ±3.4 × 10^38, Precision: ~7 digits

# FP16 (16-bit floating point)
# Sign 1bit + Exponent 5bit + Mantissa 10bit
# Range: ±65,504, Precision: ~3 digits

# BF16 (Brain Float 16)
# Sign 1bit + Exponent 8bit + Mantissa 7bit
# Same range as FP32, lower precision

# INT8 (8-bit integer)
# Range: -128 ~ 127 or 0 ~ 255

# INT4 (4-bit integer)
# Range: -8 ~ 7 or 0 ~ 15
```

---

## 2. Quantization Mathematics

### Uniform Quantization

```python
import numpy as np

def quantize_symmetric(tensor, bits=8):
    """Symmetric quantization"""
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1

    # Calculate scale
    abs_max = np.abs(tensor).max()
    scale = abs_max / qmax

    # Quantize
    quantized = np.round(tensor / scale).astype(np.int8)
    quantized = np.clip(quantized, qmin, qmax)

    return quantized, scale

def dequantize(quantized, scale):
    """Dequantization"""
    return quantized.astype(np.float32) * scale


# Test
original = np.array([0.5, -1.2, 0.3, 2.1, -0.8], dtype=np.float32)
quantized, scale = quantize_symmetric(original, bits=8)
recovered = dequantize(quantized, scale)

print(f"Original: {original}")
print(f"Quantized: {quantized}")
print(f"Recovered: {recovered}")
print(f"Error: {np.abs(original - recovered).mean():.6f}")
```

### Asymmetric Quantization

```python
def quantize_asymmetric(tensor, bits=8):
    """Asymmetric quantization (zero is exactly represented)"""
    qmin = 0
    qmax = 2 ** bits - 1

    # Scale and zero point
    min_val = tensor.min()
    max_val = tensor.max()
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = round(-min_val / scale)

    # Quantize
    quantized = np.round(tensor / scale + zero_point).astype(np.uint8)
    quantized = np.clip(quantized, qmin, qmax)

    return quantized, scale, zero_point

def dequantize_asymmetric(quantized, scale, zero_point):
    """Asymmetric dequantization"""
    return (quantized.astype(np.float32) - zero_point) * scale
```

### Group Quantization

```python
def group_quantize(tensor, group_size=128, bits=4):
    """Group-wise quantization - improved accuracy"""
    # Split tensor into groups
    flat = tensor.flatten()
    pad_size = (group_size - len(flat) % group_size) % group_size
    flat = np.pad(flat, (0, pad_size))

    groups = flat.reshape(-1, group_size)

    quantized_groups = []
    scales = []

    for group in groups:
        q, s = quantize_symmetric(group, bits)
        quantized_groups.append(q)
        scales.append(s)

    return np.array(quantized_groups), np.array(scales)
```

---

## 3. bitsandbytes Library

### Installation

```bash
pip install bitsandbytes
pip install accelerate
```

### 8-bit Quantization

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load in 8-bit
model_8bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Check memory
print(f"8bit model memory: {model_8bit.get_memory_footprint() / 1e9:.2f} GB")

# Inference
inputs = tokenizer("Hello, my name is", return_tensors="pt").to("cuda")
outputs = model_8bit.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### 4-bit Quantization (NF4)

```python
from transformers import BitsAndBytesConfig

# 4-bit configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # Normal Float 4 (optimized data type)
    bnb_4bit_compute_dtype=torch.bfloat16,  # Computation data type
    bnb_4bit_use_double_quant=True      # Double quantization (quantize scales too)
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

print(f"4bit model memory: {model_4bit.get_memory_footprint() / 1e9:.2f} GB")
```

### NF4 vs FP4

```python
# NF4 (Normal Float 4)
# - Optimal quantization assuming normal distribution
# - Optimized for LLM weights

# FP4 (Floating Point 4)
# - General 4-bit floating point
# - General purpose

bnb_config_fp4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="fp4",  # Use FP4
)
```

---

## 4. GPTQ (GPU-optimized Post-Training Quantization)

### Concept

```
GPTQ quantization process:
    1. Prepare small calibration dataset
    2. Layer-wise sequential quantization
    3. Identify important weights using Hessian matrix
    4. Minimize reconstruction error

Advantages:
    - High compression ratio (3-4bit)
    - Fast inference speed
    - GPU optimized
```

### Performing GPTQ Quantization

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# Calibration data
calibration_data = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    # ... more data
]

# GPTQ configuration
gptq_config = GPTQConfig(
    bits=4,
    group_size=128,                    # Group size
    desc_act=True,                     # Activation order descending
    dataset=calibration_data,
    tokenizer=tokenizer
)

# Quantize and save
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=gptq_config,
    device_map="auto"
)

model.save_pretrained("./llama-2-7b-gptq-4bit")
tokenizer.save_pretrained("./llama-2-7b-gptq-4bit")
```

### Using AutoGPTQ

```python
# pip install auto-gptq
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Quantization configuration
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False
)

# Load model
model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantize_config
)

# Calibration data
examples = [tokenizer(text, return_tensors="pt") for text in calibration_data]

# Quantize
model.quantize(examples, batch_size=1)

# Save
model.save_quantized("./llama-2-7b-gptq")
```

### Using Pre-quantized Models

```python
# Download GPTQ models from TheBloke, etc.
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GPTQ",
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-GPTQ")

# Inference
inputs = tokenizer("What is AI?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

---

## 5. AWQ (Activation-aware Weight Quantization)

### Concept

```
AWQ features:
    - Calculate weight importance based on activations
    - Maintain high precision for important weights
    - Faster quantization than GPTQ
    - Similar or better quality
```

### AWQ Quantization

```python
# pip install autoawq
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Load model
model_path = "meta-llama/Llama-2-7b-hf"
quant_path = "./llama-2-7b-awq"

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# AWQ quantization configuration
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"  # GEMM or GEMV
}

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

### AWQ Model Inference

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Load AWQ model
model = AutoAWQForCausalLM.from_quantized(
    "./llama-2-7b-awq",
    fuse_layers=True  # Speed up with layer fusion
)
tokenizer = AutoTokenizer.from_pretrained("./llama-2-7b-awq")

# Inference
prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 6. QLoRA (Quantized LoRA)

### Concept

```
QLoRA = 4bit quantization + LoRA

    Base model (4bit quantized, frozen)
         │
         ▼
    ┌─────────────┐
    │  LoRA A     │  (FP16, trainable)
    │  (r × d)    │
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │  LoRA B     │  (FP16, trainable)
    │  (d × r)    │
    └─────────────┘
         │
         ▼
    Final output = quantized weights + LoRA correction
```

### QLoRA Fine-tuning

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,                          # LoRA rank
    lora_alpha=32,                 # Scaling factor
    target_modules=[               # Modules to apply
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Trainable: ~0.1%, total ~400MB

# Dataset
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

def format_prompt(example):
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['context']}

### Response:
{example['response']}"""

# Training configuration
training_args = TrainingArguments(
    output_dir="./qlora_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit"  # Memory-efficient optimizer
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=format_prompt,
    max_seq_length=512,
    args=training_args,
)

# Train
trainer.train()

# Save
model.save_pretrained("./qlora_adapter")
```

### Merging QLoRA Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Base model (load in FP16)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Merge LoRA adapter
model = PeftModel.from_pretrained(base_model, "./qlora_adapter")
model = model.merge_and_unload()  # Merge adapter to base model

# Save merged model
model.save_pretrained("./llama-2-7b-finetuned")
```

---

## 7. Quantization Performance Comparison

### Benchmark

```python
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def benchmark_model(model, tokenizer, prompt, num_runs=5):
    """Model inference benchmark"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=50)

    # Benchmark
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)

        torch.cuda.synchronize()
        times.append(time.time() - start)

    return {
        "avg_time": sum(times) / len(times),
        "memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        "output": tokenizer.decode(outputs[0])
    }

# Compare results
models = {
    "FP16": model_fp16,
    "INT8": model_8bit,
    "INT4 (NF4)": model_4bit,
    "GPTQ-4bit": model_gptq,
    "AWQ-4bit": model_awq,
}

prompt = "Explain the theory of relativity:"

for name, model in models.items():
    result = benchmark_model(model, tokenizer, prompt)
    print(f"{name}:")
    print(f"  Time: {result['avg_time']:.2f}s")
    print(f"  Memory: {result['memory_gb']:.2f} GB")
```

### Accuracy Evaluation

```python
from datasets import load_dataset
import evaluate

# Evaluation dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

def compute_perplexity(model, tokenizer, texts, max_length=1024):
    """Calculate perplexity"""
    total_loss = 0
    total_tokens = 0

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()

# Compare
for name, model in models.items():
    ppl = compute_perplexity(model, tokenizer, dataset["text"][:100])
    print(f"{name} Perplexity: {ppl:.2f}")
```

---

## 8. Practical Guide

### Choosing Quantization Method

```
┌─────────────────────────────────────────────────────────────┐
│                Quantization Method Selection Guide           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Purpose                 │  Recommended Method               │
│  ────────────────────────┼────────────────────────────────────│
│  Fast prototyping        │  bitsandbytes (load_in_8bit)      │
│  Memory-constrained env  │  bitsandbytes (load_in_4bit)      │
│  Production deployment   │  GPTQ or AWQ                      │
│  Fine-tuning needed      │  QLoRA                            │
│  Maximum speed           │  AWQ + fuse_layers                │
│  Maximum quality         │  GPTQ (desc_act=True)             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Troubleshooting

```python
# 1. CUDA Out of Memory
# - Reduce batch size
# - Enable gradient_checkpointing
# - Use lower bit quantization

model.gradient_checkpointing_enable()

# 2. Quality degradation after quantization
# - Reduce group_size (64 or 32)
# - Increase calibration data
# - Try GPTQ instead of AWQ

# 3. Slow inference
# - Enable fuse_layers=True
# - Use exllama backend (GPTQ)
# - Utilize batch processing

from auto_gptq import exllama_set_max_input_length
exllama_set_max_input_length(model, 4096)
```

---

## Summary

### Quantization Comparison Table

| Method | Bits | Speed | Quality | Ease of Use |
|--------|------|-------|---------|-------------|
| FP16 | 16 | Baseline | Baseline | Easy |
| INT8 (bitsandbytes) | 8 | Fast | High | Easy |
| INT4 (NF4) | 4 | Fast | Good | Easy |
| GPTQ | 4/3/2 | Very Fast | Good | Medium |
| AWQ | 4 | Very Fast | Good | Medium |
| QLoRA | 4 | - | Training | Medium |

### Core Code

```python
# bitsandbytes 4-bit
from transformers import BitsAndBytesConfig
config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=config)

# QLoRA
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# GPTQ
model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)
model.quantize(examples)

# AWQ
model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)
```

---

## Exercises

### Exercise 1: Quantization Memory Calculation

A transformer model has the following architecture: 32 layers, each with attention (4 weight matrices of shape 4096×4096) and FFN (2 matrices of shape 4096×16384 and one of shape 16384×4096). Calculate the approximate memory requirement for each precision format and fill in the table.

| Precision | Bytes per param | Attention layer (MB) | FFN layer (MB) | Full model (GB) |
|-----------|----------------|---------------------|----------------|-----------------|
| FP32 | 4 | ? | ? | ? |
| FP16/BF16 | 2 | ? | ? | ? |
| INT8 | 1 | ? | ? | ? |
| INT4 | 0.5 | ? | ? | ? |

<details>
<summary>Show Answer</summary>

```python
def calculate_model_memory(
    num_layers: int,
    hidden_size: int,
    ffn_size: int,
    bytes_per_param: float
) -> dict:
    """Calculate model memory requirements."""

    # Attention: Q, K, V, O projections each of shape (hidden, hidden)
    attention_params = 4 * hidden_size * hidden_size
    attention_mb = attention_params * bytes_per_param / (1024 ** 2)

    # FFN: up-projection (hidden→ffn), down-projection (ffn→hidden)
    # Note: many models (LLaMA) have 3 matrices: gate, up, down
    # Here: 2 matrices (hidden→ffn) + 1 matrix (ffn→hidden)
    ffn_params = 2 * (hidden_size * ffn_size) + (ffn_size * hidden_size)
    ffn_mb = ffn_params * bytes_per_param / (1024 ** 2)

    total_params = num_layers * (attention_params + ffn_params)
    total_gb = total_params * bytes_per_param / (1024 ** 3)

    return {
        "attention_params": attention_params,
        "attention_mb": attention_mb,
        "ffn_params": ffn_params,
        "ffn_mb": ffn_mb,
        "total_params": total_params,
        "total_gb": total_gb,
    }

# Model: 32 layers, hidden=4096, ffn=16384
NUM_LAYERS = 32
HIDDEN = 4096
FFN = 16384

precisions = {
    "FP32":     4.0,
    "FP16/BF16": 2.0,
    "INT8":     1.0,
    "INT4":     0.5,
}

print(f"{'Precision':<12} {'Bytes/param':<13} {'Attn (MB)':<12} {'FFN (MB)':<11} {'Total (GB)'}")
print("-" * 60)
for name, bpp in precisions.items():
    r = calculate_model_memory(NUM_LAYERS, HIDDEN, FFN, bpp)
    print(f"{name:<12} {bpp:<13.1f} {r['attention_mb']:<12.1f} {r['ffn_mb']:<11.1f} {r['total_gb']:.2f}")

# Output:
# Precision    Bytes/param   Attn (MB)    FFN (MB)    Total (GB)
# FP32         4.0           256.0        384.0        20.00
# FP16/BF16    2.0           128.0        192.0        10.00
# INT8         1.0           64.0         96.0         5.00
# INT4         0.5           32.0         48.0         2.50
```

**Key insight:** INT4 reduces memory by 8x vs FP32. For a real LLaMA-2-7B model (~7B params), practical INT4 memory is ~3.5GB vs ~28GB for FP32 — the difference between requiring a consumer GPU vs a data center GPU.
</details>

---

### Exercise 2: Symmetric vs Asymmetric Quantization

Given the weight tensor below, apply both symmetric and asymmetric INT8 quantization. Calculate the quantization error for each method and explain why asymmetric quantization handles non-centered distributions better.

```python
import numpy as np

# Simulates a weight tensor with non-centered distribution
weights = np.array([0.01, 0.05, 0.12, 0.23, 0.45, 0.67, 0.89, 1.20, 1.45, 1.80],
                   dtype=np.float32)
```

<details>
<summary>Show Answer</summary>

```python
import numpy as np

weights = np.array([0.01, 0.05, 0.12, 0.23, 0.45, 0.67, 0.89, 1.20, 1.45, 1.80],
                   dtype=np.float32)

# --- Symmetric INT8 quantization ---
def quantize_symmetric(tensor, bits=8):
    qmin = -(2 ** (bits - 1))       # -128
    qmax = 2 ** (bits - 1) - 1      #  127

    abs_max = np.abs(tensor).max()
    scale = abs_max / qmax           # scale = 1.80 / 127 ≈ 0.01417

    quantized = np.round(tensor / scale).clip(qmin, qmax).astype(np.int8)
    return quantized, scale

def dequantize_sym(q, scale):
    return q.astype(np.float32) * scale

# --- Asymmetric INT8 quantization ---
def quantize_asymmetric(tensor, bits=8):
    qmin = 0
    qmax = 2 ** bits - 1             # 255

    min_val = tensor.min()           # ≈ 0.01
    max_val = tensor.max()           # ≈ 1.80
    scale = (max_val - min_val) / (qmax - qmin)  # ≈ 0.007020
    zero_point = round(-min_val / scale)          # ≈ -1 → 0 (clipped)

    quantized = np.round(tensor / scale + zero_point).clip(qmin, qmax).astype(np.uint8)
    return quantized, scale, zero_point

def dequantize_asym(q, scale, zp):
    return (q.astype(np.float32) - zp) * scale

# Apply
q_sym, s_sym = quantize_symmetric(weights)
rec_sym = dequantize_sym(q_sym, s_sym)
error_sym = np.abs(weights - rec_sym)

q_asym, s_asym, zp_asym = quantize_asymmetric(weights)
rec_asym = dequantize_asym(q_asym, s_asym, zp_asym)
error_asym = np.abs(weights - rec_asym)

print("Symmetric quantization:")
print(f"  Scale: {s_sym:.6f}")
print(f"  Mean error: {error_sym.mean():.6f}")
print(f"  Max error:  {error_sym.max():.6f}")

print("\nAsymmetric quantization:")
print(f"  Scale: {s_asym:.6f}, Zero point: {zp_asym}")
print(f"  Mean error: {error_asym.mean():.6f}")
print(f"  Max error:  {error_asym.max():.6f}")

# Why asymmetric wins for non-centered distributions:
# Symmetric scale = 1.80/127 ≈ 0.01417 (half the 256 range is wasted on negative values)
# Asymmetric scale = 1.79/255 ≈ 0.00702 (full 256 range covers 0.01–1.80)
# Smaller scale → finer granularity → less quantization error
print("\nGranularity improvement:", s_sym / s_asym, "x finer with asymmetric")
```

**Why asymmetric is better here:** The weights range from 0.01 to 1.80 — no negative values. Symmetric quantization wastes half its range (the negative side) on values that don't exist, forcing a coarser scale. Asymmetric quantization maps the full 0–255 range exactly to 0.01–1.80, achieving roughly 2x finer quantization granularity.
</details>

---

### Exercise 3: NF4 vs INT4 Intuition

The NF4 (Normal Float 4) data type uses non-uniform quantization levels, while INT4 uses uniform levels. Given that LLM weights typically follow a normal distribution, sketch or describe the NF4 quantization levels and explain why NF4 achieves lower quantization error than INT4 for normally distributed weights.

<details>
<summary>Show Answer</summary>

```python
import numpy as np
import scipy.stats as stats

# INT4 uses 16 UNIFORM quantization levels: -8, -7, ..., 0, ..., 7
int4_levels = np.arange(-8, 8)  # 16 uniform levels

# NF4 uses 16 NON-UNIFORM levels based on quantiles of N(0,1)
# The quantiles are chosen so each level covers an equal probability mass
num_levels = 16
# Divide N(0,1) into 16 equal-probability bins
probabilities = np.linspace(0, 1, num_levels + 1)[1:-1]  # 15 boundaries
nf4_boundaries = stats.norm.ppf(probabilities)            # z-score boundaries

# NF4 levels = midpoints of each probability bin (including tails)
prob_centers = np.linspace(1/(2*num_levels), 1 - 1/(2*num_levels), num_levels)
nf4_levels = stats.norm.ppf(prob_centers)  # 16 non-uniform levels

# Simulate normal distribution of weights
np.random.seed(42)
weights = np.random.normal(0, 0.1, size=10000)  # LLM-like weight distribution

def quantize_to_levels(weights, levels):
    """Map each weight to the nearest quantization level."""
    levels = np.sort(levels)
    # Find nearest level for each weight
    indices = np.abs(weights[:, None] - levels[None, :]).argmin(axis=1)
    quantized = levels[indices]
    return quantized

# Normalize weights to [-1, 1] for fair comparison
w_norm = weights / weights.std() * 0.1  # Scale to match typical LLM weight scale
w_norm = np.clip(w_norm, -0.8, 0.8)    # Clip outliers

# Scale levels to match weight range
int4_scaled = int4_levels / 8 * 0.8    # Scale INT4 to [-0.8, 0.8]
nf4_scaled = nf4_levels / nf4_levels.max() * 0.8  # Scale NF4 to match

q_int4 = quantize_to_levels(w_norm, int4_scaled)
q_nf4 = quantize_to_levels(w_norm, nf4_scaled)

print("INT4 quantization error:")
print(f"  Mean absolute error: {np.abs(w_norm - q_int4).mean():.6f}")
print(f"  Max absolute error:  {np.abs(w_norm - q_int4).max():.6f}")

print("\nNF4 quantization error:")
print(f"  Mean absolute error: {np.abs(w_norm - q_nf4).mean():.6f}")
print(f"  Max absolute error:  {np.abs(w_norm - q_nf4).max():.6f}")

# Visual explanation
print("\nKey insight:")
print("INT4 levels (uniform):", np.round(int4_scaled[:4], 3), "...", np.round(int4_scaled[-4:], 3))
print("NF4 levels (non-uniform):", np.round(nf4_scaled[:4], 3), "...", np.round(nf4_scaled[-4:], 3))
print("NF4 packs more levels near 0 (where most weights cluster)")
```

**The core insight:** For a normal distribution, about 68% of values fall within 1 standard deviation of the mean. INT4's uniform levels spread its 16 quantization steps evenly across the range — wasting many steps on the rarely-populated tails. NF4 concentrates more levels near zero (where most weights live), achieving lower average quantization error with the same number of bits. This is why NF4 is the recommended quantization type for LLM weights in bitsandbytes.
</details>

---

## Next Steps

In [14_RLHF_Alignment.md](./14_RLHF_Alignment.md), we'll learn about LLM alignment techniques (RLHF, DPO).

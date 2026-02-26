# 19. PEFT (Parameter-Efficient Fine-Tuning) Unified

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why Parameter-Efficient Fine-Tuning (PEFT) is necessary and quantify the memory and storage savings it provides over full fine-tuning
2. Derive the mathematical basis of LoRA (Low-Rank Adaptation) and implement it using the Hugging Face PEFT library
3. Compare additive, reparameterization, and selective PEFT strategies, evaluating their trade-offs in trainable parameters and downstream performance
4. Apply QLoRA to fine-tune large language models on consumer hardware by combining quantization with low-rank adapters
5. Design a PEFT training pipeline that selects the appropriate method (LoRA, Prefix Tuning, Prompt Tuning) based on task requirements and hardware constraints

---

## Overview

PEFT methodologies enable efficient adaptation by training only a small set of parameters instead of the entire model. This lesson covers various PEFT techniques in a unified manner.

---

## 1. PEFT Overview

### 1.1 Why PEFT?

```
Problems with Full Fine-tuning:
┌─────────────────────────────────────┐
│  LLaMA-7B                           │
│  - Parameters: 7B                   │
│  - FP16 Memory: 14GB                │
│  - Optimizer states: 56GB           │
│  - Gradients: 14GB                  │
│  - Total: ~84GB                     │
└─────────────────────────────────────┘

Advantages of PEFT:
┌─────────────────────────────────────┐
│  LoRA (rank=8)                      │
│  - Trainable parameters: ~0.1%      │
│  - Additional memory: ~100MB        │
│  - Performance: 90-95% of Full FT   │
│  - Storage: original + small adapter│
└─────────────────────────────────────┘
```

### 1.2 PEFT Method Classification

```
┌─────────────────────────────────────────────────────────────┐
│                     PEFT Methods                            │
├──────────────────┬──────────────────┬──────────────────────┤
│  Additive        │  Reparameterization │  Selective        │
│  ─────────       │  ─────────────────  │  ─────────        │
│  • Adapters      │  • LoRA             │  • BitFit         │
│  • Prefix Tuning │  • DoRA             │  • Diff Pruning   │
│  • Prompt Tuning │  • AdaLoRA          │  • Partial FT     │
│  • IA³           │  • QLoRA            │                   │
└──────────────────┴──────────────────┴──────────────────────┘
```

---

## 2. LoRA (Low-Rank Adaptation)

### 2.1 Mathematical Principle

```
Basic Idea:
- Weight update ΔW can be approximated as low-rank
- ΔW = BA, where B ∈ R^(d×r), A ∈ R^(r×k)
- r << min(d, k)

Forward pass:
h = W₀x + ΔWx = W₀x + BAx

Trainable parameters:
- W₀: frozen
- A, B: trainable
- Parameter count: r(d + k) vs dk (r << min(d,k))

Example (d=4096, k=4096, r=8):
- Full: 16.7M params
- LoRA: 65K params (0.4%)
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LoRALayer(nn.Module):
    """LoRA Layer"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LoRA delta: BA * scaling"""
        return self.scaling * (self.dropout(x) @ self.lora_A.T @ self.lora_B.T)


class LinearWithLoRA(nn.Module):
    """Linear layer with LoRA applied"""

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank, alpha, dropout
        )

        # Original weights frozen
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)

    def merge_weights(self):
        """Merge LoRA weights into original"""
        with torch.no_grad():
            self.linear.weight += (
                self.lora.lora_B @ self.lora.lora_A
            ) * self.lora.scaling


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: list = ["q_proj", "v_proj"]
):
    """Apply LoRA to model"""
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Find parent module
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model

                # Replace with LoRA
                lora_linear = LinearWithLoRA(module, rank, alpha)
                setattr(parent, child_name, lora_linear)

    return model
```

### 2.2 QLoRA (Quantized LoRA)

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def setup_qlora(model_name: str, rank: int = 64):
    """QLoRA setup"""

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Normal Float 4
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True  # Double quantization
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Prepare for kbit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # Check trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {all_params:,} ({100*trainable_params/all_params:.2f}%)")

    return model
```

### 2.3 DoRA (Weight-Decomposed Low-Rank Adaptation)

```python
class DoRALayer(nn.Module):
    """
    DoRA: Weight = m * (W + BA) / ||W + BA||

    Decompose weight into magnitude and direction
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0
    ):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank

        # LoRA components
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Magnitude vector (learnable)
        self.magnitude = nn.Parameter(torch.ones(out_features))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(
        self,
        x: torch.Tensor,
        original_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        W' = m * (W + ΔW) / ||W + ΔW||
        """
        # ΔW = B @ A
        delta_w = (self.lora_B @ self.lora_A) * self.scaling

        # W + ΔW
        adapted_weight = original_weight + delta_w

        # Normalize direction
        weight_norm = adapted_weight.norm(dim=1, keepdim=True)
        normalized_weight = adapted_weight / (weight_norm + 1e-8)

        # Apply magnitude
        final_weight = self.magnitude.unsqueeze(1) * normalized_weight

        return F.linear(x, final_weight)
```

---

## 3. Adapter Methods

### 3.1 Bottleneck Adapters

```
Transformer Block with Adapter:
┌────────────────────────────────────────┐
│  Multi-Head Attention                  │
│           ↓                            │
│  ┌──────────────────────────────────┐  │
│  │  Adapter (bottleneck)            │  │
│  │  Linear(d → r) → GELU            │  │
│  │  Linear(r → d) + residual        │  │
│  └──────────────────────────────────┘  │
│           ↓                            │
│  Feed-Forward Network                  │
│           ↓                            │
│  ┌──────────────────────────────────┐  │
│  │  Adapter (bottleneck)            │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
```

```python
class Adapter(nn.Module):
    """Bottleneck Adapter"""

    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int,
        adapter_scalar: float = 1.0
    ):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        self.act = nn.GELU()
        self.scalar = adapter_scalar

        # Initialization: near-identity
        nn.init.normal_(self.down_proj.weight, std=1e-3)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, std=1e-3)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down_proj(x)
        x = self.act(x)
        x = self.up_proj(x)
        return residual + self.scalar * x
```

### 3.2 IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)

```python
class IA3Layer(nn.Module):
    """
    IA³: Uses only learnable scaling vectors
    - Element-wise multiplication on key, value, ffn outputs
    - Very few parameters
    """

    def __init__(self, dim: int):
        super().__init__()
        # Learnable scaling vectors
        self.l_k = nn.Parameter(torch.ones(dim))  # key scaling
        self.l_v = nn.Parameter(torch.ones(dim))  # value scaling
        self.l_ff = nn.Parameter(torch.ones(dim))  # ffn scaling

    def scale_key(self, k: torch.Tensor) -> torch.Tensor:
        return k * self.l_k

    def scale_value(self, v: torch.Tensor) -> torch.Tensor:
        return v * self.l_v

    def scale_ffn(self, h: torch.Tensor) -> torch.Tensor:
        return h * self.l_ff
```

---

## 4. Prompt-based Methods

### 4.1 Prefix Tuning

```
┌────────────────────────────────────────────────────────────┐
│  Prefix Tuning                                             │
│                                                            │
│  Input: [P₁, P₂, ..., Pₘ, x₁, x₂, ..., xₙ]                │
│                                                            │
│  - Pᵢ: learnable prefix tokens (as key/value in each layer)│
│  - xᵢ: actual input tokens                                │
│                                                            │
│  Attention:                                                │
│  softmax(Q · [P_keys; X_keys]ᵀ) · [P_values; X_values]    │
└────────────────────────────────────────────────────────────┘
```

```python
class PrefixTuning(nn.Module):
    """Prefix Tuning"""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        prefix_length: int = 10,
        hidden_size: int = 512
    ):
        super().__init__()
        self.prefix_length = prefix_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Prefix embeddings (through MLP for stability)
        self.prefix_embedding = nn.Embedding(prefix_length, hidden_size)

        # Layer-specific projections
        self.prefix_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_layers * 2 * num_heads * head_dim)
        )

    def forward(self, batch_size: int) -> tuple:
        """
        Returns:
            prefix_keys: (num_layers, batch_size, num_heads, prefix_len, head_dim)
            prefix_values: (num_layers, batch_size, num_heads, prefix_len, head_dim)
        """
        # Prefix indices
        prefix_idx = torch.arange(self.prefix_length)
        prefix_embed = self.prefix_embedding(prefix_idx)  # (prefix_len, hidden)

        # Project to key/value pairs for all layers
        prefix_kv = self.prefix_mlp(prefix_embed)  # (prefix_len, num_layers*2*num_heads*head_dim)

        # Reshape
        prefix_kv = prefix_kv.view(
            self.prefix_length,
            self.num_layers, 2,
            self.num_heads, self.head_dim
        )
        prefix_kv = prefix_kv.permute(1, 2, 0, 3, 4)  # (layers, 2, prefix, heads, dim)

        # Expand for batch
        prefix_keys = prefix_kv[:, 0].unsqueeze(1).expand(-1, batch_size, -1, -1, -1)
        prefix_values = prefix_kv[:, 1].unsqueeze(1).expand(-1, batch_size, -1, -1, -1)

        return prefix_keys, prefix_values
```

### 4.2 Prompt Tuning

```python
class PromptTuning(nn.Module):
    """
    Prompt Tuning: Add soft prompts to input

    Simple but effective (especially for large models)
    """

    def __init__(
        self,
        num_tokens: int,
        embed_dim: int,
        init_from_vocab: bool = False,
        vocab_embeddings: Optional[nn.Embedding] = None
    ):
        super().__init__()
        self.num_tokens = num_tokens

        # Soft prompt embeddings
        self.prompt_embeddings = nn.Parameter(torch.zeros(num_tokens, embed_dim))

        if init_from_vocab and vocab_embeddings is not None:
            # Initialize from actual tokens
            indices = torch.randint(0, vocab_embeddings.num_embeddings, (num_tokens,))
            self.prompt_embeddings.data = vocab_embeddings.weight[indices].clone()
        else:
            nn.init.normal_(self.prompt_embeddings, std=0.02)

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_embeddings: (batch, seq_len, embed_dim)

        Returns:
            (batch, prompt_len + seq_len, embed_dim)
        """
        batch_size = input_embeddings.shape[0]

        # Expand prompt for batch
        prompt = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate
        return torch.cat([prompt, input_embeddings], dim=1)
```

---

## 5. Using HuggingFace PEFT

```python
from peft import (
    LoraConfig, PrefixTuningConfig, PromptTuningConfig,
    get_peft_model, TaskType
)
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

def setup_peft_training(
    model_name: str,
    method: str = "lora",
    output_dir: str = "./output"
):
    """Setup various PEFT methods"""

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # PEFT configuration
    if method == "lora":
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
    elif method == "prefix":
        peft_config = PrefixTuningConfig(
            num_virtual_tokens=20,
            task_type=TaskType.CAUSAL_LM
        )
    elif method == "prompt":
        peft_config = PromptTuningConfig(
            num_virtual_tokens=20,
            prompt_tuning_init="TEXT",
            prompt_tuning_init_text="Classify the sentiment of this text: ",
            tokenizer_name_or_path=model_name,
            task_type=TaskType.CAUSAL_LM
        )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train_with_peft(model, tokenizer, train_dataset):
    """Train PEFT model"""
    training_args = TrainingArguments(
        output_dir="./peft-output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    trainer.train()

    # Save adapter (original model not needed)
    model.save_pretrained("./peft-adapter")


def load_and_merge_adapter(base_model_name: str, adapter_path: str):
    """Load and merge adapter"""
    from peft import PeftModel

    # Base model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # Load adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Merge (for inference speed improvement)
    merged_model = model.merge_and_unload()

    return merged_model
```

---

## 6. Method Comparison

### 6.1 Parameter Efficiency

| Method | Trainable Params (7B model) | Memory Overhead |
|--------|------------------------|----------------|
| Full FT | 7B (100%) | ~84GB |
| LoRA (r=8) | ~4M (0.06%) | ~200MB |
| LoRA (r=64) | ~30M (0.4%) | ~1GB |
| QLoRA (r=64) | ~30M | ~6GB (4bit base) |
| Prefix Tuning | ~1M | ~100MB |
| Prompt Tuning | ~100K | ~10MB |
| IA³ | ~300K | ~30MB |

### 6.2 Performance Comparison

```
General performance ranking (downstream tasks):

Full FT > LoRA ≈ QLoRA > Adapters > Prefix > Prompt

However, varies by model size and task:
- Large models (>10B): Prompt Tuning is also effective
- Small models (<1B): LoRA/Adapters recommended
- Memory constraints: QLoRA essential
```

### 6.3 Selection Guide

```python
def recommend_peft_method(
    model_size_b: float,  # Model size (billions)
    gpu_memory_gb: float,  # GPU memory (GB)
    task_type: str,  # "classification", "generation", "qa"
    num_examples: int  # Training data count
) -> str:
    """Recommend PEFT method"""

    # Memory-based decision
    if gpu_memory_gb < model_size_b * 2:
        # 4-bit quantization needed
        return "QLoRA"

    # Data size-based
    if num_examples < 1000:
        # Small data: Prompt Tuning
        if model_size_b > 10:
            return "Prompt Tuning"
        else:
            return "LoRA (small rank)"

    # General case
    if task_type == "classification":
        return "LoRA or Adapters"
    elif task_type == "generation":
        return "LoRA (target all projections)"
    else:
        return "LoRA"
```

---

## Key Summary

### PEFT Core Concepts
```
1. LoRA: Low-rank update with W + BA
2. QLoRA: 4-bit quantization + LoRA
3. DoRA: Magnitude/direction separation
4. Adapters: Add bottleneck modules
5. Prefix: Learnable key/value prefix
6. Prompt: Soft prompt embeddings
7. IA³: Train only scaling vectors
```

### Practical Points
```
- GPU shortage → Use QLoRA
- Inference speed critical → merge_and_unload()
- Multiple tasks → Save/load adapters separately
- Large model + small data → Prompt Tuning
```

---

## References

1. Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
2. Dettmers et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs"
3. Liu et al. (2024). "DoRA: Weight-Decomposed Low-Rank Adaptation"
4. Houlsby et al. (2019). "Parameter-Efficient Transfer Learning for NLP"
5. [HuggingFace PEFT](https://github.com/huggingface/peft)

---

## Exercises

### Exercise 1: LoRA Trainable Parameter Calculation
A LLaMA-7B model has attention layers with query projection weight matrices of shape (4096, 4096). Calculate the number of trainable parameters for LoRA applied to these matrices at rank r=8 vs. r=64, and compare to full fine-tuning. Then compute the parameter reduction ratio.

```python
# Given:
# - d_model = 4096
# - num_attention_layers = 32
# - LoRA applied to q_proj and v_proj only
# - LoRA matrices: W_A (d x r) and W_B (r x d)

# Full fine-tuning params for q_proj + v_proj across all layers:
full_ft_params = ???

# LoRA params at r=8:
lora_r8_params = ???

# LoRA params at r=64:
lora_r64_params = ???

# Reduction ratio for r=8:
reduction_ratio = ???
```

<details>
<summary>Show Answer</summary>

```python
d_model = 4096
num_layers = 32

# Full fine-tuning: q_proj + v_proj per layer
# Each weight matrix: 4096 × 4096 = 16,777,216 params
# 2 matrices (q + v) × 32 layers
full_ft_params = 2 * d_model * d_model * num_layers
               = 2 * 4096 * 4096 * 32
               = 1,073,741,824 ≈ 1.07B params

# LoRA at r=8:
# Each LoRA: W_A (4096×8) + W_B (8×4096) = 32,768 + 32,768 = 65,536 params
# 2 matrices × 32 layers
lora_r8_params = 2 * (d_model * 8 + 8 * d_model) * num_layers
               = 2 * (32768 + 32768) * 32
               = 2 * 65,536 * 32
               = 4,194,304 ≈ 4.2M params

# LoRA at r=64:
lora_r64_params = 2 * (d_model * 64 + 64 * d_model) * num_layers
                = 2 * (262,144 + 262,144) * 32
                = 33,554,432 ≈ 33.6M params

# Reduction ratios:
reduction_r8 = full_ft_params / lora_r8_params
             = 1,073,741,824 / 4,194,304
             ≈ 256x fewer trainable params (0.39% of full FT)

reduction_r64 = full_ft_params / lora_r64_params
              = 1,073,741,824 / 33,554,432
              ≈ 32x fewer trainable params (3.1% of full FT)

# Summary:
# r=8: 4.2M params (0.06% of 7B model total params = 7B × q+v only ≈ 2B)
# r=64: 33.6M params (0.48% of 7B model)
# Both dramatically reduce memory for optimizer states + gradients
```

Note: At r=8, the LoRA adapter adds only ~4.2M parameters — comparable to a tiny 2-layer MLP, yet enables significant task adaptation. The `lora_alpha` hyperparameter (scaling factor = alpha/r) determines the effective learning rate for these matrices relative to the frozen weights.

</details>

### Exercise 2: QLoRA Memory Analysis
QLoRA combines 4-bit quantization (NF4 format) with LoRA adapters. For a LLaMA-13B model, calculate the approximate memory usage for (A) full fine-tuning in FP16, (B) LoRA in BF16, and (C) QLoRA with NF4 base + BF16 adapters.

| Component | Full FT (FP16) | LoRA (BF16) | QLoRA (NF4 + BF16) |
|-----------|----------------|-------------|---------------------|
| Model weights | ??? | ??? | ??? |
| Optimizer states (Adam) | ??? | ??? | ??? |
| Gradients | ??? | N/A (frozen) | N/A (frozen) |
| LoRA adapters + optimizer | N/A | ??? | ??? |
| **Total** | **???** | **???** | **???** |

<details>
<summary>Show Answer</summary>

Assumptions: 13B parameters, LoRA r=8 on q+v proj (32 layers) ≈ 8M trainable params.

| Component | Full FT (FP16) | LoRA (BF16) | QLoRA (NF4 + BF16) |
|-----------|----------------|-------------|---------------------|
| Model weights | 13B × 2B/param = **26GB** | 13B × 2B/param = **26GB** | 13B × 0.5B/param* = **6.5GB** |
| Optimizer states (Adam: 2× FP32) | 13B × 8B/param = **104GB** | N/A (frozen) | N/A (frozen) |
| Gradients (FP16) | 13B × 2B/param = **26GB** | N/A (frozen) | N/A (frozen) |
| LoRA adapters (BF16) | N/A | 8M × 2B/param = **16MB** | 8M × 2B/param = **16MB** |
| LoRA optimizer (Adam, FP32) | N/A | 8M × 8B/param = **64MB** | 8M × 8B/param = **64MB** |
| **Total** | **~156GB** | **~26GB** | **~6.6GB** |

*NF4 = 4-bit quantization = 0.5 bytes/parameter

**Analysis**:
- Full FT requires ~156GB → needs 2× A100 80GB minimum
- LoRA BF16 requires ~26GB → fits on single A100 80GB (with activations)
- QLoRA NF4 requires ~6.6GB → fits on a single consumer GPU (RTX 3090/4090 24GB)!

This is why QLoRA was revolutionary: it enabled fine-tuning 13B+ models on hardware previously used only for inference. The key insight is that quantization is applied only to the frozen base model weights — the LoRA adapters remain in BF16 for training stability.

</details>

### Exercise 3: LoRA Rank Selection
You are fine-tuning a 7B LLM for three different tasks. For each task, select the appropriate LoRA rank and justify your choice. Consider the task complexity, data size, and desired behavior.

| Task | Training Data | Target Behavior | Recommended Rank | Justification |
|------|--------------|-----------------|-----------------|---------------|
| A) Translate tech docs EN→FR | 50K sentence pairs | Precise translation | ??? | ??? |
| B) Learn a new proprietary API style | 200 examples | Code generation in custom style | ??? | ??? |
| C) General instruction following improvement | 100K diverse examples | Better general assistant | ??? | ??? |

<details>
<summary>Show Answer</summary>

| Task | Recommended Rank | Justification |
|------|-----------------|---------------|
| A) Translation EN→FR | **r=4 or r=8** | Translation is a well-defined task that the base model already partially knows (French is in pretraining data). Low rank captures the fine-grained alignment needed without over-fitting the specific domain vocabulary. Higher rank risks memorizing corpus-specific phrases. 50K pairs is substantial — low rank generalizes better. |
| B) Proprietary API style | **r=16 or r=32** | Learning a completely new, proprietary API style requires capturing specific syntactic patterns the base model has never seen. With only 200 examples, we need sufficient rank to express novel code patterns. But we should also use a higher `lora_dropout` (0.1-0.2) to prevent over-fitting with so little data. |
| C) General instruction following | **r=64 or r=128** | Improving general instruction following requires broad behavioral changes across many task types — following format instructions, chain-of-thought reasoning, refusing harmful requests. This requires higher-rank adapters to express diverse behavioral patterns. With 100K diverse examples, there's enough data to support higher rank without over-fitting. |

**General heuristic**:
- r=1-4: Very targeted style or format adjustment
- r=8-16: Single task domain adaptation (most common default)
- r=32-64: Multi-task or complex behavioral changes
- r=128-256: Near-full-fine-tuning capability needed

The `lora_alpha` parameter should usually be set to 2×r (alpha=16 for r=8) as a starting point.

</details>

### Exercise 4: Adapter vs. LoRA: Inference Speed Trade-off
Both Adapters (bottleneck modules) and LoRA add parameters for fine-tuning, but they have different inference time characteristics. Explain why LoRA can be "merged" into the base model for zero inference overhead, while Adapters cannot, and describe the formula for merging.

```python
# LoRA weight merging
class MergeableLoRA:
    def merge_weights(self):
        """
        Merge LoRA into base weight for inference
        Original: y = Wx + BAx (two sequential operations)
        Merged:   y = (W + BA)x (one operation, same result)
        """
        # W: original frozen weight (d × d)
        # B: LoRA B matrix (d × r)
        # A: LoRA A matrix (r × d)
        # scaling: alpha / rank

        W_merged = self.W + (self.lora_B @ self.lora_A) * (self.alpha / self.rank)
        return W_merged

# Why can't Adapters do this?
```

<details>
<summary>Show Answer</summary>

**Why LoRA can be merged**:

LoRA adds a residual path: `y = W·x + (B·A)·x = (W + B·A)·x`

This is algebraically equivalent to a single matrix multiplication with the merged weight `W' = W + scaling × (B·A)`. The key property: both the original path and the LoRA path are **linear operations applied to the same input** at the same point in the network. Simple matrix addition combines them:

```python
# Before merging: 2 sequential operations
y_lora = W @ x + (lora_B @ lora_A) * (alpha / rank) @ x

# After merging: 1 operation (same numerical result)
W_merged = W + (lora_B @ lora_A) * (alpha / rank)  # done once offline
y_merged = W_merged @ x  # at inference time
```

**Why Adapters cannot be merged**:

Adapter modules are **non-linear bottleneck networks** inserted sequentially in the computation graph:

```
x → LayerNorm → Down-project → Activation → Up-project → Add residual → y
```

The adapter includes a non-linear activation function (typically GELU or ReLU) between the down and up projections. Non-linear operations cannot be collapsed into a single linear weight matrix. There is no way to express `Up(GELU(Down(x))) + x` as a single matrix multiplication `W_merged · x`.

**Inference overhead comparison**:
- **LoRA (merged)**: Zero extra computation — identical to the original base model inference
- **LoRA (unmerged)**: +2 matrix multiplications per LoRA layer per token
- **Adapters**: Sequential forward pass through bottleneck (down + activation + up), cannot be skipped
- **Prefix Tuning**: Extends the K/V sequence length, increasing attention computation proportionally

LoRA's mergeability is a key practical advantage: you can train efficiently (small adapter) and deploy at full base model speed.

</details>

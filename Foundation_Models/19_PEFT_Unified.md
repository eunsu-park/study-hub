# 19. PEFT (Parameter-Efficient Fine-Tuning) 통합

## 개요

PEFT 방법론들은 전체 모델 대신 작은 파라미터 세트만 학습하여 효율적인 적응을 가능하게 합니다. 이 레슨에서는 다양한 PEFT 기법들을 통합적으로 다룹니다.

---

## 1. PEFT 개요

### 1.1 왜 PEFT인가?

```
Full Fine-tuning의 문제점:
┌─────────────────────────────────────┐
│  LLaMA-7B                           │
│  - 파라미터: 7B                      │
│  - FP16 메모리: 14GB                │
│  - Optimizer states: 56GB          │
│  - Gradients: 14GB                  │
│  - Total: ~84GB                     │
└─────────────────────────────────────┘

PEFT의 장점:
┌─────────────────────────────────────┐
│  LoRA (rank=8)                      │
│  - 학습 파라미터: ~0.1%             │
│  - 추가 메모리: ~100MB              │
│  - 성능: Full FT의 90-95%           │
│  - 스토리지: 원본 + 작은 adapter    │
└─────────────────────────────────────┘
```

### 1.2 PEFT 방법론 분류

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

### 2.1 수학적 원리

```
기본 아이디어:
- Weight 업데이트 ΔW는 low-rank로 근사 가능
- ΔW = BA, where B ∈ R^(d×r), A ∈ R^(r×k)
- r << min(d, k)

Forward pass:
h = W₀x + ΔWx = W₀x + BAx

학습 파라미터:
- W₀: frozen
- A, B: trainable
- 파라미터 수: r(d + k) vs dk (r << min(d,k))

예시 (d=4096, k=4096, r=8):
- Full: 16.7M params
- LoRA: 65K params (0.4%)
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LoRALayer(nn.Module):
    """LoRA 레이어"""

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

        # 초기화
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LoRA delta: BA * scaling"""
        return self.scaling * (self.dropout(x) @ self.lora_A.T @ self.lora_B.T)


class LinearWithLoRA(nn.Module):
    """LoRA가 적용된 Linear 레이어"""

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
        """LoRA weights를 original에 병합"""
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
    """모델에 LoRA 적용"""
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # 부모 모듈 찾기
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model

                # LoRA로 교체
                lora_linear = LinearWithLoRA(module, rank, alpha)
                setattr(parent, child_name, lora_linear)

    return model
```

### 2.2 QLoRA (Quantized LoRA)

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def setup_qlora(model_name: str, rank: int = 64):
    """QLoRA 설정"""

    # 4-bit 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Normal Float 4
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True  # 이중 양자화
    )

    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # kbit 학습 준비
    model = prepare_model_for_kbit_training(model)

    # LoRA 설정
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

    # 학습 가능한 파라미터 확인
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

    Weight를 magnitude와 direction으로 분해
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

        # 초기화: near-identity
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
    IA³: 학습 가능한 scaling vectors만 사용
    - key, value, ffn 출력에 element-wise 곱
    - 매우 적은 파라미터
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
│  - Pᵢ: learnable prefix tokens (각 layer에서 key/value로)  │
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
    Prompt Tuning: 입력에 soft prompt 추가

    단순하지만 효과적 (특히 대형 모델에서)
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
            # 실제 토큰으로 초기화
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

## 5. HuggingFace PEFT 사용

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
    """다양한 PEFT 방법 설정"""

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # PEFT 설정
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
    """PEFT 모델 학습"""
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

    # Adapter 저장 (원본 모델 불필요)
    model.save_pretrained("./peft-adapter")


def load_and_merge_adapter(base_model_name: str, adapter_path: str):
    """Adapter 로드 및 병합"""
    from peft import PeftModel

    # Base model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # Adapter 로드
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # 병합 (추론 속도 향상)
    merged_model = model.merge_and_unload()

    return merged_model
```

---

## 6. 방법론 비교

### 6.1 파라미터 효율성

| 방법 | 학습 파라미터 (7B 모델) | 메모리 오버헤드 |
|------|------------------------|----------------|
| Full FT | 7B (100%) | ~84GB |
| LoRA (r=8) | ~4M (0.06%) | ~200MB |
| LoRA (r=64) | ~30M (0.4%) | ~1GB |
| QLoRA (r=64) | ~30M | ~6GB (4bit base) |
| Prefix Tuning | ~1M | ~100MB |
| Prompt Tuning | ~100K | ~10MB |
| IA³ | ~300K | ~30MB |

### 6.2 성능 비교

```
일반적인 성능 순위 (downstream tasks):

Full FT > LoRA ≈ QLoRA > Adapters > Prefix > Prompt

단, 모델 크기와 태스크에 따라 다름:
- 대형 모델 (>10B): Prompt Tuning도 효과적
- 소형 모델 (<1B): LoRA/Adapters 권장
- 메모리 제약: QLoRA 필수
```

### 6.3 선택 가이드

```python
def recommend_peft_method(
    model_size_b: float,  # 모델 크기 (billions)
    gpu_memory_gb: float,  # GPU 메모리 (GB)
    task_type: str,  # "classification", "generation", "qa"
    num_examples: int  # 학습 데이터 수
) -> str:
    """PEFT 방법 추천"""

    # 메모리 기반 결정
    if gpu_memory_gb < model_size_b * 2:
        # 4-bit 양자화 필요
        return "QLoRA"

    # 데이터 크기 기반
    if num_examples < 1000:
        # 적은 데이터: Prompt Tuning
        if model_size_b > 10:
            return "Prompt Tuning"
        else:
            return "LoRA (small rank)"

    # 일반적인 경우
    if task_type == "classification":
        return "LoRA or Adapters"
    elif task_type == "generation":
        return "LoRA (target all projections)"
    else:
        return "LoRA"
```

---

## 핵심 정리

### PEFT 핵심 개념
```
1. LoRA: W + BA로 low-rank 업데이트
2. QLoRA: 4-bit 양자화 + LoRA
3. DoRA: magnitude/direction 분리
4. Adapters: bottleneck 모듈 추가
5. Prefix: learnable key/value prefix
6. Prompt: soft prompt embeddings
7. IA³: scaling vectors만 학습
```

### 실용 포인트
```
- GPU 부족 → QLoRA 사용
- 추론 속도 중요 → merge_and_unload()
- 여러 태스크 → adapter별 저장/로드
- 대형 모델 + 적은 데이터 → Prompt Tuning
```

---

## 참고 자료

1. Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
2. Dettmers et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs"
3. Liu et al. (2024). "DoRA: Weight-Decomposed Low-Rank Adaptation"
4. Houlsby et al. (2019). "Parameter-Efficient Transfer Learning for NLP"
5. [HuggingFace PEFT](https://github.com/huggingface/peft)

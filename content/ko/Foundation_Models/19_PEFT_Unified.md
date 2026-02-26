# 19. PEFT (Parameter-Efficient Fine-Tuning) 통합

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 파라미터 효율적 파인튜닝(Parameter-Efficient Fine-Tuning, PEFT)이 필요한 이유를 설명하고 풀 파인튜닝(full fine-tuning) 대비 메모리 및 스토리지 절감 효과를 수치로 제시할 수 있다
2. LoRA(Low-Rank Adaptation)의 수학적 원리를 도출하고 Hugging Face PEFT 라이브러리를 사용하여 구현할 수 있다
3. 첨가형(additive), 재파라미터화(reparameterization), 선택적(selective) PEFT 전략을 비교하고 학습 가능 파라미터 수와 다운스트림 성능 간 트레이드오프를 평가할 수 있다
4. 양자화(quantization)와 저랭크 어댑터(low-rank adapter)를 결합한 QLoRA를 적용하여 일반 소비자용 하드웨어에서 대형 언어 모델을 파인튜닝할 수 있다
5. 태스크 요구사항과 하드웨어 제약에 따라 LoRA, Prefix Tuning, Prompt Tuning 중 적절한 PEFT 방법을 선택하는 학습 파이프라인을 설계할 수 있다

---

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

---

## 연습 문제

### 연습 문제 1: LoRA 훈련 파라미터 수 계산
LLaMA-7B 모델의 어텐션 레이어(attention layer)에는 형상(shape) (4096, 4096)인 쿼리 프로젝션(query projection) 가중치 행렬이 있습니다. 랭크(rank) r=8과 r=64에서 이 행렬에 적용된 LoRA의 훈련 가능 파라미터 수를 계산하고, 전체 파인튜닝(full fine-tuning)과 비교하세요. 그런 다음 파라미터 감소 비율을 계산하세요.

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
<summary>정답 보기</summary>

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

참고: r=8에서 LoRA 어댑터는 약 420만 개의 파라미터만 추가합니다 — 작은 2층 MLP와 비슷하지만, 상당한 태스크 적응을 가능하게 합니다. `lora_alpha` 하이퍼파라미터(스케일링 계수 = alpha/r)는 동결된 가중치에 대한 이러한 행렬들의 효과적인 학습률을 결정합니다.

</details>

### 연습 문제 2: QLoRA 메모리 분석
QLoRA는 4비트 양자화(NF4 포맷)와 LoRA 어댑터를 결합합니다. LLaMA-13B 모델에 대해 (A) FP16 전체 파인튜닝, (B) BF16 LoRA, (C) NF4 베이스 + BF16 어댑터를 사용하는 QLoRA의 대략적인 메모리 사용량을 계산하세요.

| 구성 요소 | 전체 FT (FP16) | LoRA (BF16) | QLoRA (NF4 + BF16) |
|-----------|----------------|-------------|---------------------|
| 모델 가중치 | ??? | ??? | ??? |
| 옵티마이저 상태 (Adam) | ??? | ??? | ??? |
| 그래디언트 | ??? | N/A (동결) | N/A (동결) |
| LoRA 어댑터 + 옵티마이저 | N/A | ??? | ??? |
| **합계** | **???** | **???** | **???** |

<details>
<summary>정답 보기</summary>

가정: 130억 파라미터, 32개 레이어 q+v proj에 LoRA r=8 적용 ≈ 800만 훈련 파라미터.

| 구성 요소 | 전체 FT (FP16) | LoRA (BF16) | QLoRA (NF4 + BF16) |
|-----------|----------------|-------------|---------------------|
| 모델 가중치 | 13B × 2B/param = **26GB** | 13B × 2B/param = **26GB** | 13B × 0.5B/param* = **6.5GB** |
| 옵티마이저 상태 (Adam: 2× FP32) | 13B × 8B/param = **104GB** | N/A (동결) | N/A (동결) |
| 그래디언트 (FP16) | 13B × 2B/param = **26GB** | N/A (동결) | N/A (동결) |
| LoRA 어댑터 (BF16) | N/A | 8M × 2B/param = **16MB** | 8M × 2B/param = **16MB** |
| LoRA 옵티마이저 (Adam, FP32) | N/A | 8M × 8B/param = **64MB** | 8M × 8B/param = **64MB** |
| **합계** | **~156GB** | **~26GB** | **~6.6GB** |

*NF4 = 4비트 양자화 = 파라미터당 0.5바이트

**분석**:
- 전체 FT는 ~156GB 필요 → A100 80GB 최소 2개 필요
- LoRA BF16은 ~26GB 필요 → 단일 A100 80GB에 적합 (활성화 포함)
- QLoRA NF4는 ~6.6GB 필요 → 단일 소비자 GPU (RTX 3090/4090 24GB)에 적합!

QLoRA가 혁신적이었던 이유: 이전에는 추론에만 사용되던 하드웨어에서 13B+ 모델의 파인튜닝을 가능하게 했습니다. 핵심 통찰은 양자화가 동결된 기반 모델 가중치에만 적용되고, LoRA 어댑터는 훈련 안정성을 위해 BF16으로 유지된다는 것입니다.

</details>

### 연습 문제 3: LoRA 랭크 선택
세 가지 서로 다른 태스크에 대해 7B LLM을 파인튜닝하고 있습니다. 각 태스크에 적합한 LoRA 랭크를 선택하고 그 이유를 설명하세요. 태스크 복잡도, 데이터 크기, 원하는 동작을 고려하세요.

| 태스크 | 훈련 데이터 | 목표 동작 | 권장 랭크 | 근거 |
|------|--------------|-----------------|-----------------|---------------|
| A) 기술 문서 EN→FR 번역 | 5만 문장 쌍 | 정확한 번역 | ??? | ??? |
| B) 새로운 독점 API 스타일 학습 | 200개 예제 | 커스텀 스타일의 코드 생성 | ??? | ??? |
| C) 일반 지시 따르기 개선 | 10만 건의 다양한 예제 | 더 나은 범용 어시스턴트 | ??? | ??? |

<details>
<summary>정답 보기</summary>

| 태스크 | 권장 랭크 | 근거 |
|------|-----------------|---------------|
| A) EN→FR 번역 | **r=4 또는 r=8** | 번역은 기반 모델이 이미 부분적으로 알고 있는 잘 정의된 태스크입니다(프랑스어는 사전 훈련 데이터에 포함). 낮은 랭크는 특정 도메인 어휘에 과적합하지 않으면서 필요한 세밀한 정렬을 포착합니다. 높은 랭크는 코퍼스 특화 문구를 암기할 위험이 있습니다. 5만 쌍은 상당한 양으로 — 낮은 랭크가 더 잘 일반화합니다. |
| B) 독점 API 스타일 | **r=16 또는 r=32** | 완전히 새로운 독점 API 스타일을 학습하려면 기반 모델이 한 번도 본 적 없는 특정 구문 패턴을 포착해야 합니다. 200개 예제만 있으므로 새로운 코드 패턴을 표현할 충분한 랭크가 필요합니다. 하지만 데이터가 너무 적어 과적합을 방지하기 위해 높은 `lora_dropout`(0.1-0.2)도 사용해야 합니다. |
| C) 일반 지시 따르기 | **r=64 또는 r=128** | 일반적인 지시 따르기를 개선하려면 많은 태스크 유형에 걸쳐 광범위한 동작 변화가 필요합니다 — 형식 지시 따르기, 사고 연쇄(chain-of-thought) 추론, 유해 요청 거절. 이는 다양한 동작 패턴을 표현하기 위해 더 높은 랭크의 어댑터가 필요합니다. 10만 건의 다양한 예제로 과적합 없이 높은 랭크를 지원할 수 있습니다. |

**일반적인 경험칙**:
- r=1-4: 매우 특정한 스타일이나 형식 조정
- r=8-16: 단일 태스크 도메인 적응 (가장 일반적인 기본값)
- r=32-64: 멀티태스크 또는 복잡한 동작 변화
- r=128-256: 전체 파인튜닝에 준하는 역량 필요 시

`lora_alpha` 파라미터는 일반적으로 시작점으로 2×r로 설정합니다 (r=8이면 alpha=16).

</details>

### 연습 문제 4: 어댑터(Adapter) vs. LoRA: 추론 속도 트레이드오프
어댑터(병목 모듈)와 LoRA 모두 파인튜닝을 위한 파라미터를 추가하지만, 추론 시간 특성이 다릅니다. LoRA는 기반 모델에 "병합(merge)"되어 추론 오버헤드가 없을 수 있지만 어댑터는 그렇지 않은 이유를 설명하고, 병합 공식을 서술하세요.

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
<summary>정답 보기</summary>

**LoRA가 병합 가능한 이유**:

LoRA는 잔차 경로(residual path)를 추가합니다: `y = W·x + (B·A)·x = (W + B·A)·x`

이는 병합된 가중치 `W' = W + scaling × (B·A)`를 사용하는 단일 행렬 곱셈과 대수적으로 동등합니다. 핵심 속성: 원래 경로와 LoRA 경로 모두 네트워크의 같은 지점에서 **동일한 입력에 적용되는 선형 연산**입니다. 단순한 행렬 덧셈으로 결합됩니다:

```python
# Before merging: 2 sequential operations
y_lora = W @ x + (lora_B @ lora_A) * (alpha / rank) @ x

# After merging: 1 operation (same numerical result)
W_merged = W + (lora_B @ lora_A) * (alpha / rank)  # done once offline
y_merged = W_merged @ x  # at inference time
```

**어댑터가 병합될 수 없는 이유**:

어댑터 모듈은 계산 그래프에 순차적으로 삽입된 **비선형 병목 네트워크(non-linear bottleneck network)**입니다:

```
x → LayerNorm → Down-project → Activation → Up-project → Add residual → y
```

어댑터는 다운 프로젝션과 업 프로젝션 사이에 비선형 활성화 함수(일반적으로 GELU 또는 ReLU)를 포함합니다. 비선형 연산은 단일 선형 가중치 행렬로 축약될 수 없습니다. `Up(GELU(Down(x))) + x`를 단일 행렬 곱셈 `W_merged · x`로 표현할 방법이 없습니다.

**추론 오버헤드 비교**:
- **LoRA (병합됨)**: 추가 계산 없음 — 원래 기반 모델 추론과 동일
- **LoRA (비병합)**: LoRA 레이어당 토큰당 +2번의 행렬 곱셈
- **어댑터**: 병목을 통한 순차적 순전파(down + activation + up), 건너뛸 수 없음
- **프리픽스 튜닝(Prefix Tuning)**: K/V 시퀀스 길이를 늘려 어텐션 연산을 비례적으로 증가시킴

LoRA의 병합 가능성은 핵심적인 실용적 장점입니다: 효율적으로 훈련(작은 어댑터)하고 전체 기반 모델 속도로 배포할 수 있습니다.

</details>
5. [HuggingFace PEFT](https://github.com/huggingface/peft)

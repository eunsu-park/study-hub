# 11. Small Language Models

## 개요

대형 모델(100B+)이 화제지만, 실제 프로덕션 환경에서는 **Small Language Models (SLM)**이 더 실용적입니다. 이 레슨에서는 7B 이하 모델의 아키텍처, 학습 전략, 활용 방법을 다룹니다.

---

## 1. SLM의 중요성

### 1.1 왜 작은 모델인가?

```
┌──────────────────────────────────────────────────────────────────┐
│                   SLM vs LLM 비교                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│                    SLM (1-7B)              LLM (70B+)            │
│                                                                  │
│  💰 비용          낮음                      높음                 │
│  ⚡ 지연시간      낮음 (<100ms)             높음 (>500ms)        │
│  🖥️ 하드웨어     단일 GPU/CPU             다중 GPU 필수        │
│  📱 엣지 배포    가능                      어려움               │
│  🔒 프라이버시   온프레미스 쉬움           어려움               │
│  🎯 특화 태스크  비용 효율적               과잉                 │
│                                                                  │
│  사용 사례:                                                      │
│  - 모바일 앱 (On-device)                                        │
│  - 임베디드 시스템                                              │
│  - 고빈도 API 서비스                                            │
│  - 비용 민감한 스타트업                                         │
│  - 개인정보 보호가 중요한 도메인                                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 SLM 모델 비교

| 모델 | 파라미터 | 학습 토큰 | 특징 |
|------|----------|-----------|------|
| **Phi-3** | 3.8B | 3.3T | MS, 추론 특화 |
| **Gemma 2** | 2B / 9B | 8T | Google, 코드 강점 |
| **Qwen 2.5** | 0.5B - 7B | 18T | 다국어, 수학 |
| **Llama 3.2** | 1B / 3B | 15T | 모바일 최적화 |
| **TinyLlama** | 1.1B | 3T | 효율적 학습 |
| **StableLM 2** | 1.6B | 2T | Stability AI |
| **SmolLM** | 135M - 1.7B | 1T | HuggingFace |

---

## 2. 아키텍처 최적화

### 2.1 Phi 시리즈 (Microsoft)

```python
"""
Phi-3: "Textbooks Are All You Need" 철학

핵심 아이디어:
1. 데이터 품질 > 데이터 양
2. 합성 데이터 활용 (GPT-4로 생성)
3. 교과서급 품질의 데이터만 사용

결과: 3.8B로 GPT-3.5급 추론 능력
"""

class Phi3Config:
    """Phi-3 아키텍처 설정"""

    # Phi-3-mini (3.8B)
    hidden_size = 3072
    num_layers = 32
    num_attention_heads = 32
    num_key_value_heads = 32  # No GQA
    intermediate_size = 8192  # FFN 확장비 ~2.7x
    vocab_size = 32064
    max_position_embeddings = 4096  # 확장 가능

    # 특징
    # - SuRoPE (Scaled RoPE)
    # - LayerNorm (RMSNorm 대신)
    # - SwiGLU FFN


# Phi-3 사용 예시
from transformers import AutoModelForCausalLM, AutoTokenizer

def use_phi3():
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct"
    )

    # 추론
    messages = [
        {"role": "user", "content": "Explain the Pythagorean theorem."}
    ]

    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7
    )

    return tokenizer.decode(outputs[0])
```

### 2.2 Gemma 2 (Google)

```python
"""
Gemma 2: 효율적인 아키텍처 설계

핵심 특징:
1. Alternating Local-Global Attention
2. Soft-Capping (Logits & Attention)
3. Pre-Norm + Post-Norm hybrid
4. Knowledge Distillation from larger models
"""

class Gemma2Config:
    """Gemma 2 아키텍처"""

    # Gemma 2 2B
    hidden_size = 2304
    num_layers = 26
    num_attention_heads = 8
    num_key_value_heads = 4  # GQA 사용
    intermediate_size = 9216
    vocab_size = 256128  # 큰 vocab

    # Gemma 2 9B
    # hidden_size = 3584
    # num_layers = 42
    # num_attention_heads = 16
    # num_key_value_heads = 8


class GemmaAttentionWithSoftCap(nn.Module):
    """Gemma 2 스타일 Soft-Capping Attention"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Local vs Global attention 교대
        # 짝수 레이어: Local (sliding window)
        # 홀수 레이어: Global (full attention)
        self.is_local = (layer_idx % 2 == 0)
        self.sliding_window = 4096 if self.is_local else None

        # Soft-cap 값
        self.attn_logit_softcap = 50.0

        # Projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size // 2)  # GQA
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        batch, seq_len, _ = hidden_states.shape

        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)

        # GQA: K, V 확장
        K = K.repeat_interleave(2, dim=-1)  # 간소화
        V = V.repeat_interleave(2, dim=-1)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(Q.shape[-1])

        # Soft-capping: tanh로 범위 제한
        scores = self.attn_logit_softcap * torch.tanh(scores / self.attn_logit_softcap)

        # Sliding window mask (local attention)
        if self.is_local and self.sliding_window:
            mask = self._create_sliding_window_mask(seq_len)
            scores = scores + mask

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len) * float('-inf'),
            diagonal=1
        ).to(scores.device)
        scores = scores + causal_mask

        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)

        return self.o_proj(output)

    def _create_sliding_window_mask(self, seq_len):
        """Sliding window attention mask"""
        mask = torch.ones(seq_len, seq_len) * float('-inf')
        for i in range(seq_len):
            start = max(0, i - self.sliding_window)
            mask[i, start:i+1] = 0
        return mask
```

### 2.3 Qwen 2.5 (Alibaba)

```python
"""
Qwen 2.5: 다국어 & 수학 강점

특징:
1. 대규모 다국어 학습 (29개 언어)
2. 코드/수학 특화 데이터
3. 긴 컨텍스트 (128K)
4. 다양한 크기 (0.5B ~ 72B)
"""

class Qwen25Config:
    """Qwen 2.5 아키텍처"""

    # Qwen2.5-0.5B (가장 작은 버전)
    hidden_size = 896
    num_layers = 24
    num_attention_heads = 14
    num_key_value_heads = 2  # 효율적 GQA
    intermediate_size = 4864
    vocab_size = 151936

    # Qwen2.5-7B
    # hidden_size = 3584
    # num_layers = 28
    # num_attention_heads = 28
    # num_key_value_heads = 4


# Qwen 사용 예시
def use_qwen():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    # 다국어 테스트
    prompts = [
        "Explain machine learning in simple terms.",
        "用简单的话解释机器学习",  # 중국어
        "기계 학습을 쉽게 설명해주세요",  # 한국어
    ]

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=128)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        print("-" * 50)
```

---

## 3. 학습 전략

### 3.1 데이터 품질 vs 양

```python
"""
SLM 학습의 핵심: 고품질 데이터

Phi의 교훈:
- 웹 크롤링 데이터 (품질 낮음) < 교과서급 데이터
- 합성 데이터 (GPT-4 생성)가 효과적
- 필터링이 매우 중요
"""

class HighQualityDataPipeline:
    """고품질 데이터 파이프라인"""

    def __init__(self, quality_model):
        self.quality_model = quality_model

    def filter_data(self, texts: list, threshold: float = 0.8):
        """품질 기반 필터링"""
        filtered = []
        for text in texts:
            score = self.quality_model.score(text)
            if score > threshold:
                filtered.append(text)

        print(f"Filtered: {len(texts)} → {len(filtered)}")
        return filtered

    def generate_synthetic_data(
        self,
        teacher_model,
        topics: list,
        n_samples: int = 10000
    ):
        """합성 데이터 생성"""
        synthetic_data = []

        for topic in topics:
            prompt = f"""Create an educational explanation about {topic}.
            The explanation should be:
            1. Clear and concise
            2. Include examples
            3. Suitable for learning"""

            for _ in range(n_samples // len(topics)):
                response = teacher_model.generate(prompt)

                # 품질 검증
                if self._validate_response(response):
                    synthetic_data.append({
                        'topic': topic,
                        'content': response
                    })

        return synthetic_data

    def _validate_response(self, response: str) -> bool:
        """응답 품질 검증"""
        # 길이 체크
        if len(response.split()) < 50:
            return False

        # 반복 체크
        sentences = response.split('.')
        if len(set(sentences)) / len(sentences) < 0.8:
            return False

        return True
```

### 3.2 Knowledge Distillation

```python
"""
Knowledge Distillation: 큰 모델 → 작은 모델

Teacher (대형 모델)의 지식을 Student (SLM)에게 전달
"""

class DistillationTrainer:
    """KD 기반 SLM 학습"""

    def __init__(
        self,
        teacher_model,  # 예: Llama 70B
        student_model,  # 예: 3B 모델
        temperature: float = 2.0,
        alpha: float = 0.5  # soft/hard loss 비율
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha

        # Teacher는 학습 안 함
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Distillation Loss = α × Soft Loss + (1-α) × Hard Loss

        Soft Loss: KL(student_soft || teacher_soft)
        Hard Loss: CrossEntropy(student, labels)
        """
        T = self.temperature

        # Soft targets (temperature scaling)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        student_soft = F.log_softmax(student_logits / T, dim=-1)

        # KL Divergence (soft loss)
        soft_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (T ** 2)  # Temperature scaling 보정

        # Cross Entropy (hard loss)
        hard_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        # Combined loss
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return loss

    def train_step(self, batch):
        """학습 스텝"""
        input_ids = batch['input_ids']
        labels = batch['labels']

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(input_ids)
            teacher_logits = teacher_outputs.logits

        # Student forward
        student_outputs = self.student(input_ids)
        student_logits = student_outputs.logits

        # Distillation loss
        loss = self.distillation_loss(
            student_logits, teacher_logits, labels
        )

        return loss


# Response-level Distillation (더 효과적)
class ResponseDistillation:
    """응답 수준 KD"""

    def __init__(self, teacher_model, student_model):
        self.teacher = teacher_model
        self.student = student_model

    def generate_training_data(self, prompts: list):
        """Teacher 응답으로 학습 데이터 생성"""
        training_data = []

        for prompt in prompts:
            # Teacher 응답 생성
            teacher_response = self.teacher.generate(
                prompt,
                max_new_tokens=512,
                temperature=0.7
            )

            training_data.append({
                'prompt': prompt,
                'response': teacher_response
            })

        return training_data

    def train_on_responses(self, training_data):
        """Teacher 응답으로 Student 학습"""
        # Standard SFT (Supervised Fine-Tuning)
        for item in training_data:
            full_text = f"{item['prompt']}\n{item['response']}"
            # ... SFT 학습
```

### 3.3 효율적 학습 기법

```python
"""
SLM 학습 효율화 기법
"""

# 1. Gradient Accumulation (작은 배치로 큰 effective batch)
def train_with_grad_accumulation(
    model,
    dataloader,
    accumulation_steps: int = 8
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for i, batch in enumerate(dataloader):
        outputs = model(**batch)
        loss = outputs.loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()


# 2. LoRA로 효율적 fine-tuning
from peft import LoraConfig, get_peft_model

def setup_lora_training(model):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none"
    )

    model = get_peft_model(model, lora_config)

    # 학습 가능 파라미터 확인
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


# 3. QLoRA (양자화 + LoRA)
from transformers import BitsAndBytesConfig

def setup_qlora_training(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # LoRA 추가
    return setup_lora_training(model)
```

---

## 4. 배포 최적화

### 4.1 양자화

```python
"""
SLM 양자화: 메모리 & 속도 최적화
"""

# 1. GPTQ (Post-Training Quantization)
from transformers import GPTQConfig

def quantize_with_gptq(model_name):
    gptq_config = GPTQConfig(
        bits=4,
        dataset="c4",
        tokenizer=AutoTokenizer.from_pretrained(model_name)
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=gptq_config,
        device_map="auto"
    )

    return model


# 2. AWQ (Activation-aware Weight Quantization)
from awq import AutoAWQForCausalLM

def quantize_with_awq(model_path, output_path):
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 양자화
    model.quantize(
        tokenizer,
        quant_config={
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM"
        }
    )

    # 저장
    model.save_quantized(output_path)


# 3. llama.cpp (GGUF 포맷)
"""
llama.cpp 양자화 레벨:
- Q2_K: 2비트 (매우 작음, 품질 저하)
- Q4_K_M: 4비트 (권장, 품질/크기 균형)
- Q5_K_M: 5비트 (높은 품질)
- Q8_0: 8비트 (거의 원본 품질)

명령어:
./quantize model.gguf model-q4_k_m.gguf Q4_K_M
"""


# 메모리 사용량 비교
def compare_memory_usage():
    """파라미터 수에 따른 메모리"""
    configs = [
        ("3B FP16", 3e9 * 2),       # 6GB
        ("3B Q8", 3e9 * 1),         # 3GB
        ("3B Q4", 3e9 * 0.5),       # 1.5GB
        ("7B FP16", 7e9 * 2),       # 14GB
        ("7B Q4", 7e9 * 0.5),       # 3.5GB
    ]

    print("Model\t\tMemory (GB)")
    print("-" * 30)
    for name, memory in configs:
        print(f"{name}\t\t{memory / 1e9:.1f}")
```

### 4.2 On-Device 배포

```python
"""
모바일/엣지 디바이스 배포
"""

# 1. ONNX 변환
def convert_to_onnx(model, tokenizer, output_path):
    from optimum.onnxruntime import ORTModelForCausalLM

    # ONNX 변환 및 최적화
    ort_model = ORTModelForCausalLM.from_pretrained(
        model,
        export=True,
        provider="CPUExecutionProvider"
    )

    ort_model.save_pretrained(output_path)


# 2. TensorRT-LLM (NVIDIA GPU)
"""
TensorRT-LLM 사용:
1. 모델 변환: python convert_checkpoint.py
2. 엔진 빌드: trtllm-build
3. 추론: python run.py
"""


# 3. llama.cpp (CPU 추론)
"""
llama.cpp 사용:
1. GGUF 변환
2. llama-cli 실행

./llama-cli -m model.gguf \
    -n 256 \
    -p "Hello, how are you?" \
    -t 4  # threads
"""


# 4. MLC-LLM (다양한 플랫폼)
"""
MLC-LLM: iOS, Android, WebGPU, CUDA

mlc_chat 앱으로 모바일 배포 가능
"""
```

---

## 5. 벤치마크 & 평가

### 5.1 SLM 벤치마크 결과

```
┌──────────────────────────────────────────────────────────────────┐
│            SLM 벤치마크 비교 (2024.10 기준)                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Model          Params  MMLU    GSM8K   HumanEval  TriviaQA     │
│  ─────────────────────────────────────────────────────────────  │
│  Phi-3-mini     3.8B    69.9%   82.5%   57.9%      63.5%        │
│  Gemma-2-9B     9B      71.3%   68.6%   54.3%      73.5%        │
│  Qwen2.5-7B     7B      74.2%   82.6%   75.6%      71.4%        │
│  Llama-3.2-3B   3B      63.4%   44.4%   36.0%      63.4%        │
│  SmolLM-1.7B    1.7B    42.3%   18.2%   28.7%      42.1%        │
│                                                                  │
│  참고: GPT-4    -       86.4%   92.0%   67.0%      87.6%        │
│                                                                  │
│  ※ Phi-3은 작은 크기 대비 뛰어난 추론 능력                       │
│  ※ Qwen2.5는 코드(HumanEval)에서 강점                            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 태스크별 SLM 선택 가이드

```python
"""
태스크별 SLM 추천
"""

TASK_MODEL_RECOMMENDATIONS = {
    # 일반 대화
    "general_chat": {
        "best": "Qwen2.5-7B-Instruct",
        "budget": "Qwen2.5-1.5B-Instruct",
        "mobile": "Qwen2.5-0.5B-Instruct"
    },

    # 코드 생성
    "code_generation": {
        "best": "Qwen2.5-Coder-7B",
        "budget": "CodeGemma-2B",
        "mobile": "Phi-3-mini"
    },

    # 수학/추론
    "math_reasoning": {
        "best": "Qwen2.5-Math-7B",
        "budget": "Phi-3-mini",
        "mobile": "Phi-3-mini"
    },

    # 한국어
    "korean": {
        "best": "Qwen2.5-7B-Instruct",  # 다국어 강점
        "budget": "EXAONE-3.0-7.8B-Instruct",
        "mobile": "Qwen2.5-1.5B-Instruct"
    },

    # RAG/검색
    "rag": {
        "best": "Gemma-2-9B",
        "budget": "Llama-3.2-3B",
        "mobile": "Phi-3-mini"
    },

    # 요약
    "summarization": {
        "best": "Qwen2.5-7B-Instruct",
        "budget": "Gemma-2-2B",
        "mobile": "SmolLM-1.7B"
    }
}


def select_model(task: str, constraint: str = "best"):
    """태스크와 제약에 맞는 모델 선택"""
    if task in TASK_MODEL_RECOMMENDATIONS:
        return TASK_MODEL_RECOMMENDATIONS[task].get(constraint)
    return "Qwen2.5-7B-Instruct"  # 기본값
```

---

## 6. 실습: SLM Fine-tuning

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

def finetune_slm():
    """SLM QLoRA Fine-tuning 예제"""

    # 1. 모델 로드 (4비트 양자화)
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. LoRA 설정
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. 데이터셋
    dataset = load_dataset("timdettmers/openassistant-guanaco")

    def preprocess(examples):
        texts = []
        for text in examples['text']:
            # Qwen chat format
            texts.append(text + tokenizer.eos_token)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=1024,
            padding="max_length"
        )
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized

    tokenized_dataset = dataset['train'].map(
        preprocess,
        batched=True,
        remove_columns=dataset['train'].column_names
    )

    # 4. 학습
    training_args = TrainingArguments(
        output_dir="./qwen-finetuned",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=500,
        bf16=True,
        optim="paged_adamw_8bit"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # 5. 저장
    model.save_pretrained("./qwen-lora-adapter")

    print("Fine-tuning complete!")


if __name__ == "__main__":
    finetune_slm()
```

---

## 참고 자료

### 논문
- Gunasekar et al. (2023). "Textbooks Are All You Need" (Phi)
- Gemma Team (2024). "Gemma 2: Improving Open Language Models"
- Yang et al. (2024). "Qwen2 Technical Report"

### 모델
- [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [Gemma 2](https://huggingface.co/google/gemma-2-9b)
- [Qwen 2.5](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

### 관련 레슨
- [../LLM_and_NLP/11_Model_Quantization.md](../LLM_and_NLP/11_Model_Quantization.md)
- [19_PEFT_Unified.md](19_PEFT_Unified.md)

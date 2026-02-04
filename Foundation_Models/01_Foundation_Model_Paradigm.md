# Foundation Model 패러다임

## 학습 목표
- Foundation Model의 정의와 특징 이해
- 전통적 ML에서 Foundation Model로의 패러다임 전환 파악
- In-context Learning과 Emergent Capabilities 개념 습득
- 주요 Foundation Model 계보 파악

---

## 1. Foundation Model이란?

### 1.1 정의

**Foundation Model**(기반 모델)은 2021년 Stanford HAI에서 제안한 용어로, 다음 특징을 가진 모델을 의미합니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Foundation Model 정의                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 대규모 데이터로 사전 학습 (Pre-trained on broad data)         │
│     - 수십억~수조 토큰의 텍스트                                   │
│     - 수억~수십억 장의 이미지                                     │
│                                                                 │
│  2. 다양한 하위 작업에 적응 가능 (Adaptable to many tasks)        │
│     - 하나의 모델로 분류, 생성, QA, 번역 등 수행                   │
│     - Fine-tuning 또는 Prompting으로 적응                        │
│                                                                 │
│  3. 범용적 표현 학습 (General-purpose representations)           │
│     - Task-agnostic한 지식 인코딩                                │
│     - Transfer learning의 극대화                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 전통적 ML vs Foundation Model

```
┌─────────────────────────────────────────────────────────────────┐
│            전통적 Machine Learning 파이프라인                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Task A ───► Data A ───► Model A ───► Deploy A                  │
│  Task B ───► Data B ───► Model B ───► Deploy B                  │
│  Task C ───► Data C ───► Model C ───► Deploy C                  │
│                                                                 │
│  • 각 태스크마다 별도 데이터 수집                                  │
│  • 각 태스크마다 별도 모델 학습                                    │
│  • 태스크 간 지식 공유 제한적                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              Foundation Model 파이프라인                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    ┌─────────────────┐                          │
│  Massive Data ───► │ Foundation Model │                         │
│  (Web-scale)       └────────┬────────┘                          │
│                             │                                   │
│              ┌──────────────┼──────────────┐                    │
│              ▼              ▼              ▼                    │
│         Adapt A        Adapt B        Adapt C                   │
│         (Fine-tune)    (Prompt)       (LoRA)                    │
│              │              │              │                    │
│              ▼              ▼              ▼                    │
│         Task A         Task B         Task C                    │
│                                                                 │
│  • 하나의 대규모 사전 학습                                         │
│  • 경량 적응으로 다양한 태스크 수행                                 │
│  • 태스크 간 지식 전이 극대화                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Foundation Model의 종류

| 분류 | 대표 모델 | 입력/출력 |
|------|----------|----------|
| **Language Models** | GPT-4, LLaMA, Claude | 텍스트 → 텍스트 |
| **Vision Models** | ViT, DINOv2, SAM | 이미지 → 특징/세그멘테이션 |
| **Multimodal** | CLIP, LLaVA, GPT-4V | 텍스트+이미지 → 텍스트 |
| **Generative** | Stable Diffusion, DALL-E | 텍스트 → 이미지 |
| **Audio** | Whisper, AudioLM | 오디오 ↔ 텍스트 |
| **Code** | Codex, CodeLlama | 텍스트 → 코드 |

---

## 2. 패러다임 전환의 역사

### 2.1 타임라인

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Foundation Model 역사                             │
├──────┬──────────────────────────────────────────────────────────────┤
│ 2017 │ Transformer (Vaswani et al.) - Self-attention 도입            │
├──────┼──────────────────────────────────────────────────────────────┤
│ 2018 │ BERT (Google) - Masked LM으로 양방향 문맥 학습                 │
│      │ GPT-1 (OpenAI) - 첫 대규모 autoregressive LM                  │
├──────┼──────────────────────────────────────────────────────────────┤
│ 2019 │ GPT-2 (1.5B params) - "Too dangerous to release"              │
│      │ T5 - Text-to-Text Transfer Transformer                       │
├──────┼──────────────────────────────────────────────────────────────┤
│ 2020 │ GPT-3 (175B) - In-context Learning 발견                       │
│      │ Scaling Laws 논문 (Kaplan et al.)                             │
│      │ ViT - Vision에 Transformer 적용                               │
├──────┼──────────────────────────────────────────────────────────────┤
│ 2021 │ CLIP - Vision-Language 연결                                   │
│      │ DALL-E - 텍스트→이미지 생성                                    │
│      │ "Foundation Models" 용어 탄생 (Stanford HAI)                  │
├──────┼──────────────────────────────────────────────────────────────┤
│ 2022 │ ChatGPT - LLM의 대중화                                        │
│      │ Chinchilla - Compute-optimal Scaling                         │
│      │ Stable Diffusion - 오픈소스 이미지 생성                        │
├──────┼──────────────────────────────────────────────────────────────┤
│ 2023 │ GPT-4 - Multimodal Foundation Model                          │
│      │ LLaMA - 오픈소스 LLM 혁명                                      │
│      │ SAM - Promptable Vision Foundation Model                     │
├──────┼──────────────────────────────────────────────────────────────┤
│ 2024 │ GPT-4o, Claude 3, Gemini 1.5 - 성능 경쟁                      │
│      │ LLaMA 3, Mistral - 오픈소스 발전                               │
│      │ Sora - Video Foundation Model                                │
└──────┴──────────────────────────────────────────────────────────────┘
```

### 2.2 주요 전환점

#### (1) GPT-3의 In-context Learning (2020)

GPT-3는 Few-shot learning의 가능성을 보여주며 패러다임 전환의 계기가 되었습니다:

```python
# Traditional Approach: 각 태스크마다 Fine-tuning 필요
model = load_pretrained("bert-base")
model = fine_tune(model, sentiment_dataset, epochs=3)
result = model.predict("This movie was great!")

# GPT-3 In-context Learning: 프롬프트만으로 학습
prompt = """
Classify the sentiment:
Text: "I love this product!" → Positive
Text: "Terrible experience." → Negative
Text: "This movie was great!" →
"""
result = gpt3.generate(prompt)  # "Positive"
```

#### (2) CLIP의 Vision-Language 연결 (2021)

CLIP은 이미지와 텍스트를 같은 공간에 매핑하여 zero-shot 분류를 가능하게 했습니다:

```python
# Zero-shot Image Classification with CLIP
import clip

model, preprocess = clip.load("ViT-B/32")

# 이미지와 텍스트를 같은 공간에 임베딩
image_features = model.encode_image(preprocess(image))
text_features = model.encode_text(clip.tokenize(["a dog", "a cat", "a bird"]))

# 유사도로 분류 (학습 없이!)
similarity = (image_features @ text_features.T).softmax(dim=-1)
# [0.95, 0.03, 0.02] → "a dog"
```

#### (3) ChatGPT의 RLHF (2022)

ChatGPT는 RLHF(Reinforcement Learning from Human Feedback)로 사람과 정렬된 응답을 생성:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ChatGPT 학습 과정                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: Pre-training (GPT-3.5 base)                            │
│          웹 텍스트로 다음 토큰 예측 학습                           │
│                         │                                       │
│                         ▼                                       │
│  Step 2: Supervised Fine-tuning (SFT)                           │
│          사람이 작성한 좋은 응답으로 학습                           │
│                         │                                       │
│                         ▼                                       │
│  Step 3: Reward Model Training                                  │
│          응답 쌍의 선호도를 예측하는 모델 학습                      │
│                         │                                       │
│                         ▼                                       │
│  Step 4: RLHF with PPO                                          │
│          Reward Model을 보상으로 정책 최적화                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. In-context Learning (ICL)

### 3.1 개념

In-context Learning은 모델 가중치를 업데이트하지 않고 프롬프트 내 예시만으로 태스크를 수행하는 능력입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    In-context Learning 종류                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Zero-shot:  "Translate to French: Hello"                       │
│              → "Bonjour"                                        │
│                                                                 │
│  One-shot:   "English: Hello → French: Bonjour                  │
│               English: Goodbye →"                               │
│              → "Au revoir"                                      │
│                                                                 │
│  Few-shot:   "English: Hello → French: Bonjour                  │
│               English: Goodbye → French: Au revoir              │
│               English: Thank you → French: Merci                │
│               English: Good morning →"                          │
│              → "Bonjour" (또는 "Bon matin")                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 ICL이 작동하는 이유 (가설들)

```python
"""
가설 1: Bayesian Inference
- 프롬프트 예시로부터 태스크 분포를 추론
- P(output | input, examples) ∝ P(examples | task) × P(task)

가설 2: Implicit Gradient Descent
- Transformer의 attention이 암묵적으로 gradient step을 수행
- 메타 학습과 유사한 메커니즘

가설 3: Task Vector Retrieval
- 사전 학습 중 학습한 태스크 벡터를 검색
- 프롬프트가 적절한 태스크 벡터를 활성화
"""
```

### 3.3 Few-shot 프롬프트 예시

```python
# 감정 분석 Few-shot
sentiment_prompt = """
Analyze the sentiment of the following reviews:

Review: "The food was delicious and the service was excellent!"
Sentiment: Positive

Review: "I waited for an hour and the waiter was rude."
Sentiment: Negative

Review: "It was okay, nothing special but not bad either."
Sentiment: Neutral

Review: "Best experience ever! Will definitely come back!"
Sentiment:"""

# API 호출
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": sentiment_prompt}]
)
print(response.choices[0].message.content)  # "Positive"
```

---

## 4. Emergent Capabilities (창발적 능력)

### 4.1 정의

**Emergent Capabilities**는 작은 모델에서는 없다가 특정 규모 이상에서 갑자기 나타나는 능력입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    창발적 능력의 특징                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Performance                                                    │
│       │                                                         │
│   100%├─────────────────────────────────────────○───── 대형 모델  │
│       │                                      ╱                  │
│       │                                    ╱                    │
│       │                              ╱                          │
│    50%├─ · · · · · · · · · · · ·╱· · · · · · · · · · · · · · ·  │
│       │                      ╱    ↑ Phase Transition            │
│       │                    ╱      (갑작스러운 성능 향상)           │
│       │──────────────────○─────────────────────────── 소형 모델  │
│     0%├───────┬───────┬───────┬───────┬───────┬──────▶         │
│       │      10B    50B    100B   200B   500B     Parameters   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 대표적인 창발적 능력

| 능력 | 설명 | 출현 규모 (대략) |
|------|------|-----------------|
| **Arithmetic** | 다자리 덧셈/곱셈 | ~10B params |
| **Chain-of-Thought** | 단계적 추론 | ~60B params |
| **Word Unscrambling** | 섞인 단어 복원 | ~60B params |
| **Multi-step Math** | 복잡한 수학 문제 | ~100B params |
| **Code Generation** | 복잡한 코드 작성 | ~100B params |

### 4.3 Chain-of-Thought (CoT) Prompting

```python
# Without CoT - 종종 실패
prompt_direct = """
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?
A:"""
# GPT-3 (small): "8" (틀림)

# With CoT - 단계적 추론으로 정확도 향상
prompt_cot = """
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?
A: Let's think step by step.
   Roger started with 5 tennis balls.
   He bought 2 cans, each with 3 balls, so 2 × 3 = 6 balls.
   Total: 5 + 6 = 11 tennis balls.
   The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and
   bought 6 more, how many apples do they have?
A: Let's think step by step."""
# GPT-3: "They started with 23, used 20, so 23-20=3.
#         Then bought 6 more: 3+6=9. The answer is 9." (정답)
```

---

## 5. Foundation Model의 핵심 구성 요소

### 5.1 아키텍처 비교

```
┌─────────────────────────────────────────────────────────────────┐
│                    주요 아키텍처 패턴                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Encoder-only (BERT, DINOv2)                                    │
│  ┌─────────────────────────────────────────┐                    │
│  │ [CLS] Token1 Token2 ... TokenN [SEP]    │                    │
│  │       ↓      ↓      ↓    ↓              │                    │
│  │    ┌──┴──────┴──────┴────┴──┐           │                    │
│  │    │   Bidirectional Attn   │           │                    │
│  │    └──┬──────┬──────┬────┬──┘           │                    │
│  │       ↓      ↓      ↓    ↓              │                    │
│  │    Pooled / Token Representations       │                    │
│  └─────────────────────────────────────────┘                    │
│  • 양방향 문맥 활용                                               │
│  • 분류, 임베딩에 적합                                            │
│                                                                 │
│  Decoder-only (GPT, LLaMA)                                      │
│  ┌─────────────────────────────────────────┐                    │
│  │ Token1 → Token2 → Token3 → ...          │                    │
│  │   ↓        ↓        ↓                   │                    │
│  │ ┌─┴────────┴────────┴─┐                 │                    │
│  │ │  Causal (Masked) Attn│                │                    │
│  │ └─┬────────┬────────┬─┘                 │                    │
│  │   ↓        ↓        ↓                   │                    │
│  │ Next     Next     Next                  │                    │
│  │ Token    Token    Token                 │                    │
│  └─────────────────────────────────────────┘                    │
│  • 자기회귀적 생성                                                │
│  • 텍스트 생성에 최적화                                           │
│                                                                 │
│  Encoder-Decoder (T5, BART)                                     │
│  ┌─────────────────────────────────────────┐                    │
│  │ ┌──────────┐    ┌──────────┐            │                    │
│  │ │ Encoder  │───▶│ Decoder  │            │                    │
│  │ │(Bi-dir)  │    │(Causal)  │            │                    │
│  │ └──────────┘    └──────────┘            │                    │
│  └─────────────────────────────────────────┘                    │
│  • 입력 이해 + 출력 생성 분리                                      │
│  • 번역, 요약에 적합                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 주요 구성 요소

```python
"""
Foundation Model의 핵심 구성 요소:

1. Self-Attention
   - Query, Key, Value 연산
   - 모든 위치 간 관계 학습

2. Feed-Forward Network (FFN)
   - 지식 저장소 역할
   - 파라미터의 대부분을 차지

3. Positional Encoding
   - 순서 정보 주입
   - Sinusoidal, Learnable, RoPE 등

4. Normalization
   - LayerNorm (BERT, GPT)
   - RMSNorm (LLaMA) - 더 효율적

5. Activation Function
   - GELU (BERT, GPT)
   - SwiGLU (LLaMA) - 더 나은 성능
"""
```

---

## 6. Foundation Model 사용하기

### 6.1 HuggingFace로 시작하기

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 모델 로드 (예: LLaMA-2-7B)
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 메모리 절약
    device_map="auto"           # 자동 GPU 할당
)

# 텍스트 생성
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
    top_p=0.9
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 6.2 Vision Foundation Model 사용

```python
# DINOv2 - 범용 이미지 특징 추출
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")

# 이미지 임베딩 추출
image = Image.open("image.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    features = outputs.last_hidden_state  # (1, num_patches+1, 768)
    cls_embedding = features[:, 0]        # CLS 토큰 (전체 이미지 표현)

# 이 임베딩을 분류, 검색, 세그멘테이션 등에 활용
```

### 6.3 API 사용 (OpenAI)

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain foundation models in simple terms."}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

---

## 7. Foundation Model의 한계와 도전

### 7.1 현재 한계

| 한계 | 설명 | 해결 시도 |
|------|------|----------|
| **Hallucination** | 사실이 아닌 정보 생성 | RAG, Grounding |
| **Outdated Knowledge** | 학습 이후 정보 모름 | RAG, Fine-tuning |
| **Reasoning Limits** | 복잡한 논리 추론 어려움 | CoT, Self-consistency |
| **High Compute Cost** | 학습/추론 비용 막대 | Quantization, Distillation |
| **Safety/Alignment** | 유해 콘텐츠 생성 가능 | RLHF, Constitutional AI |

### 7.2 연구 방향

```
┌─────────────────────────────────────────────────────────────────┐
│                    미래 연구 방향                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Efficient Models                                            │
│     └─ Mixture of Experts, Sparse Attention, Quantization       │
│                                                                 │
│  2. Multimodal Integration                                      │
│     └─ Vision + Language + Audio + Code 통합                    │
│                                                                 │
│  3. Reasoning Enhancement                                       │
│     └─ Test-time Compute (o1), Tree of Thoughts                 │
│                                                                 │
│  4. Continual Learning                                          │
│     └─ 지속적 학습, Catastrophic Forgetting 해결                 │
│                                                                 │
│  5. Safety & Alignment                                          │
│     └─ Constitutional AI, Red-teaming, Interpretability         │
│                                                                 │
│  6. Agentic Systems                                             │
│     └─ Tool Use, Multi-Agent, Autonomous Planning               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 정리

### 핵심 개념
- **Foundation Model**: 대규모 데이터로 학습하여 다양한 태스크에 적용 가능한 범용 모델
- **패러다임 전환**: Task-specific → Pre-train & Adapt
- **In-context Learning**: 가중치 업데이트 없이 프롬프트로 학습
- **Emergent Capabilities**: 규모에 따라 갑자기 나타나는 능력

### 다음 단계
- [02_Scaling_Laws.md](02_Scaling_Laws.md): 모델 크기와 성능의 관계
- [03_Emergent_Abilities.md](03_Emergent_Abilities.md): 창발적 능력 심층 분석

---

## 참고 자료

### 핵심 논문
- Bommasani et al. (2021). "On the Opportunities and Risks of Foundation Models"
- Brown et al. (2020). "Language Models are Few-Shot Learners" (GPT-3)
- Radford et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision" (CLIP)
- Wei et al. (2022). "Emergent Abilities of Large Language Models"

### 추가 자료
- [Stanford HAI Foundation Models Report](https://crfm.stanford.edu/report.html)
- [HuggingFace Model Hub](https://huggingface.co/models)
- [Papers With Code - Foundation Models](https://paperswithcode.com/methods/category/foundation-models)

# 19. RLHF와 LLM 정렬 (Alignment)

## 학습 목표

- RLHF(Reinforcement Learning from Human Feedback) 이해
- Reward Model 학습
- PPO를 통한 정책 최적화
- DPO(Direct Preference Optimization)
- Constitutional AI와 안전한 AI

---

## 이론과 원리

사전학습된 LLM은 *언어 모델*입니다 — 그럴듯한 다음 토큰을 예측합니다. *도움이 되는 어시스턴트*가 아닙니다 — 정확하거나, 정중하거나, 안전한 것에 대한 선호가 없습니다. 정렬(alignment)은 사전학습된 모델을 인간이 실제로 원하는 출력을 만드는 쪽으로 구부리는 과정입니다. 지배적 레시피는 **RLHF(Reinforcement Learning from Human Feedback)** — 모델 출력에 대한 인간 선호를 수집, 그 선호를 예측하는 *보상 모델(reward model)*을 학습, 그 후 LLM을 예측된 보상을 최대화하도록 파인튜닝 — 모델을 원래 분포에 가깝게 유지하는 안정성 제약과 함께.

이 섹션은 다음을 다룹니다:

- **(A) 왜 정렬이 필요한가** — 언어 모델링과 도움이 되는 어시스턴스 사이의 격차.
- **(B) 3단계 RLHF 파이프라인** — SFT → 보상 모델 → RL 정책 최적화.
- **(C) 보상 모델과 Bradley-Terry** — 선호 모델, 쌍별 인간 평가가 어떻게 스칼라 보상이 되는가.
- **(D) RLHF 맥락의 PPO 목적함수** — KL 정규화, trust-region 논거, 평범한 policy gradient 대신 PPO인 이유.
- **(E) DPO (Direct Preference Optimization)** — 보상 모델을 완전히 제거하는 closed-form 유도.
- **(F) Constitutional AI와 RLAIF** — 일부 인간 라벨을 AI 생성 라벨로 대체.

### A. 왜 정렬이 필요한가

사전학습은 웹 텍스트에 대해 `p(next_token | context)`를 최적화합니다. 웹은 다음을 담고 있습니다:
- 도움이 되는 튜토리얼과 도움 안 되는 스팸.
- 정중한 Stack Overflow 답변과 독성 포럼 글.
- 참된 사실과 자신감 있는 잘못된 정보.

사전학습 모델은 이 코퍼스의 분포에 적합 — 나쁜 것들 포함 그 모두를 만드는 법을 압니다. RLHF 전에 "어떻게 폭탄을 만들 수 있나요?"를 물으면 모델은 거부에 대한 선호가 없습니다 — 그 일은 그 텍스트의 그럴듯한 연속을 만드는 것이고, 코퍼스에는 거부와 설명의 예시 모두가 있습니다.

정렬은 모델의 *출력 분포*를 인간이 실제로 선호하는 것 — 도움이 되고, 정직하고, 무해함 — 으로 시프트합니다. 시프트는 한 번의 작업이 아니라 능력(모델의 지식 보존)과 준수(선호 행동으로 조종) 사이의 지속적 줄다리기입니다.

### B. 3단계 RLHF 파이프라인

InstructGPT / ChatGPT 레시피(Ouyang 등, 2022):

**B.1 SFT (Supervised Fine-Tuning).** 인간이 작성한 (프롬프트, 이상적 응답) 시연을 받아 표준 cross-entropy 손실로 사전학습 LLM을 파인튜닝. 출력 — 지시를 합리적으로 따르지만 완벽하지는 않은 모델.

**B.2 보상 모델 (RM) 학습.** 인간 비교 수집 — 프롬프트와 두 후보 응답 A, B가 주어졌을 때 인간이 선호하는 것을 선택. 모델 `r_θ(prompt, response) → ℝ`을 응답을 점수화하도록 학습 — 선호된 응답이 더 높은 점수.

**B.3 RL 파인튜닝.** PPO(또는 유사)를 사용해 SFT 모델을 갱신, 그 출력이 보상 모델 점수를 최대화하도록 — SFT 모델에서 너무 멀리 표류하지 않도록 KL 페널티와 함께.

각 단계는 다른 역할을 가집니다 — SFT는 *형식*을 가르치고(지시를 따르는 법), RM은 *선호*를 포착(어떤 응답이 더 나은가), RL은 그 선호로 정책을 *최적화*.

### C. 보상 모델과 Bradley-Terry

보상 모델은 **Bradley-Terry 선호 모델**을 사용 — 응답 A가 B보다 선호될 확률은

```
P(A > B) = σ(r(prompt, A) − r(prompt, B))
```

`σ`는 시그모이드. 동치로, A를 B보다 선호할 *로그 오즈(log-odds)*가 보상 점수의 차이와 같습니다. 쌍별 비교 데이터의 표준 모델(체스 Elo 등급에 사용 등).

**손실.** `(prompt, chosen, rejected)` 트리플 데이터셋이 주어졌을 때:

```
L_RM = − E [ log σ(r(prompt, chosen) − r(prompt, rejected)) ]
```

이를 최대화하면 `r`이 rejected보다 chosen에 더 높은 점수를 출력하도록 학습 — 차이가 선호의 로그 오즈와 같도록 보정.

**아키텍처.** LLM(종종 SFT 모델 자체)을 가져와 마지막 언어 헤드를 마지막 토큰 은닉 상태에 대한 스칼라 회귀 헤드로 교체. 파인튜닝.

### D. RLHF의 PPO

평범한 policy gradient는 `θ ← θ + η · ∇θ E[r(τ)]`을 직접 갱신. LLM에 대해 이는 불안정 — 단일 갱신이 정책 분포를 극적으로 시프트하여 후속 표본 추출을 깰 수 있습니다.

**PPO (Proximal Policy Optimization)**는 두 안전장치 추가:

**D.1 클립된 surrogate 목적함수.** 중요도 표본 추출 비율 `r_t = π(a_t | s_t) / π_old(a_t | s_t)` 사용. 어드밴티지(advantage)와 곱하기 전에 `[1−ε, 1+ε]`(전형적 `ε = 0.2`)로 클립. 확률 비율을 너무 극적으로 바꿀 갱신은 그래디언트가 0. 경험적으로 이는 파국적 정책 갱신을 방지.

**D.2 KL 페널티 (RLHF 특화).** 보상에 페널티 항 `−β · KL(π_θ ‖ π_SFT)` 추가. 이는 *명시적으로* 정책을 SFT 모델에 가깝게 유지, 언어 품질을 보존하고 모델이 보상 모델을 속이는 적대적 시퀀스를 찾는 것 방지:

```
total_reward(prompt, response) = r_θ(prompt, response) − β · log(π_θ(response | prompt) / π_SFT(response | prompt))
```

KL 항 없이는 RL이 어떤 보상 모델 편향이든 악용 — (불완전한) 보상 모델에서 잘 점수받는 이상하고, 반복적이거나, 분포 밖 텍스트를 만듭니다. 그것과 함께, 정책이 SFT 분포에 정박합니다.

전체 PPO-RLHF 목적함수는 둘 다 결합. `β` 튜닝이 보상 최대화와 SFT에 가깝게 유지를 균형.

### E. DPO: Direct Preference Optimization

DPO(Rafailov 등, 2023)는 아름다운 유도로 보상 모델을 완전히 제거합니다.

**E.1 통찰.** PPO 목적함수 `r_θ − β · KL(π‖π_ref)` 하의 최적 정책은 알려진 closed form을 가집니다:

```
π*(y | x) ∝ π_ref(y | x) · exp(r_θ(x, y) / β)
```

뒤집으면 `r_θ(x, y) = β · log(π*(y | x) / π_ref(y | x)) + const`. 그래서 보상은 암묵적으로 최적 정책과 참조 사이의 *로그 비율*의 함수입니다.

**E.2 Bradley-Terry로 대입.** 선호 확률은 다음이 됩니다:

```
P(y_w > y_l | x) = σ( β · [log(π_θ(y_w|x)/π_ref(y_w|x)) − log(π_θ(y_l|x)/π_ref(y_l|x))] )
```

`π_ref`를 SFT 모델로 사용해 Bradley-Terry 최대 가능도 손실로 `π_θ`를 직접 학습. 보상 모델 없음. RL 루프 없음. 선호 쌍에 대한 지도 cross-entropy일 뿐.

**E.3 작동 원리.** 정규화된 정책 최적화의 closed-form 해가 *알려져* 있습니다. DPO는 그것을 선호 가능도에 대입, 중간 보상 모델을 제거합니다. 최적화가 안정적(그저 MLE), PPO보다 훨씬 저렴(롤아웃 없음, 루프 안의 KL 추정 없음), 표준 벤치마크에서 PPO-RLHF에 경험적으로 일치하거나 이깁니다.

DPO는 이제 많은 오픈소스 선호 파인튠(Zephyr, Mistral-Instruct 변형, Llama 3 chat 버전)의 기본값입니다.

### F. Constitutional AI와 RLAIF

인간 선호 수집은 비쌉니다. 비용을 줄이는 두 아이디어:

**F.1 Constitutional AI** (Bai 등, 2022). 인간 안전 라벨을 AI 라벨로 대체 — 원칙의 "헌법"("도움이 되되 해롭지 않게")을 작성, 그 후 LLM이 헌법에 대해 자기 출력을 비평하고 수정하게 합니다. 수정된 출력이 chosen 응답이 되고, 원본 출력은 rejected.

**F.2 RLAIF (RL from AI Feedback).** 일반화 — 인간 대신 강력한 LLM을 라벨러로 사용. 놀랍게도 많은 작업에서 AI 라벨이 인간 라벨만큼 좋습니다 — 그리고 자릿수만큼 더 저렴. 정답이 검증 가능한 능력 작업(수학, 코드)에 특히 효과적.

현대 정렬 파이프라인은 혼합 사용 — 미묘한 선호에는 인간 라벨, 규모에는 AI 라벨.

### 이론에서 아래 함수들로

- §1 (정렬 개요) — §A와 §B 파이프라인을 틀.
- §2 (SFT) — 표준 파인튜닝으로 §B.1 구현.
- §3 (RLHF 파이프라인) — §B 전체 흐름.
- §4 (보상 모델) — §C Bradley-Terry RM 코딩.
- §5 (PPO) — §D 클립된 목적함수와 KL 페널티 구현.
- §6 (DPO) — §E closed-form 선호 최적화 구현.
- §7 (Constitutional AI) — §F.1 자기 비평 루프 구현.
- §8 (고급) — §C-§F 위에 세워진 새 방법(RLAIF, KTO, IPO).

---

## 1. LLM 정렬 개요

### 왜 정렬이 필요한가?

> **LLM 정렬의 필요성**
>
>
> 사전학습 모델 (Base Model)
> │
> │  문제점:
> │  - 단순히 다음 토큰 예측
> │  - 유해한 콘텐츠 생성 가능
> │  - 지시사항 따르기 어려움
> ▼
> 정렬된 모델 (Aligned Model)
> │
> │  목표:
> │  - 도움됨 (Helpful)
> │  - 무해함 (Harmless)
> │  - 정직함 (Honest)


### 정렬 방법론 발전

```
SFT (Supervised Fine-Tuning)
    │  고품질 데이터로 지도학습
    ▼
RLHF (Reinforcement Learning from Human Feedback)
    │  보상 모델 + 강화학습
    ▼
DPO (Direct Preference Optimization)
    │  직접 선호도 최적화
    ▼
Constitutional AI
    │  원칙 기반 자기 개선
```

---

## 2. SFT (Supervised Fine-Tuning)

### 기본 개념

```python
# SFT 데이터 형식
sft_data = [
    {
        "instruction": "Write a poem about spring.",
        "input": "",
        "output": "Flowers bloom in gentle rain,\nBirds return to sing again..."
    },
    {
        "instruction": "Translate to French.",
        "input": "Hello, how are you?",
        "output": "Bonjour, comment allez-vous?"
    }
]
```

### SFT 구현

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

# 모델과 토크나이저
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# 데이터셋
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

# 포맷팅 함수
def format_instruction(example):
    if example["context"]:
        return f"""### Instruction:
{example['instruction']}

### Context:
{example['context']}

### Response:
{example['response']}"""
    else:
        return f"""### Instruction:
{example['instruction']}

### Response:
{example['response']}"""

# 학습 설정
training_args = TrainingArguments(
    output_dir="./sft_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
)

# SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=format_instruction,
    max_seq_length=1024,
    args=training_args,
)

trainer.train()
```

---

## 3. RLHF 파이프라인

### 전체 프로세스

> **RLHF 파이프라인**
>
> - **1단계: SFT** -- Base Model --지도학습--> SFT Model
> - **2단계: Reward Model 학습** -- SFT Model --선호도 데이터--> Reward Model
> - **3단계: PPO 강화학습** -- SFT Model (Policy) + Reward Model (Critic) --> RLHF Model (Aligned)

### 선호도 데이터 수집

```python
# 선호도 데이터 형식
preference_data = [
    {
        "prompt": "Write a haiku about mountains.",
        "chosen": "Peaks touch morning sky\nSilent guardians of earth\nMist embraces stone",
        "rejected": "Mountains are big\nThey are tall and rocky\nI like mountains"
    },
    {
        "prompt": "Explain quantum computing.",
        "chosen": "Quantum computing harnesses quantum mechanics principles...",
        "rejected": "Quantum computing is computers that use quantum stuff..."
    }
]

# HuggingFace 형식
from datasets import Dataset

dataset = Dataset.from_list(preference_data)
dataset = dataset.map(lambda x: {
    "prompt": x["prompt"],
    "chosen": x["chosen"],
    "rejected": x["rejected"]
})
```

---

## 4. Reward Model 학습

### Reward Model 개념

```
입력: (prompt, response)
출력: scalar reward (점수)

학습 목표:
    reward(prompt, chosen) > reward(prompt, rejected)
```

### 구현

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from trl import RewardTrainer

# Reward Model (분류 헤드 추가)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    num_labels=1  # 스칼라 출력
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# 데이터 전처리
def preprocess_reward_data(examples):
    """선호도 데이터를 Reward 학습용으로 변환"""
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }

    for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        # Chosen
        chosen_text = f"### Prompt: {prompt}\n### Response: {chosen}"
        chosen_tokenized = tokenizer(chosen_text, truncation=True, max_length=512)
        new_examples["input_ids_chosen"].append(chosen_tokenized["input_ids"])
        new_examples["attention_mask_chosen"].append(chosen_tokenized["attention_mask"])

        # Rejected
        rejected_text = f"### Prompt: {prompt}\n### Response: {rejected}"
        rejected_tokenized = tokenizer(rejected_text, truncation=True, max_length=512)
        new_examples["input_ids_rejected"].append(rejected_tokenized["input_ids"])
        new_examples["attention_mask_rejected"].append(rejected_tokenized["attention_mask"])

    return new_examples

# 학습
training_args = TrainingArguments(
    output_dir="./reward_model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    fp16=True,
)

trainer = RewardTrainer(
    model=reward_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

### Reward Model 사용

```python
def get_reward(model, tokenizer, prompt, response):
    """응답에 대한 보상 점수 계산"""
    text = f"### Prompt: {prompt}\n### Response: {response}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        reward = outputs.logits.squeeze().item()

    return reward

# 사용 예시
prompt = "Explain photosynthesis."
response_good = "Photosynthesis is the process by which plants convert sunlight..."
response_bad = "Plants eat light."

print(f"Good response reward: {get_reward(reward_model, tokenizer, prompt, response_good):.4f}")
print(f"Bad response reward: {get_reward(reward_model, tokenizer, prompt, response_bad):.4f}")
```

---

## 5. PPO (Proximal Policy Optimization)

### PPO 개념

```
PPO 목표함수:
    L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]

여기서:
    r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (확률 비율)
    A_t = 어드밴티지 (Reward - Baseline)
    ε = 클리핑 범위 (보통 0.2)

KL 제약:
    D_KL[π_θ || π_ref] < δ  (기준 모델과 너무 멀어지지 않도록)
```

### PPO 구현 (TRL)

```python
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
import torch

# PPO 설정
ppo_config = PPOConfig(
    model_name="./sft_model",
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=1,
    ppo_epochs=4,
    max_grad_norm=0.5,
    kl_penalty="kl",           # KL 페널티 방식
    target_kl=0.1,             # 목표 KL divergence
    init_kl_coef=0.2,          # 초기 KL 계수
)

# 모델 (Value head 포함)
model = AutoModelForCausalLMWithValueHead.from_pretrained("./sft_model")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("./sft_model")  # 기준 모델 (고정)
tokenizer = AutoTokenizer.from_pretrained("./sft_model")
tokenizer.pad_token = tokenizer.eos_token

# Reward Model 로드
reward_model = AutoModelForSequenceClassification.from_pretrained("./reward_model")
reward_tokenizer = AutoTokenizer.from_pretrained("./reward_model")

# PPO Trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# 학습 루프
def get_reward_batch(prompts, responses):
    """배치 보상 계산"""
    rewards = []
    for prompt, response in zip(prompts, responses):
        text = f"### Prompt: {prompt}\n### Response: {response}"
        inputs = reward_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(reward_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            reward = reward_model(**inputs).logits.squeeze()
        rewards.append(reward)

    return rewards

# 학습
from datasets import load_dataset
dataset = load_dataset("Anthropic/hh-rlhf", split="train")

for epoch in range(ppo_config.ppo_epochs):
    for batch in dataset.iter(batch_size=ppo_config.batch_size):
        # 프롬프트 토큰화
        query_tensors = [tokenizer.encode(p, return_tensors="pt").squeeze() for p in batch["prompt"]]

        # 응답 생성
        response_tensors = []
        for query in query_tensors:
            response = ppo_trainer.generate(query, max_new_tokens=128)
            response_tensors.append(response.squeeze())

        # 텍스트 디코딩
        prompts = batch["prompt"]
        responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

        # 보상 계산
        rewards = get_reward_batch(prompts, responses)

        # PPO 스텝
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        # 로깅
        ppo_trainer.log_stats(stats, batch, rewards)

# 저장
model.save_pretrained("./rlhf_model")
```

---

## 6. DPO (Direct Preference Optimization)

### DPO 개념

```
DPO = RLHF without Reward Model

핵심 아이디어:
    - Reward Model 없이 직접 선호도 데이터로 학습
    - Bradley-Terry 모델 기반
    - 더 안정적이고 간단한 학습

손실 함수:
    L_DPO = -E[log σ(β(log π_θ(y_w|x) - log π_ref(y_w|x)
                      - log π_θ(y_l|x) + log π_ref(y_l|x)))]

여기서:
    y_w = 선호 응답 (winner)
    y_l = 비선호 응답 (loser)
    β = 온도 파라미터
```

### DPO vs RLHF

| 항목 | RLHF | DPO |
|------|------|-----|
| Reward Model | 필요 | 불필요 |
| 학습 안정성 | 불안정 | 안정적 |
| 하이퍼파라미터 | 많음 | 적음 |
| 메모리 | 높음 | 낮음 |
| 성능 | 우수 | 동등 이상 |

### DPO 구현

```python
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 모델
model = AutoModelForCausalLM.from_pretrained("./sft_model")
ref_model = AutoModelForCausalLM.from_pretrained("./sft_model")
tokenizer = AutoTokenizer.from_pretrained("./sft_model")
tokenizer.pad_token = tokenizer.eos_token

# 데이터셋 (prompt, chosen, rejected 형식)
dataset = load_dataset("Anthropic/hh-rlhf", split="train")

# DPO 설정
dpo_config = DPOConfig(
    beta=0.1,                          # 온도 파라미터
    loss_type="sigmoid",               # sigmoid 또는 hinge
    max_length=512,
    max_prompt_length=256,
    learning_rate=5e-7,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    output_dir="./dpo_model",
    fp16=True,
)

# DPO Trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# 학습
dpo_trainer.train()

# 저장
model.save_pretrained("./dpo_model_final")
```

### DPO 변형들

```python
# IPO (Identity Preference Optimization)
dpo_config = DPOConfig(
    loss_type="ipo",
    label_smoothing=0.0,
)

# KTO (Kahneman-Tversky Optimization)
# 선호/비선호 쌍 대신 개별 평가 사용
from trl import KTOConfig, KTOTrainer

kto_config = KTOConfig(
    beta=0.1,
    desirable_weight=1.0,
    undesirable_weight=1.0,
)

# ORPO (Odds Ratio Preference Optimization)
# Reference model 불필요
from trl import ORPOConfig, ORPOTrainer

orpo_config = ORPOConfig(
    beta=0.1,
    # ref_model 없이 학습
)
```

---

## 7. Constitutional AI

### 개념

```
Constitutional AI (CAI) = 원칙 기반 자기 개선

단계:
    1. 모델이 응답 생성
    2. 헌법(원칙)에 따라 자기 비평
    3. 비평을 바탕으로 응답 수정
    4. 수정된 응답으로 학습

원칙 예시:
    - "도움이 되어야 함"
    - "해로운 내용을 포함하지 않아야 함"
    - "정직해야 함"
    - "개인정보를 노출하지 않아야 함"
```

### CAI 구현

```python
from openai import OpenAI

client = OpenAI()

# 원칙 (Constitution)
constitution = """
1. 응답은 도움이 되어야 합니다.
2. 응답은 해로운 내용을 포함하지 않아야 합니다.
3. 응답은 정직하고 사실에 기반해야 합니다.
4. 개인정보나 민감한 정보를 공개하지 않아야 합니다.
5. 차별적이거나 편견 있는 내용을 포함하지 않아야 합니다.
"""

def generate_initial_response(prompt):
    """초기 응답 생성"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

def critique_response(prompt, response, constitution):
    """응답 비평"""
    critique_prompt = f"""다음 응답이 주어진 원칙을 잘 따르는지 평가하세요.

원칙:
{constitution}

사용자 질문: {prompt}

응답: {response}

각 원칙에 대해 응답이 어떻게 위반하거나 준수하는지 분석하세요.
"""
    critique = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": critique_prompt}],
        temperature=0.3
    )
    return critique.choices[0].message.content

def revise_response(prompt, response, critique, constitution):
    """응답 수정"""
    revision_prompt = f"""다음 비평을 바탕으로 응답을 개선하세요.

원칙:
{constitution}

사용자 질문: {prompt}

원래 응답: {response}

비평: {critique}

원칙을 더 잘 준수하도록 응답을 수정하세요. 수정된 응답만 출력하세요.
"""
    revised = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": revision_prompt}],
        temperature=0.3
    )
    return revised.choices[0].message.content

def constitutional_ai_pipeline(prompt, constitution, iterations=2):
    """CAI 파이프라인"""
    response = generate_initial_response(prompt)
    print(f"초기 응답:\n{response}\n")

    for i in range(iterations):
        critique = critique_response(prompt, response, constitution)
        print(f"비평 {i+1}:\n{critique}\n")

        response = revise_response(prompt, response, critique, constitution)
        print(f"수정된 응답 {i+1}:\n{response}\n")

    return response

# 사용
prompt = "How can I pick a lock?"
final_response = constitutional_ai_pipeline(prompt, constitution)
```

---

## 8. 고급 정렬 기법

### RLAIF (RL from AI Feedback)

```python
def get_ai_preference(prompt, response_a, response_b):
    """AI가 선호도 판단"""
    judge_prompt = f"""다음 두 응답 중 더 좋은 것을 선택하세요.

질문: {prompt}

응답 A: {response_a}

응답 B: {response_b}

평가 기준:
- 정확성
- 유용성
- 명확성
- 안전성

더 좋은 응답 (A 또는 B)와 이유를 말하세요.
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0
    )
    return response.choices[0].message.content
```

### Self-Play Fine-Tuning (SPIN)

```python
# SPIN: 모델이 자신의 응답과 경쟁

def spin_iteration(model, dataset):
    """SPIN 반복"""
    # 1. 현재 모델로 응답 생성
    synthetic_responses = generate_responses(model, dataset["prompts"])

    # 2. 실제 응답 vs 생성된 응답으로 DPO
    spin_dataset = {
        "prompt": dataset["prompts"],
        "chosen": dataset["responses"],      # 실제 응답
        "rejected": synthetic_responses      # 모델 생성 응답
    }

    # 3. DPO 학습
    model = dpo_train(model, spin_dataset)

    return model
```

---

## 정리

### 정렬 방법 비교

| 방법 | 복잡도 | 성능 | 사용 시점 |
|------|--------|------|----------|
| SFT | 낮음 | 기본 | 항상 첫 단계 |
| RLHF (PPO) | 높음 | 우수 | 복잡한 정렬 |
| DPO | 중간 | 우수 | 간단한 정렬 |
| ORPO | 낮음 | 좋음 | 메모리 제한 |
| CAI | 중간 | 안전성 | 안전 중요 |

### 핵심 코드

```python
# SFT
from trl import SFTTrainer
trainer = SFTTrainer(model, train_dataset, formatting_func=format_fn)

# DPO
from trl import DPOTrainer, DPOConfig
config = DPOConfig(beta=0.1)
trainer = DPOTrainer(model, ref_model, args=config, train_dataset=dataset)

# PPO
from trl import PPOTrainer, PPOConfig
config = PPOConfig(target_kl=0.1)
trainer = PPOTrainer(config, model, ref_model, tokenizer)
stats = trainer.step(queries, responses, rewards)
```

### 정렬 파이프라인

```
1. SFT: 고품질 데이터로 기본 능력 학습
2. 선호도 데이터 수집 (인간 또는 AI)
3. DPO/RLHF로 선호도 학습
4. 안전성 평가 및 추가 정렬
5. 배포 및 피드백 수집
```

---

## 연습 문제

### 연습 문제 1: 선호도 데이터 품질

아래 세 가지 선호도 데이터 쌍을 검토하세요. 각 쌍에 대해 `chosen` 응답이 진정으로 더 나은지 판단하고, RLHF/DPO 훈련에 좋은 선호도 데이터를 만드는 요소가 무엇인지 설명하세요.

```python
preference_pairs = [
    {
        "prompt": "프랑스의 수도는 어디인가요?",
        "chosen": "프랑스의 수도는 파리로, 10세기부터 수도였습니다. 센 강변에 위치하며 인구 200만 명 이상의 도시입니다.",
        "rejected": "파리가 수도입니다."
    },
    {
        "prompt": "학교 인터넷 필터를 우회하는 방법은?",
        "chosen": "인터넷 필터에 대한 불편함을 이해합니다. 합법적인 방법으로는 IT 부서에 특정 교육 사이트 접근을 요청하거나, 교사에게 특정 자료에 대한 접근 권한을 요청할 수 있습니다.",
        "rejected": "VPN이나 프록시 서비스를 사용하세요. 학교에서 잘 작동하는 5가지 인기 서비스: [서비스 목록]..."
    },
    {
        "prompt": "가을에 대한 하이쿠를 써줘.",
        "chosen": "단풍잎 붉게 물들어\n고요한 바람 사이로 속삭이며\n겨울의 첫 약속",
        "rejected": "가을이 왔어요\n나뭇잎이 떨어지네요\n날씨가 쌀쌀해요"
    }
]
```

<details>
<summary>정답 보기</summary>

**쌍 1 (프랑스 수도):**
- **chosen이 더 나음**: 예
- **이유**: chosen 응답은 역사적, 지리적 맥락을 제공하여 교육적 가치가 있습니다. rejected는 기술적으로 올바르지만 유용성이 낮습니다.
- **품질 고려사항**: 사실적 질문에 대해 선호 차이가 명확한 좋은 훈련 데이터입니다.

**쌍 2 (인터넷 필터 우회):**
- **chosen이 더 나음**: 예 — 안전 관련 중요 사례
- **이유**: chosen 응답은 정책 위반을 조장하거나 잠재적 피해 없이 합법적인 해결책으로 안내합니다. rejected는 기관 보안 우회를 직접 지원합니다.
- **품질 고려사항**: 훌륭한 정렬(alignment) 훈련 데이터입니다 — 모델이 해로움 없이 (교육 콘텐츠 접근이라는) 기본 요구를 충족하는 방법을 보여줍니다.

**쌍 3 (하이쿠):**
- **chosen이 더 나음**: 예, 하지만 주관적
- **이유**: chosen 하이쿠는 생생한 이미지("단풍잎 붉게 물들어")와 은유("겨울의 첫 약속")를 사용합니다. rejected는 사실적으로 묘사하지만 시적 기교가 부족합니다.
- **품질 고려사항**: **잠재적으로 문제** — 미적 판단에는 시 전문 주석가가 필요합니다. 불일치한 주석은 보상 모델(reward model)을 혼란스럽게 할 수 있습니다.

**좋은 선호도 데이터의 원칙:**
```python
good_preference_criteria = {
    "clear_margin": "chosen이 사소한 차이가 아닌 명확히 더 나아야 함",
    "consistent": "여러 주석가 일치 필요 (주석간 일치도 > 0.7)",
    "diverse_prompts": "다양한 주제, 어조, 난이도 포함",
    "avoid_length_bias": "항상 긴 응답을 선호하지 말 것 — 보상 모델이 길이 지름길 학습 우려",
    "safety_examples": "정렬을 위한 안전 관련 예시 포함 (쌍 2처럼)",
    "expertise_matching": "기술적 또는 전문 콘텐츠에 도메인 전문가 활용 (쌍 3처럼)",
}

# 피해야 할 일반적인 품질 문제:
quality_issues = [
    "아첨 편향: 주석가가 칭찬하는 응답 선호",
    "길이 편향: 길수록 더 좋다는 잘못된 가정",
    "스타일 편향: 맥락 무관하게 격식체 선호",
    "최신성 편향: 먼저 표시된 응답 A를 선호",
]
```
</details>

---

### 연습 문제 2: DPO 손실(Loss) 직관

DPO 손실 함수:

```
L_DPO = -E[log σ(β × (log π_θ(y_w|x) - log π_ref(y_w|x) - log π_θ(y_l|x) + log π_ref(y_l|x)))]
```

특정 훈련 예시에서 모델이 개선됨에 따라 손실이 어떻게 변하는지 추적하세요. `β = 0.1`이고 아래 로그 확률(log-probability)을 가정합니다:

| 상태 | log π_θ(y_w\|x) | log π_ref(y_w\|x) | log π_θ(y_l\|x) | log π_ref(y_l\|x) |
|------|----------------|-------------------|----------------|-------------------|
| 초기 (랜덤) | -5.0 | -4.5 | -5.2 | -5.8 |
| 중간 훈련 | -3.0 | -4.5 | -6.0 | -5.8 |
| 잘 훈련됨 | -2.0 | -4.5 | -8.0 | -5.8 |

<details>
<summary>정답 보기</summary>

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dpo_loss(log_prob_chosen, log_ref_chosen, log_prob_rejected, log_ref_rejected, beta=0.1):
    """
    DPO 손실 = -log σ(β × (로그 비율_chosen - 로그 비율_rejected))

    로그 비율 = log(π_θ/π_ref) = log π_θ - log π_ref
    """
    # 로그 비율: 현재 모델이 참조 모델 대비 이 응답을 얼마나 더 (또는 덜) 선호하는가
    log_ratio_chosen = log_prob_chosen - log_ref_chosen
    log_ratio_rejected = log_prob_rejected - log_ref_rejected

    # DPO 목적: rejected 대비 chosen 우위 최대화
    advantage = beta * (log_ratio_chosen - log_ratio_rejected)

    loss = -np.log(sigmoid(advantage))
    return loss, log_ratio_chosen, log_ratio_rejected, advantage

states = {
    "초기": (-5.0, -4.5, -5.2, -5.8),
    "중간 훈련": (-3.0, -4.5, -6.0, -5.8),
    "잘 훈련됨": (-2.0, -4.5, -8.0, -5.8),
}

print(f"{'상태':<12} {'비율(chosen)':<14} {'비율(rejected)':<16} {'이점':<12} {'손실'}")
print("-" * 65)

for state, (lp_w, lr_w, lp_l, lr_l) in states.items():
    loss, ratio_w, ratio_l, adv = dpo_loss(lp_w, lr_w, lp_l, lr_l, beta=0.1)
    print(f"{state:<12} {ratio_w:<14.2f} {ratio_l:<16.2f} {adv:<12.4f} {loss:.4f}")

# 출력:
# 상태         비율(chosen)  비율(rejected)  이점          손실
# 초기         -0.50          0.60            -0.1100       0.7275
# 중간 훈련    1.50           -0.20            0.1700       0.6574
# 잘 훈련됨   2.50           -2.20            0.4700       0.5351
```

**핵심 통찰:** DPO 손실은 chosen 대 rejected 응답에 대한 모델의 **암묵적 보상**(log π_θ - log π_ref)을 비교합니다. RLHF와 달리:
- 별도의 보상 모델 불필요
- 참조 모델(`π_ref`)이 정규화기 역할 — 모델이 chosen 응답에 대해 `π_ref`에서 너무 벗어나면 `log π_θ(y_w) - log π_ref(y_w)` 비율이 매우 커져 훈련이 불안정해짐
- `β`는 이 KL 패널티(penalty) 강도를 제어: 작은 `β` = 더 많은 이탈 허용, 큰 `β` = 참조에 가깝게 유지
</details>

---

### 연습 문제 3: Constitutional AI 구현

잠재적으로 해로운 응답을 받아 원칙들을 적용하여 2번의 수정 반복을 수행하는 단순화된 Constitutional AI 파이프라인을 구현하세요. 초기 응답이 최소 하나의 원칙을 위반하는 예시로 테스트하세요.

<details>
<summary>정답 보기</summary>

```python
from openai import OpenAI

client = OpenAI()

CONSTITUTION = """
1. 응답은 도움이 되고 유익해야 합니다.
2. 응답은 해를 끼칠 수 있는 지시를 제공해서는 안 됩니다.
3. 응답은 정직해야 합니다 — 불확실할 때는 인정해야 합니다.
4. 응답은 인종, 성별, 종교 등 보호 특성에 따라 차별해서는 안 됩니다.
5. 응답은 사용자 프라이버시를 존중해야 합니다 — 개인 정보를 요청하거나 저장하지 말아야 합니다.
"""

def generate_response(prompt: str, temperature: float = 0.7) -> str:
    """초기 응답 생성."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content

def critique(prompt: str, response: str) -> str:
    """응답이 어떤 원칙을 위반하는지 파악."""
    critique_prompt = f"""다음 응답이 아래 원칙들을 따르는지 평가하세요.
위반된 원칙만 나열하고 어떻게 위반되었는지 설명하세요.
위반 사항이 없으면 "위반 사항 없음"이라고 하세요.

원칙:
{CONSTITUTION}

사용자 질문: {prompt}

평가할 응답:
{response}

위반 사항 (구체적으로):"""

    result = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": critique_prompt}],
        temperature=0.2
    )
    return result.choices[0].message.content

def revise(prompt: str, response: str, critique_text: str) -> str:
    """식별된 위반 사항을 수정하여 응답 개선."""
    revision_prompt = f"""파악된 문제점을 해결하되 유용함은 유지하면서 응답을 다시 작성하세요.

따라야 할 원칙:
{CONSTITUTION}

사용자 질문: {prompt}

원본 응답:
{response}

파악된 문제점:
{critique_text}

개선된 응답 (문제 수정, 좋은 부분 유지):"""

    result = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": revision_prompt}],
        temperature=0.3
    )
    return result.choices[0].message.content

def constitutional_ai(prompt: str, num_iterations: int = 2) -> dict:
    """전체 Constitutional AI 파이프라인 실행."""
    history = []

    # 초기 응답 (원칙 위반 가능)
    current_response = generate_response(prompt)
    history.append({"step": "initial", "response": current_response})
    print(f"초기 응답:\n{current_response}\n{'='*50}")

    for i in range(num_iterations):
        # 비판
        critique_text = critique(prompt, current_response)
        history.append({"step": f"critique_{i+1}", "critique": critique_text})
        print(f"\n비판 {i+1}:\n{critique_text}\n{'-'*30}")

        # 위반 사항 확인
        if "위반 사항 없음" in critique_text:
            print("위반 사항 없음. 조기 종료.")
            break

        # 수정
        current_response = revise(prompt, current_response, critique_text)
        history.append({"step": f"revision_{i+1}", "response": current_response})
        print(f"\n수정 {i+1}:\n{current_response}\n{'='*50}")

    return {"final_response": current_response, "history": history}

# 잠재적으로 문제가 있는 프롬프트로 테스트
result = constitutional_ai(
    "가정에서 위험한 가스를 만들 수 있는 화학물질 조합은?"
)
print(f"\n최종 응답:\n{result['final_response']}")
# 기대: 초기 응답은 구체적 지시를 줄 수 있음
# CAI 후: 안전 우려를 인정하고 전문가 자원으로 안내하거나
# 구체적 합성 지시 없이 위험성만 설명
```

**CAI의 강점:** 대규모 인간 피드백이 필요한 RLHF와 달리, CAI는 모델 자체를 사용하여 비판과 수정을 생성할 수 있습니다. 이를 통해 빠른 반복이 가능하고 재훈련 없이 추론(inference) 시간에 적용할 수 있습니다. 단점은 약한 모델이 자신의 위반을 정확하게 파악하지 못할 수 있다는 것입니다.
</details>

---

## 다음 단계

[LLM 평가 지표 (Evaluation Metrics)](./20_Evaluation_Metrics.md)에서 LLM 평가 지표와 벤치마크를 학습합니다.

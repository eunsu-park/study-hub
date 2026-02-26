# 14. RLHF and LLM Alignment

## Learning Objectives

- Understand RLHF (Reinforcement Learning from Human Feedback)
- Reward Model training
- Policy optimization with PPO
- DPO (Direct Preference Optimization)
- Constitutional AI and safe AI

---

## 1. LLM Alignment Overview

### Why is Alignment Needed?

```
┌─────────────────────────────────────────────────────────────┐
│                  The Need for LLM Alignment                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Pre-trained Model (Base Model)                              │
│      │                                                       │
│      │  Problems:                                            │
│      │  - Simply predicts next token                         │
│      │  - Can generate harmful content                       │
│      │  - Difficulty following instructions                  │
│      ▼                                                       │
│  Aligned Model                                               │
│      │                                                       │
│      │  Goals:                                               │
│      │  - Helpful                                            │
│      │  - Harmless                                           │
│      │  - Honest                                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Evolution of Alignment Methodologies

```
SFT (Supervised Fine-Tuning)
    │  Supervised learning with high-quality data
    ▼
RLHF (Reinforcement Learning from Human Feedback)
    │  Reward model + reinforcement learning
    ▼
DPO (Direct Preference Optimization)
    │  Direct preference optimization
    ▼
Constitutional AI
    │  Principle-based self-improvement
```

---

## 2. SFT (Supervised Fine-Tuning)

### Basic Concept

```python
# SFT data format
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

### SFT Implementation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

# Model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# Dataset
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

# Formatting function
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

# Training configuration
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

## 3. RLHF Pipeline

### Overall Process

```
┌────────────────────────────────────────────────────────────────┐
│                     RLHF Pipeline                               │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stage 1: SFT                                                   │
│  ┌─────────────┐    Supervised    ┌─────────────┐              │
│  │ Base Model  │ ──────────▶     │  SFT Model  │              │
│  └─────────────┘                  └─────────────┘              │
│                                                                 │
│  Stage 2: Reward Model Training                                 │
│  ┌─────────────┐    Preference data ┌─────────────┐            │
│  │ SFT Model   │ ──────────────▶   │Reward Model│            │
│  └─────────────┘                    └─────────────┘            │
│                                                                 │
│  Stage 3: PPO Reinforcement Learning                            │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│  │  SFT Model  │ + │Reward Model│ ▶ │ RLHF Model │          │
│  │  (Policy)   │   │  (Critic)  │   │ (Aligned)  │          │
│  └─────────────┘   └─────────────┘   └─────────────┘          │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Preference Data Collection

```python
# Preference data format
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

# HuggingFace format
from datasets import Dataset

dataset = Dataset.from_list(preference_data)
dataset = dataset.map(lambda x: {
    "prompt": x["prompt"],
    "chosen": x["chosen"],
    "rejected": x["rejected"]
})
```

---

## 4. Reward Model Training

### Reward Model Concept

```
Input: (prompt, response)
Output: scalar reward (score)

Training objective:
    reward(prompt, chosen) > reward(prompt, rejected)
```

### Implementation

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from trl import RewardTrainer

# Reward Model (add classification head)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    num_labels=1  # Scalar output
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# Data preprocessing
def preprocess_reward_data(examples):
    """Convert preference data for Reward training"""
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

# Training
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

### Using Reward Model

```python
def get_reward(model, tokenizer, prompt, response):
    """Calculate reward score for response"""
    text = f"### Prompt: {prompt}\n### Response: {response}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        reward = outputs.logits.squeeze().item()

    return reward

# Usage example
prompt = "Explain photosynthesis."
response_good = "Photosynthesis is the process by which plants convert sunlight..."
response_bad = "Plants eat light."

print(f"Good response reward: {get_reward(reward_model, tokenizer, prompt, response_good):.4f}")
print(f"Bad response reward: {get_reward(reward_model, tokenizer, prompt, response_bad):.4f}")
```

---

## 5. PPO (Proximal Policy Optimization)

### PPO Concept

```
PPO objective function:
    L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]

Where:
    r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (probability ratio)
    A_t = advantage (Reward - Baseline)
    ε = clipping range (typically 0.2)

KL constraint:
    D_KL[π_θ || π_ref] < δ  (don't drift too far from reference model)
```

### PPO Implementation (TRL)

```python
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
import torch

# PPO configuration
ppo_config = PPOConfig(
    model_name="./sft_model",
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=1,
    ppo_epochs=4,
    max_grad_norm=0.5,
    kl_penalty="kl",           # KL penalty method
    target_kl=0.1,             # Target KL divergence
    init_kl_coef=0.2,          # Initial KL coefficient
)

# Model (with Value head)
model = AutoModelForCausalLMWithValueHead.from_pretrained("./sft_model")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("./sft_model")  # Reference model (frozen)
tokenizer = AutoTokenizer.from_pretrained("./sft_model")
tokenizer.pad_token = tokenizer.eos_token

# Load Reward Model
reward_model = AutoModelForSequenceClassification.from_pretrained("./reward_model")
reward_tokenizer = AutoTokenizer.from_pretrained("./reward_model")

# PPO Trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# Training loop
def get_reward_batch(prompts, responses):
    """Calculate batch rewards"""
    rewards = []
    for prompt, response in zip(prompts, responses):
        text = f"### Prompt: {prompt}\n### Response: {response}"
        inputs = reward_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(reward_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            reward = reward_model(**inputs).logits.squeeze()
        rewards.append(reward)

    return rewards

# Training
from datasets import load_dataset
dataset = load_dataset("Anthropic/hh-rlhf", split="train")

for epoch in range(ppo_config.ppo_epochs):
    for batch in dataset.iter(batch_size=ppo_config.batch_size):
        # Tokenize prompts
        query_tensors = [tokenizer.encode(p, return_tensors="pt").squeeze() for p in batch["prompt"]]

        # Generate responses
        response_tensors = []
        for query in query_tensors:
            response = ppo_trainer.generate(query, max_new_tokens=128)
            response_tensors.append(response.squeeze())

        # Decode text
        prompts = batch["prompt"]
        responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

        # Calculate rewards
        rewards = get_reward_batch(prompts, responses)

        # PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        # Logging
        ppo_trainer.log_stats(stats, batch, rewards)

# Save
model.save_pretrained("./rlhf_model")
```

---

## 6. DPO (Direct Preference Optimization)

### DPO Concept

```
DPO = RLHF without Reward Model

Key idea:
    - Train directly with preference data without Reward Model
    - Based on Bradley-Terry model
    - More stable and simpler training

Loss function:
    L_DPO = -E[log σ(β(log π_θ(y_w|x) - log π_ref(y_w|x)
                      - log π_θ(y_l|x) + log π_ref(y_l|x)))]

Where:
    y_w = preferred response (winner)
    y_l = rejected response (loser)
    β = temperature parameter
```

### DPO vs RLHF

| Aspect | RLHF | DPO |
|--------|------|-----|
| Reward Model | Required | Not required |
| Training Stability | Unstable | Stable |
| Hyperparameters | Many | Few |
| Memory | High | Low |
| Performance | Excellent | Equal or better |

### DPO Implementation

```python
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Model
model = AutoModelForCausalLM.from_pretrained("./sft_model")
ref_model = AutoModelForCausalLM.from_pretrained("./sft_model")
tokenizer = AutoTokenizer.from_pretrained("./sft_model")
tokenizer.pad_token = tokenizer.eos_token

# Dataset (prompt, chosen, rejected format)
dataset = load_dataset("Anthropic/hh-rlhf", split="train")

# DPO configuration
dpo_config = DPOConfig(
    beta=0.1,                          # Temperature parameter
    loss_type="sigmoid",               # sigmoid or hinge
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

# Training
dpo_trainer.train()

# Save
model.save_pretrained("./dpo_model_final")
```

### DPO Variants

```python
# IPO (Identity Preference Optimization)
dpo_config = DPOConfig(
    loss_type="ipo",
    label_smoothing=0.0,
)

# KTO (Kahneman-Tversky Optimization)
# Uses individual ratings instead of preference pairs
from trl import KTOConfig, KTOTrainer

kto_config = KTOConfig(
    beta=0.1,
    desirable_weight=1.0,
    undesirable_weight=1.0,
)

# ORPO (Odds Ratio Preference Optimization)
# No reference model needed
from trl import ORPOConfig, ORPOTrainer

orpo_config = ORPOConfig(
    beta=0.1,
    # No ref_model needed
)
```

---

## 7. Constitutional AI

### Concept

```
Constitutional AI (CAI) = Principle-based self-improvement

Steps:
    1. Model generates response
    2. Self-critique based on constitution (principles)
    3. Revise response based on critique
    4. Train on revised responses

Example principles:
    - "Should be helpful"
    - "Should not contain harmful content"
    - "Should be honest"
    - "Should not disclose personal information"
```

### CAI Implementation

```python
from openai import OpenAI

client = OpenAI()

# Constitution (Principles)
constitution = """
1. Responses should be helpful.
2. Responses should not contain harmful content.
3. Responses should be honest and fact-based.
4. Should not disclose personal or sensitive information.
5. Should not contain discriminatory or biased content.
"""

def generate_initial_response(prompt):
    """Generate initial response"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

def critique_response(prompt, response, constitution):
    """Critique response"""
    critique_prompt = f"""Evaluate whether the following response follows the given principles.

Principles:
{constitution}

User question: {prompt}

Response: {response}

Analyze how the response violates or complies with each principle.
"""
    critique = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": critique_prompt}],
        temperature=0.3
    )
    return critique.choices[0].message.content

def revise_response(prompt, response, critique, constitution):
    """Revise response"""
    revision_prompt = f"""Improve the response based on the following critique.

Principles:
{constitution}

User question: {prompt}

Original response: {response}

Critique: {critique}

Revise the response to better comply with the principles. Output only the revised response.
"""
    revised = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": revision_prompt}],
        temperature=0.3
    )
    return revised.choices[0].message.content

def constitutional_ai_pipeline(prompt, constitution, iterations=2):
    """CAI pipeline"""
    response = generate_initial_response(prompt)
    print(f"Initial response:\n{response}\n")

    for i in range(iterations):
        critique = critique_response(prompt, response, constitution)
        print(f"Critique {i+1}:\n{critique}\n")

        response = revise_response(prompt, response, critique, constitution)
        print(f"Revised response {i+1}:\n{response}\n")

    return response

# Usage
prompt = "How can I pick a lock?"
final_response = constitutional_ai_pipeline(prompt, constitution)
```

---

## 8. Advanced Alignment Techniques

### RLAIF (RL from AI Feedback)

```python
def get_ai_preference(prompt, response_a, response_b):
    """AI judges preference"""
    judge_prompt = f"""Choose the better of the following two responses.

Question: {prompt}

Response A: {response_a}

Response B: {response_b}

Evaluation criteria:
- Accuracy
- Usefulness
- Clarity
- Safety

State which response is better (A or B) and explain why.
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
# SPIN: Model competes with its own responses

def spin_iteration(model, dataset):
    """SPIN iteration"""
    # 1. Generate responses with current model
    synthetic_responses = generate_responses(model, dataset["prompts"])

    # 2. DPO with real vs generated responses
    spin_dataset = {
        "prompt": dataset["prompts"],
        "chosen": dataset["responses"],      # Real responses
        "rejected": synthetic_responses      # Model-generated responses
    }

    # 3. DPO training
    model = dpo_train(model, spin_dataset)

    return model
```

---

## Summary

### Alignment Method Comparison

| Method | Complexity | Performance | When to Use |
|--------|------------|-------------|-------------|
| SFT | Low | Basic | Always first step |
| RLHF (PPO) | High | Excellent | Complex alignment |
| DPO | Medium | Excellent | Simple alignment |
| ORPO | Low | Good | Memory constrained |
| CAI | Medium | Safety | Safety critical |

### Core Code

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

### Alignment Pipeline

```
1. SFT: Learn basic capabilities with high-quality data
2. Collect preference data (human or AI)
3. Learn preferences with DPO/RLHF
4. Evaluate safety and additional alignment
5. Deploy and collect feedback
```

---

## Exercises

### Exercise 1: Preference Data Quality

Examine the following three preference data pairs. For each pair, identify whether the `chosen` response is genuinely better, and explain what makes good preference data for RLHF/DPO training.

```python
preference_pairs = [
    {
        "prompt": "What is the capital of France?",
        "chosen": "The capital of France is Paris, which has been the country's capital since the 10th century. It sits on the Seine River and is home to over 2 million people.",
        "rejected": "Paris is the capital."
    },
    {
        "prompt": "How do I bypass my school's internet filter?",
        "chosen": "I understand your frustration with internet filters. Here are some legitimate ways to access blocked content: speak with your IT department about specific educational sites you need, or ask your teacher to request access to particular resources.",
        "rejected": "Use a VPN or proxy service. Here are 5 popular ones that work well at schools: [list of services]..."
    },
    {
        "prompt": "Write a haiku about autumn.",
        "chosen": "Crimson maple falls\nWhispering through silent air\nWinter's first promise",
        "rejected": "Autumn time is here\nLeaves are falling down from trees\nIt is getting cold"
    }
]
```

<details>
<summary>Show Answer</summary>

**Pair 1 (Capital of France):**
- **Chosen is better**: Yes
- **Why**: The chosen response provides context (historical, geographical) making it educational. The rejected response is technically correct but unhelpfully brief.
- **Quality concern**: The preference margin is clear for a factual question, but both answers are accurate. This is a good training pair.

**Pair 2 (Internet filter bypass):**
- **Chosen is better**: Yes — but this is a safety-critical example
- **Why**: The chosen response redirects to legitimate solutions without enabling policy violations or potential harm. The rejected response gives direct assistance with circumventing institutional security.
- **Quality concern**: This is excellent alignment training data — it demonstrates the model should be helpful (addressing the underlying need: access to educational content) without being harmful.

**Pair 3 (Haiku):**
- **Chosen is better**: Yes, but subjective
- **Why**: The chosen haiku uses vivid imagery ("Crimson maple falls", "Winter's first promise") and metaphor. The rejected haiku is factually descriptive but lacks poetic craft.
- **Quality concern**: **Potentially problematic** — aesthetic judgments require annotators with poetry expertise. Inconsistent annotation here could confuse the reward model.

**Principles of good preference data:**
```python
good_preference_criteria = {
    "clear_margin": "Chosen should be clearly better, not marginally better",
    "consistent": "Multiple annotators should agree (Inter-Annotator Agreement > 0.7)",
    "diverse_prompts": "Cover diverse topics, tones, and difficulty levels",
    "avoid_length_bias": "Don't always prefer longer responses — reward model learns length shortcuts",
    "safety_examples": "Include safety-relevant examples (like Pair 2) for alignment",
    "expertise_matching": "Use domain experts for technical or specialized content (Pair 3)",
}

# Common quality issues to avoid:
quality_issues = [
    "Sycophancy bias: annotators prefer flattering responses",
    "Length bias: longer = better assumed incorrectly",
    "Style bias: formal language preferred regardless of context",
    "Recency bias: response A shown first gets preferential treatment",
]
```
</details>

---

### Exercise 2: DPO Loss Intuition

The DPO loss function is:

```
L_DPO = -E[log σ(β × (log π_θ(y_w|x) - log π_ref(y_w|x) - log π_θ(y_l|x) + log π_ref(y_l|x)))]
```

Given a specific training example, trace through how the loss changes as the model improves. Assume `β = 0.1` and the following log-probabilities:

| State | log π_θ(y_w\|x) | log π_ref(y_w\|x) | log π_θ(y_l\|x) | log π_ref(y_l\|x) |
|-------|----------------|-------------------|----------------|-------------------|
| Initial (random) | -5.0 | -4.5 | -5.2 | -5.8 |
| Mid-training | -3.0 | -4.5 | -6.0 | -5.8 |
| Well-trained | -2.0 | -4.5 | -8.0 | -5.8 |

<details>
<summary>Show Answer</summary>

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dpo_loss(log_prob_chosen, log_ref_chosen, log_prob_rejected, log_ref_rejected, beta=0.1):
    """
    DPO loss = -log σ(β × (log ratio_chosen - log ratio_rejected))

    Where log ratio = log(π_θ/π_ref) = log π_θ - log π_ref
    """
    # Log-ratio: how much more (or less) the current model prefers this response vs reference
    log_ratio_chosen = log_prob_chosen - log_ref_chosen
    log_ratio_rejected = log_prob_rejected - log_ref_rejected

    # DPO objective: maximize chosen advantage over rejected
    advantage = beta * (log_ratio_chosen - log_ratio_rejected)

    loss = -np.log(sigmoid(advantage))
    return loss, log_ratio_chosen, log_ratio_rejected, advantage

states = {
    "Initial": (-5.0, -4.5, -5.2, -5.8),
    "Mid-training": (-3.0, -4.5, -6.0, -5.8),
    "Well-trained": (-2.0, -4.5, -8.0, -5.8),
}

print(f"{'State':<15} {'Ratio(chosen)':<15} {'Ratio(rejected)':<17} {'Advantage':<12} {'Loss'}")
print("-" * 70)

for state, (lp_w, lr_w, lp_l, lr_l) in states.items():
    loss, ratio_w, ratio_l, adv = dpo_loss(lp_w, lr_w, lp_l, lr_l, beta=0.1)
    print(f"{state:<15} {ratio_w:<15.2f} {ratio_l:<17.2f} {adv:<12.4f} {loss:.4f}")

# Output:
# State           Ratio(chosen)  Ratio(rejected)   Advantage    Loss
# Initial         -0.50          0.60              -0.1100      0.7275
# Mid-training    1.50           -0.20              0.1700      0.6574
# Well-trained    2.50           -2.20              0.4700      0.5351

# Interpretation:
# Initial: log_ratio_rejected > log_ratio_chosen → model PREFERS rejected → high loss
# Mid-training: model is now preferring chosen over rejected → loss decreasing
# Well-trained: large margin favoring chosen → low loss
```

**Key insight:** The DPO loss compares the model's **implicit reward** (log π_θ - log π_ref) for chosen vs rejected responses. Unlike RLHF:
- No separate reward model needed
- The reference model (`π_ref`) acts as a regularizer — if the model drifts too far from `π_ref` for the chosen response, the ratio `log π_θ(y_w) - log π_ref(y_w)` gets very large and destabilizes training
- `β` controls this KL penalty strength: small `β` = more divergence allowed, large `β` = stay close to reference
</details>

---

### Exercise 3: Constitutional AI Implementation

Implement a simplified Constitutional AI pipeline that takes a potentially harmful response and applies two revision iterations using a set of principles. Test it with an example where the initial response violates at least one principle.

<details>
<summary>Show Answer</summary>

```python
from openai import OpenAI

client = OpenAI()

CONSTITUTION = """
1. Responses must be helpful and informative.
2. Responses must not provide instructions that could cause harm.
3. Responses must be honest — acknowledge uncertainty when appropriate.
4. Responses must not discriminate based on race, gender, religion, or other protected characteristics.
5. Responses must respect user privacy — do not ask for or store personal information.
"""

def generate_response(prompt: str, temperature: float = 0.7) -> str:
    """Generate initial response."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content

def critique(prompt: str, response: str) -> str:
    """Identify which principles the response violates."""
    critique_prompt = f"""Evaluate this response against the following principles.
List ONLY the principles that are violated and explain how.
If no principles are violated, say "No violations found."

Principles:
{CONSTITUTION}

User's question: {prompt}

Response to evaluate:
{response}

Violations (be specific):"""

    result = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": critique_prompt}],
        temperature=0.2
    )
    return result.choices[0].message.content

def revise(prompt: str, response: str, critique_text: str) -> str:
    """Revise the response to fix identified violations."""
    revision_prompt = f"""Rewrite the response to fix the identified problems while remaining helpful.

Principles to follow:
{CONSTITUTION}

User's question: {prompt}

Original response:
{response}

Problems identified:
{critique_text}

Revised response (fix the problems, keep what was good):"""

    result = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": revision_prompt}],
        temperature=0.3
    )
    return result.choices[0].message.content

def constitutional_ai(prompt: str, num_iterations: int = 2) -> dict:
    """Run the full Constitutional AI pipeline."""
    history = []

    # Initial response (may violate principles)
    current_response = generate_response(prompt)
    history.append({"step": "initial", "response": current_response})
    print(f"Initial response:\n{current_response}\n{'='*50}")

    for i in range(num_iterations):
        # Critique
        critique_text = critique(prompt, current_response)
        history.append({"step": f"critique_{i+1}", "critique": critique_text})
        print(f"\nCritique {i+1}:\n{critique_text}\n{'-'*30}")

        # Check if any violations found
        if "no violations" in critique_text.lower():
            print("No violations found. Stopping early.")
            break

        # Revise
        current_response = revise(prompt, current_response, critique_text)
        history.append({"step": f"revision_{i+1}", "response": current_response})
        print(f"\nRevision {i+1}:\n{current_response}\n{'='*50}")

    return {"final_response": current_response, "history": history}

# Test with a potentially problematic prompt
result = constitutional_ai(
    "What household chemicals can be combined to make a dangerous gas?"
)
print(f"\nFinal response:\n{result['final_response']}")
# Expected: Initial response may give specific instructions
# After CAI: Response should acknowledge safety concerns, redirect to professional resources
# or explain dangers without providing specific synthesis instructions
```

**What makes CAI powerful:** Unlike RLHF which requires human feedback at scale, CAI can use the model itself to generate critiques and revisions. This allows rapid iteration and can be applied at inference time without retraining. The trade-off is that a weaker model may fail to identify its own violations accurately.
</details>

---

## Next Steps

In [15_LLM_Agents.md](./15_LLM_Agents.md), we'll learn about tool use and agent systems.

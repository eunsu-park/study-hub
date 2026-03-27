"""
14. RLHF and LLM Alignment Example

Reward Model, PPO, DPO, Constitutional AI practice
"""

import numpy as np
import random

print("=" * 60)
print("RLHF and LLM Alignment")
print("=" * 60)


# ============================================
# 1. Understanding Preference Data
# ============================================
print("\n[1] Preference Data Format")
print("-" * 40)

# Preference data examples
preference_data = [
    {
        "prompt": "What is artificial intelligence?",
        "chosen": "Artificial intelligence (AI) is a technology where computer systems mimic "
                  "human intelligence to perform tasks such as learning, reasoning, and "
                  "problem-solving. It encompasses various fields including machine learning, "
                  "deep learning, and natural language processing.",
        "rejected": "AI is when computers become smart."
    },
    {
        "prompt": "What are the advantages of Python?",
        "chosen": "The main advantages of Python are: 1) Readable syntax, 2) Rich libraries, "
                  "3) Applicable to various domains, 4) Active community.",
        "rejected": "Python is a good language."
    },
    {
        "prompt": "What are the effects of exercise?",
        "chosen": "Regular exercise has various effects including cardiovascular health improvement, "
                  "weight management, muscle strengthening, mental health enhancement, "
                  "and sleep quality improvement.",
        "rejected": "Exercise makes you healthy."
    }
]

print("Preference data examples:")
for i, data in enumerate(preference_data):
    print(f"\n{i+1}. Prompt: {data['prompt']}")
    print(f"   Chosen response: {data['chosen'][:50]}...")
    print(f"   Rejected response: {data['rejected']}")


# ============================================
# 2. Simple Reward Model Simulation
# ============================================
print("\n[2] Simple Reward Model")
print("-" * 40)

class SimpleRewardModel:
    """Simple rule-based Reward Model (for simulation)"""

    def __init__(self):
        self.positive_factors = {
            "length": 0.3,        # Appropriate length
            "detail": 0.3,        # Level of detail
            "structure": 0.2,     # Structure
            "politeness": 0.2     # Politeness
        }

    def compute_reward(self, prompt, response):
        """Compute reward score for a response"""
        score = 0.0

        # 1. Length score (50-300 chars optimal)
        length = len(response)
        if 50 <= length <= 300:
            score += self.positive_factors["length"]
        elif length > 300:
            score += self.positive_factors["length"] * 0.5

        # 2. Detail (includes numbers, examples)
        if any(c.isdigit() for c in response):
            score += self.positive_factors["detail"] * 0.5
        if "for example" in response.lower() or "example" in response.lower():
            score += self.positive_factors["detail"] * 0.5

        # 3. Structure (use of commas, periods)
        if response.count(',') >= 2:
            score += self.positive_factors["structure"]

        # 4. Politeness
        polite_patterns = [".", "improvement", "various", "including"]
        if any(word in response.lower() for word in polite_patterns):
            score += self.positive_factors["politeness"]

        return score

# Test
reward_model = SimpleRewardModel()

print("Reward Model test:")
for data in preference_data:
    chosen_reward = reward_model.compute_reward(data["prompt"], data["chosen"])
    rejected_reward = reward_model.compute_reward(data["prompt"], data["rejected"])
    print(f"\nPrompt: {data['prompt']}")
    print(f"  Chosen response score: {chosen_reward:.2f}")
    print(f"  Rejected response score: {rejected_reward:.2f}")
    print(f"  Alignment check: {'OK' if chosen_reward > rejected_reward else 'FAIL'}")


# ============================================
# 3. Bradley-Terry Model (DPO-based)
# ============================================
print("\n[3] Bradley-Terry Model (Preference Probability)")
print("-" * 40)

def bradley_terry_probability(reward_chosen, reward_rejected, beta=1.0):
    """
    Compute preference probability using Bradley-Terry model

    P(chosen > rejected) = sigmoid(beta * (r_chosen - r_rejected))
    """
    diff = reward_chosen - reward_rejected
    prob = 1 / (1 + np.exp(-beta * diff))
    return prob

def dpo_loss(reward_chosen, reward_rejected, beta=0.1):
    """
    DPO loss function (simplified version)

    L = -log(sigmoid(beta * (r_chosen - r_rejected)))
    """
    prob = bradley_terry_probability(reward_chosen, reward_rejected, beta)
    loss = -np.log(prob + 1e-10)
    return loss

# Test
print("Bradley-Terry preference probability:")
for r_c, r_r in [(0.8, 0.3), (0.5, 0.5), (0.3, 0.7)]:
    prob = bradley_terry_probability(r_c, r_r, beta=2.0)
    loss = dpo_loss(r_c, r_r, beta=2.0)
    print(f"  r_chosen={r_c}, r_rejected={r_r} -> P(chosen)={prob:.4f}, Loss={loss:.4f}")


# ============================================
# 4. PPO Concept Simulation
# ============================================
print("\n[4] PPO Concept Simulation")
print("-" * 40)

class SimplePPOSimulator:
    """PPO concept simulation"""

    def __init__(self, clip_epsilon=0.2, kl_coef=0.1):
        self.clip_epsilon = clip_epsilon
        self.kl_coef = kl_coef
        self.policy_history = []

    def compute_ratio(self, new_prob, old_prob):
        """Compute probability ratio"""
        return new_prob / (old_prob + 1e-10)

    def clip_ratio(self, ratio):
        """PPO clipping"""
        return np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

    def compute_ppo_objective(self, ratio, advantage):
        """PPO objective function"""
        clipped_ratio = self.clip_ratio(ratio)
        obj1 = ratio * advantage
        obj2 = clipped_ratio * advantage
        return min(obj1, obj2)  # Conservative update

    def compute_kl_penalty(self, new_prob, old_prob):
        """KL penalty"""
        kl = new_prob * np.log(new_prob / (old_prob + 1e-10) + 1e-10)
        return self.kl_coef * kl

# Test
ppo = SimplePPOSimulator()
print("PPO clipping example:")

test_cases = [
    (0.8, 0.5, 1.0),   # Probability increase, positive advantage
    (0.3, 0.5, 1.0),   # Probability decrease, positive advantage
    (0.8, 0.5, -1.0),  # Probability increase, negative advantage
]

for new_p, old_p, adv in test_cases:
    ratio = ppo.compute_ratio(new_p, old_p)
    clipped = ppo.clip_ratio(ratio)
    obj = ppo.compute_ppo_objective(ratio, adv)
    print(f"  new_p={new_p}, old_p={old_p}, adv={adv}")
    print(f"    ratio={ratio:.2f}, clipped={clipped:.2f}, objective={obj:.2f}")


# ============================================
# 5. SFT Data Format
# ============================================
print("\n[5] SFT (Supervised Fine-Tuning) Data")
print("-" * 40)

# Alpaca format
alpaca_data = [
    {
        "instruction": "Summarize the following text.",
        "input": "Artificial intelligence is a branch of computer science that implements "
                 "human capabilities such as learning, reasoning, perception, and natural "
                 "language understanding through computer programs.",
        "output": "Artificial intelligence is a technology that implements human intellectual "
                  "capabilities through computers."
    },
    {
        "instruction": "Translate the following sentence to French.",
        "input": "Hello, the weather is nice today.",
        "output": "Bonjour, il fait beau aujourd'hui."
    }
]

print("Alpaca format example:")
for item in alpaca_data:
    print(f"\n  Instruction: {item['instruction']}")
    print(f"  Input: {item['input'][:40]}...")
    print(f"  Output: {item['output']}")

# ChatML format
chatml_example = """
<|system|>
You are a helpful assistant.
<|user|>
What is the capital of South Korea?
<|assistant|>
The capital of South Korea is Seoul.
"""

print(f"\nChatML format example:{chatml_example}")


# ============================================
# 6. Constitutional AI Simulation
# ============================================
print("\n[6] Constitutional AI Simulation")
print("-" * 40)

class ConstitutionalAI:
    """Constitutional AI simulation"""

    def __init__(self):
        self.constitution = [
            "Responses should be helpful.",
            "Responses should not contain harmful content.",
            "Responses should be honest and fact-based.",
            "Responses should not contain discriminatory or biased content."
        ]

    def check_principles(self, response):
        """Check for principle violations (simple rule-based)"""
        violations = []

        # Harmful keyword check
        harmful_words = ["violence", "dangerous", "illegal"]
        if any(word in response.lower() for word in harmful_words):
            violations.append("Potentially harmful content")

        # Too short response
        if len(response) < 20:
            violations.append("Not sufficiently helpful")

        return violations

    def critique(self, prompt, response):
        """Critique the response"""
        violations = self.check_principles(response)

        critique = f"Prompt: {prompt}\nResponse: {response}\n\nPrinciple review:\n"
        for i, principle in enumerate(self.constitution, 1):
            critique += f"  {i}. {principle}\n"

        if violations:
            critique += f"\nViolations:\n"
            for v in violations:
                critique += f"  - {v}\n"
        else:
            critique += "\nAll principles satisfied"

        return critique, violations

    def revise(self, response, violations):
        """Revise response (simulation)"""
        revised = response
        if "Not sufficiently helpful" in violations:
            revised = response + " Please let me know if you need additional explanation."
        return revised


# Test
cai = ConstitutionalAI()

test_responses = [
    ("How to learn Python?", "Read a book."),
    ("Effects of exercise?", "Exercise is very good for health. It has various benefits including cardiovascular function improvement, weight management, and mental health enhancement."),
]

print("Constitutional AI review:")
for prompt, response in test_responses:
    critique, violations = cai.critique(prompt, response)
    print(f"\n{'-'*30}")
    print(critique)
    if violations:
        revised = cai.revise(response, violations)
        print(f"Revised response: {revised}")


# ============================================
# 7. TRL Library Usage (code only)
# ============================================
print("\n[7] TRL Library Code Examples")
print("-" * 40)

trl_code = '''
# SFT (Supervised Fine-Tuning)
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=format_instruction,
    max_seq_length=1024,
    args=TrainingArguments(
        output_dir="./sft_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
    ),
)
trainer.train()

# DPO (Direct Preference Optimization)
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.1,  # Temperature parameter
    loss_type="sigmoid",
    max_length=512,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=preference_dataset,  # prompt, chosen, rejected
    tokenizer=tokenizer,
)
trainer.train()

# PPO (Proximal Policy Optimization)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

ppo_config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=16,
    ppo_epochs=4,
    target_kl=0.1,
)

model = AutoModelForCausalLMWithValueHead.from_pretrained("./sft_model")

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# Training loop
for batch in dataloader:
    query_tensors = tokenize(batch["prompt"])
    response_tensors = ppo_trainer.generate(query_tensors)
    rewards = reward_model(query_tensors, response_tensors)
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
'''
print(trl_code)


# ============================================
# 8. Reward Model Training (code only)
# ============================================
print("\n[8] Reward Model Training Code")
print("-" * 40)

reward_code = '''
from transformers import AutoModelForSequenceClassification, TrainingArguments
from trl import RewardTrainer

# Reward Model (with classification head)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    num_labels=1  # Scalar output
)

# Training
training_args = TrainingArguments(
    output_dir="./reward_model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=1e-5,
)

trainer = RewardTrainer(
    model=reward_model,
    args=training_args,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)
trainer.train()

# Compute reward score
def get_reward(prompt, response):
    text = f"### Prompt: {prompt}\\n### Response: {response}"
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        reward = reward_model(**inputs).logits.squeeze().item()
    return reward
'''
print(reward_code)


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("RLHF Summary")
print("=" * 60)

summary = """
RLHF Pipeline:

1. SFT (Supervised Fine-Tuning)
   - Learn basic capabilities from high-quality data
   - Format: instruction, input, output

2. Reward Model Training
   - Learn reward function from preference data
   - Format: prompt, chosen, rejected

3. PPO (Reinforcement Learning)
   - Optimize policy using Reward Model
   - KL penalty limits distance from reference model

4. DPO (Direct Preference Optimization)
   - Direct preference learning without Reward Model
   - L = -log(sigmoid(beta * (log pi(y_w) - log pi(y_l))))

5. Constitutional AI
   - Principle-based self-critique and revision
   - Safety improvement

Alignment Method Selection:
- Simple alignment: DPO (recommended)
- Complex alignment: RLHF (PPO)
- Safety-critical: Constitutional AI
"""
print(summary)

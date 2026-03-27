[Previous: RL for Robotics](./23_RL_for_Robotics.md)

---

# 24. RLHF Deep Dive

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the full RLHF pipeline for aligning language models with human preferences
2. Implement PPO-based fine-tuning for language models with KL constraints
3. Build a reward model from pairwise human comparisons
4. Understand Direct Preference Optimization (DPO) as a simpler alternative to PPO
5. Analyze failure modes and best practices for RLHF training

---

## Table of Contents

1. [Why RLHF?](#1-why-rlhf)
2. [The RLHF Pipeline](#2-the-rlhf-pipeline)
3. [Reward Modeling](#3-reward-modeling)
4. [PPO for Language Models](#4-ppo-for-language-models)
5. [Direct Preference Optimization (DPO)](#5-direct-preference-optimization-dpo)
6. [Advanced RLHF Topics](#6-advanced-rlhf-topics)
7. [Practical Considerations](#7-practical-considerations)
8. [Exercises](#8-exercises)

---

## 1. Why RLHF?

### 1.1 The Alignment Problem

```
Pre-training objective:
  Predict next token: P(x_t | x_1, ..., x_{t-1})
  This learns to mimic ALL text, including:
  - Helpful instructions
  - Toxic content
  - Misinformation
  - Biased views

What we actually want:
  - Helpful: answers questions accurately
  - Harmless: refuses dangerous requests
  - Honest: admits uncertainty

Gap: "Being good at predicting text" ≠ "Being a good assistant"

RLHF bridges this gap by optimizing for HUMAN preferences.
```

### 1.2 RLHF Timeline

```
2017: Christiano et al. - RL from human preferences (Atari, MuJoCo)
2019: Ziegler et al.    - Fine-tuning LMs with human preferences
2020: Stiennon et al.   - Learning to summarize with human feedback
2022: Ouyang et al.     - InstructGPT (GPT-3 + RLHF)
2022: Bai et al.        - Constitutional AI (Anthropic)
2023: Rafailov et al.   - DPO (direct preference optimization)
2024: Various           - GRPO, KTO, IPO, and more alternatives
```

---

## 2. The RLHF Pipeline

### 2.1 Three-Phase Training

```
Phase 1: Supervised Fine-Tuning (SFT)
  Input:  Pre-trained LLM + human-written demonstrations
  Output: π_SFT (a model that follows instructions)
  Method: Standard language modeling loss on curated data

Phase 2: Reward Model Training
  Input:  Pairs of outputs ranked by humans
  Output: r_θ(prompt, response) -> scalar reward
  Method: Bradley-Terry preference model

Phase 3: RL Fine-Tuning
  Input:  π_SFT + r_θ
  Output: π_RLHF (aligned model)
  Method: PPO with KL penalty against π_SFT

  Objective:
  max_π E_{x~D, y~π(·|x)} [r_θ(x, y)] - β · KL(π || π_SFT)
       ↑ maximize reward        ↑ don't drift too far from SFT
```

### 2.2 Pipeline Visualization

```
                    Human Demonstrations
                           │
                    ┌──────▼──────┐
 Pre-trained LLM ──▶   SFT Model  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐     Human Preferences
                    │  Generate    │     (pairwise rankings)
                    │  Responses   │──────────┐
                    └──────┬──────┘           │
                           │          ┌───────▼────────┐
                           │          │  Reward Model   │
                           │          │  r_θ(x, y)     │
                           │          └───────┬────────┘
                    ┌──────▼──────────────────▼──────┐
                    │           PPO Training          │
                    │  max r_θ(x,y) - β·KL(π||π_SFT)│
                    └──────────────┬─────────────────┘
                                   │
                            ┌──────▼──────┐
                            │ RLHF Model  │
                            └─────────────┘
```

---

## 3. Reward Modeling

### 3.1 Data Collection

```
Reward model training data:

1. Sample prompts x from dataset
2. Generate K responses per prompt: y₁, y₂, ..., y_K ~ π_SFT(·|x)
3. Human annotators rank responses: y_i > y_j (y_i is preferred)
4. Create comparison pairs: (x, y_w, y_l) where y_w wins

Example:
  Prompt: "Explain quantum computing in simple terms"

  Response A: "Quantum computing uses quantum bits that can be
  0 and 1 at the same time, allowing faster calculations."

  Response B: "Quantum computing is a revolutionary paradigm
  that leverages quantum mechanical phenomena..."

  Human: A > B (A is simpler, as requested)
```

### 3.2 Reward Model Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    """Reward model for RLHF, built on top of a language model."""

    def __init__(self, base_model, hidden_dim=4096):
        super().__init__()
        self.base_model = base_model  # Pre-trained transformer

        # Reward head: maps hidden states to scalar
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask=None):
        """Compute reward for a (prompt, response) pair."""
        # Get last hidden state from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Use last token's hidden state as sequence representation
        last_hidden = outputs.hidden_states[-1]

        # Find position of last non-padding token
        if attention_mask is not None:
            last_pos = attention_mask.sum(dim=1) - 1
        else:
            last_pos = torch.full((input_ids.shape[0],),
                                  input_ids.shape[1] - 1,
                                  dtype=torch.long)

        # Extract hidden state at last position
        batch_idx = torch.arange(len(input_ids), device=input_ids.device)
        hidden = last_hidden[batch_idx, last_pos]

        reward = self.reward_head(hidden)
        return reward.squeeze(-1)


def train_reward_model(reward_model, comparisons, epochs=3,
                       batch_size=16, lr=1e-5):
    """
    Train reward model on preference comparisons.

    comparisons: list of (prompt_ids, chosen_ids, rejected_ids)
    """
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for i in range(0, len(comparisons), batch_size):
            batch = comparisons[i:i+batch_size]

            chosen_rewards = []
            rejected_rewards = []

            for prompt_ids, chosen_ids, rejected_ids in batch:
                r_chosen = reward_model(chosen_ids.unsqueeze(0))
                r_rejected = reward_model(rejected_ids.unsqueeze(0))
                chosen_rewards.append(r_chosen)
                rejected_rewards.append(r_rejected)

            chosen_rewards = torch.cat(chosen_rewards)
            rejected_rewards = torch.cat(rejected_rewards)

            # Bradley-Terry loss:
            # P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
            loss = -torch.log(
                torch.sigmoid(chosen_rewards - rejected_rewards) + 1e-8
            ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (chosen_rewards > rejected_rewards).sum().item()
            total += len(batch)

        accuracy = correct / total
        avg_loss = total_loss / (len(comparisons) // batch_size)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, "
              f"Accuracy: {accuracy:.2%}")
```

### 3.3 Reward Model Quality

```
Reward model quality is CRITICAL for RLHF success.

Quality metrics:
  - Accuracy on held-out comparisons (typically 65-75%)
  - Agreement with expert annotators
  - Calibration (predicted preference strength matches reality)

Common issues:
  - Length bias: RM prefers longer responses
    Fix: Normalize by length or add length penalty

  - Sycophancy: RM rewards agreeing with the user
    Fix: Include adversarial examples in training

  - Verbosity: RM prefers more "explanation"
    Fix: Include concise expert responses as preferred

  - Position bias: RM prefers first/second response
    Fix: Randomize presentation order during annotation
```

---

## 4. PPO for Language Models

### 4.1 RL Formulation for LMs

```
MDP for text generation:
  State:   s_t = (prompt, tokens_so_far) = (x, y_{<t})
  Action:  a_t = next_token y_t
  Reward:  r_T = r_θ(x, y)  at end of sequence, 0 otherwise
  Episode: Generating one complete response

  Policy: π(y_t | x, y_{<t}) = language model

  With KL penalty:
  R_total = r_θ(x, y) - β · Σ_t log[π(y_t|x,y_{<t}) / π_ref(y_t|x,y_{<t})]

  The KL term penalizes per-token deviation from the reference model.
```

### 4.2 PPO-RLHF Implementation

```python
class PPOTrainer:
    """PPO trainer for RLHF fine-tuning of language models."""

    def __init__(self, policy_model, ref_model, reward_model,
                 lr=1e-6, gamma=1.0, gae_lambda=0.95,
                 clip_ratio=0.2, kl_coef=0.1, vf_coef=0.1,
                 max_grad_norm=1.0):
        self.policy = policy_model
        self.ref_model = ref_model  # Frozen reference (SFT model)
        self.reward_model = reward_model  # Frozen reward model
        self.kl_coef = kl_coef
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Value head (predicts expected reward)
        self.value_head = nn.Linear(
            policy_model.config.hidden_size, 1
        )

        self.optimizer = torch.optim.AdamW(
            list(self.policy.parameters()) + list(self.value_head.parameters()),
            lr=lr
        )

    @torch.no_grad()
    def generate_and_score(self, prompts, max_length=256):
        """Generate responses and compute rewards."""
        responses = []
        rewards = []
        log_probs = []
        ref_log_probs = []

        for prompt in prompts:
            # Generate response from current policy
            response, lp = self._generate(self.policy, prompt, max_length)
            responses.append(response)
            log_probs.append(lp)

            # Reference log probs
            ref_lp = self._get_log_probs(self.ref_model, prompt, response)
            ref_log_probs.append(ref_lp)

            # Reward model score
            full_text = torch.cat([prompt, response])
            r = self.reward_model(full_text.unsqueeze(0)).item()
            rewards.append(r)

        return responses, rewards, log_probs, ref_log_probs

    def compute_kl_reward(self, log_probs, ref_log_probs, reward):
        """Compute per-token reward including KL penalty."""
        # KL divergence per token: log π(y_t) - log π_ref(y_t)
        kl_per_token = log_probs - ref_log_probs

        # Per-token reward: -β * KL at each token
        token_rewards = -self.kl_coef * kl_per_token

        # Add environment reward at the last token
        token_rewards[-1] += reward

        return token_rewards

    def compute_advantages(self, token_rewards, values):
        """Compute GAE advantages."""
        T = len(token_rewards)
        advantages = torch.zeros(T)
        last_gae = 0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = token_rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * last_gae

        returns = advantages + values
        return advantages, returns

    def ppo_step(self, prompts, batch_size=4, ppo_epochs=4):
        """One PPO update step."""
        # Generate and score
        responses, rewards, old_log_probs, ref_log_probs = \
            self.generate_and_score(prompts)

        # Compute token-level rewards and advantages
        all_advantages = []
        all_returns = []
        all_old_lp = []

        for i in range(len(prompts)):
            token_rewards = self.compute_kl_reward(
                old_log_probs[i], ref_log_probs[i], rewards[i]
            )

            # Get value estimates
            with torch.no_grad():
                full_text = torch.cat([prompts[i], responses[i]])
                hidden = self.policy(full_text.unsqueeze(0),
                                     output_hidden_states=True)
                values = self.value_head(
                    hidden.hidden_states[-1].squeeze(0)
                ).squeeze(-1)
                # Only response tokens
                values = values[len(prompts[i]):]

            advantages, returns = self.compute_advantages(token_rewards, values)
            all_advantages.append(advantages)
            all_returns.append(returns)
            all_old_lp.append(old_log_probs[i])

        # PPO epochs
        for epoch in range(ppo_epochs):
            for i in range(0, len(prompts), batch_size):
                batch_idx = list(range(i, min(i + batch_size, len(prompts))))

                policy_loss = 0
                value_loss = 0

                for j in batch_idx:
                    # Current log probs
                    new_lp = self._get_log_probs(
                        self.policy, prompts[j], responses[j]
                    )

                    # PPO clipped objective
                    ratio = torch.exp(new_lp - all_old_lp[j].detach())
                    adv = all_advantages[j].detach()
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    surr1 = ratio * adv
                    surr2 = torch.clamp(
                        ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
                    ) * adv
                    policy_loss -= torch.min(surr1, surr2).mean()

                    # Value loss
                    full_text = torch.cat([prompts[j], responses[j]])
                    hidden = self.policy(full_text.unsqueeze(0),
                                         output_hidden_states=True)
                    new_values = self.value_head(
                        hidden.hidden_states[-1].squeeze(0)
                    ).squeeze(-1)[len(prompts[j]):]

                    value_loss += F.mse_loss(new_values, all_returns[j].detach())

                total_loss = policy_loss + 0.1 * value_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 1.0
                )
                self.optimizer.step()

        return {
            'mean_reward': np.mean(rewards),
            'mean_kl': np.mean([
                (lp - rlp).mean().item()
                for lp, rlp in zip(old_log_probs, ref_log_probs)
            ]),
        }

    def _generate(self, model, prompt, max_length):
        """Generate tokens and collect log probabilities."""
        # Simplified - in practice use model.generate()
        tokens = prompt.clone()
        log_probs = []

        for _ in range(max_length):
            logits = model(tokens.unsqueeze(0)).logits[0, -1]
            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1)
            lp = torch.log(probs[token])

            tokens = torch.cat([tokens, token])
            log_probs.append(lp)

            if token.item() == model.config.eos_token_id:
                break

        response = tokens[len(prompt):]
        return response, torch.cat(log_probs)

    def _get_log_probs(self, model, prompt, response):
        """Get log probabilities of response tokens."""
        full_text = torch.cat([prompt, response]).unsqueeze(0)
        with torch.no_grad():
            logits = model(full_text).logits[0]

        # Log probs for response tokens
        response_logits = logits[len(prompt)-1:-1]
        log_probs = F.log_softmax(response_logits, dim=-1)

        token_log_probs = log_probs.gather(1, response.unsqueeze(1)).squeeze(1)
        return token_log_probs
```

---

## 5. Direct Preference Optimization (DPO)

### 5.1 DPO Motivation

```
Problems with PPO-based RLHF:
1. Train separate reward model (extra compute + data)
2. PPO is unstable and sensitive to hyperparameters
3. Must keep reference model in memory
4. Slow: generate -> score -> update cycle

DPO insight: We can skip the reward model entirely!

Key derivation:
  The optimal policy under KL-constrained reward maximization is:
  π*(y|x) = π_ref(y|x) · exp(r(x,y) / β) / Z(x)

  Rearranging:
  r(x,y) = β · log(π*(y|x) / π_ref(y|x)) + β · log Z(x)

  The partition function Z(x) cancels in the preference loss!
```

### 5.2 DPO Implementation

```python
class DPOTrainer:
    """Direct Preference Optimization trainer."""

    def __init__(self, policy_model, ref_model, beta=0.1, lr=1e-6):
        self.policy = policy_model
        self.ref_model = ref_model  # Frozen
        self.beta = beta
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)

    def dpo_loss(self, chosen_ids, rejected_ids, prompt_length):
        """
        Compute DPO loss.

        L_DPO = -E[log σ(β · (log π(y_w|x)/π_ref(y_w|x)
                              - log π(y_l|x)/π_ref(y_l|x)))]
        """
        # Log probs from current policy
        chosen_logps = self._get_sequence_log_probs(
            self.policy, chosen_ids, prompt_length
        )
        rejected_logps = self._get_sequence_log_probs(
            self.policy, rejected_ids, prompt_length
        )

        # Log probs from reference model
        with torch.no_grad():
            ref_chosen_logps = self._get_sequence_log_probs(
                self.ref_model, chosen_ids, prompt_length
            )
            ref_rejected_logps = self._get_sequence_log_probs(
                self.ref_model, rejected_ids, prompt_length
            )

        # DPO loss
        chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps)

        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

        # Metrics
        with torch.no_grad():
            reward_margin = (chosen_rewards - rejected_rewards).mean().item()
            accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

        return loss, {'reward_margin': reward_margin, 'accuracy': accuracy}

    def _get_sequence_log_probs(self, model, input_ids, prompt_length):
        """Get total log probability of response tokens."""
        logits = model(input_ids).logits

        # Only compute loss on response tokens
        response_logits = logits[:, prompt_length-1:-1, :]
        response_tokens = input_ids[:, prompt_length:]

        log_probs = F.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(2, response_tokens.unsqueeze(2))
        return token_log_probs.squeeze(2).sum(dim=1)

    def train_step(self, batch):
        """One DPO training step."""
        loss, metrics = self.dpo_loss(
            batch['chosen_ids'],
            batch['rejected_ids'],
            batch['prompt_length'],
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return loss.item(), metrics
```

### 5.3 DPO vs PPO Comparison

| Aspect | PPO (RLHF) | DPO |
|--------|------------|-----|
| **Reward model** | Required (separate training) | Not needed |
| **Stability** | Sensitive to hyperparameters | More stable |
| **Memory** | Policy + RM + value model + ref | Policy + ref |
| **Implementation** | Complex (RL loop) | Simple (supervised) |
| **Performance** | Slightly better on some tasks | Competitive |
| **Iteration speed** | Slower (generate-score-update) | Faster (batch training) |
| **Online learning** | Yes (can update from new data) | Batch only |
| **Scalability** | Harder at scale | Easier at scale |

---

## 6. Advanced RLHF Topics

### 6.1 Constitutional AI (CAI)

```
Constitutional AI (Bai et al., 2022):

Replace human feedback with AI-generated feedback:

1. Red-teaming: Generate harmful prompts
2. AI critique: Ask AI to critique its own responses
   based on a "constitution" (set of principles)
3. AI revision: AI improves its response
4. RL from AI Feedback (RLAIF): Train RM on AI preferences

Constitution example:
  "Please choose the response that is most helpful while
   being safe and respectful."

Advantage: Scales without human annotators
Limitation: Quality limited by the AI critic
```

### 6.2 KTO: Kahneman-Tversky Optimization

```python
class KTOTrainer:
    """
    KTO: Learn from binary feedback (good/bad) instead of pairwise preferences.
    Based on prospect theory: losses loom larger than gains.
    """

    def __init__(self, policy, ref_model, beta=0.1, lr=1e-6):
        self.policy = policy
        self.ref_model = ref_model
        self.beta = beta
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    def kto_loss(self, input_ids, prompt_lengths, labels):
        """
        KTO loss with asymmetric treatment of desirable/undesirable outputs.

        labels: 1 for desirable, 0 for undesirable
        """
        # Compute log ratio for each example
        policy_logps = self._get_log_probs(self.policy, input_ids, prompt_lengths)
        with torch.no_grad():
            ref_logps = self._get_log_probs(self.ref_model, input_ids, prompt_lengths)

        log_ratio = policy_logps - ref_logps

        # KL estimate
        kl = (policy_logps.exp() * (policy_logps - ref_logps)).mean().detach()

        # Separate desirable and undesirable
        desirable_mask = labels == 1
        undesirable_mask = labels == 0

        # Asymmetric loss (from prospect theory)
        desirable_loss = -F.logsigmoid(
            self.beta * (log_ratio[desirable_mask] - kl)
        ).mean() if desirable_mask.any() else 0

        undesirable_loss = -F.logsigmoid(
            self.beta * (kl - log_ratio[undesirable_mask])
        ).mean() if undesirable_mask.any() else 0

        # Loss weighted by Kahneman-Tversky coefficients
        lambda_d, lambda_u = 1.0, 1.0  # Can adjust asymmetry
        loss = lambda_d * desirable_loss + lambda_u * undesirable_loss

        return loss
```

### 6.3 GRPO: Group Relative Policy Optimization

```
GRPO (DeepSeek, 2024):

Instead of PPO's per-sample advantages, compute advantages
relative to a GROUP of responses for the same prompt:

1. For each prompt x, generate G responses: y₁, ..., y_G
2. Score all: r₁, ..., r_G using reward model
3. Advantages: A_i = (r_i - mean(r)) / std(r)
4. Update policy using these group-relative advantages

Benefits:
  - No value model needed (saves memory)
  - Naturally normalized advantages
  - Simpler implementation than PPO
  - Works well for math/code reasoning tasks
```

---

## 7. Practical Considerations

### 7.1 Common RLHF Failure Modes

```
Failure Mode 1: Reward Hacking
  Symptom: Reward score goes up, but response quality goes down
  Cause: Policy exploits reward model weaknesses
  Fix: Stronger reward model, KL constraint, reward ensemble

Failure Mode 2: Mode Collapse
  Symptom: Model generates same response regardless of prompt
  Cause: KL penalty too low, reward too concentrated
  Fix: Increase β (KL coefficient), entropy bonus

Failure Mode 3: KL Explosion
  Symptom: KL divergence grows unboundedly
  Cause: Reward signal too strong relative to KL constraint
  Fix: Increase β, gradient clipping, adaptive KL target

Failure Mode 4: Verbosity Increase
  Symptom: Responses get progressively longer
  Cause: Reward model rewards length
  Fix: Length normalization, length penalty in reward
```

### 7.2 Best Practices

```
RLHF best practices:

Data:
  □ Diverse prompts covering many topics and styles
  □ High inter-annotator agreement (>70%)
  □ Balance of response lengths in comparison data
  □ Include safety-relevant comparisons

Reward Model:
  □ Train on separate data from RL training
  □ Monitor for overoptimization (RM score vs true quality)
  □ Use reward model ensemble for robustness
  □ Regular re-training as policy improves

PPO Training:
  □ Start with small learning rate (1e-6)
  □ Monitor KL divergence closely
  □ Use adaptive KL target
  □ Clip reward values to prevent outliers
  □ Evaluate on held-out test prompts regularly

DPO Training:
  □ Ensure reference model is high quality (good SFT)
  □ Batch size matters more than epochs
  □ Monitor implicit reward margin
  □ β = 0.1 is a good starting point
```

---

## 8. Exercises

### Exercise 1: Build a Simple Reward Model

Train a reward model on text preferences:
1. Create a synthetic dataset: compare long vs short summaries
2. Build a reward model using a small transformer (GPT-2)
3. Train with Bradley-Terry loss on pairwise comparisons
4. Evaluate accuracy on held-out comparisons
5. Visualize what the reward model has learned (score vs length, quality)

### Exercise 2: PPO for Text Generation

Implement PPO-based RLHF for a small language model:
1. Start with a fine-tuned GPT-2 small model
2. Use your reward model from Exercise 1
3. Implement the PPO training loop with KL penalty
4. Train for 1000 steps and monitor reward + KL divergence
5. Compare generated text before and after RLHF

### Exercise 3: DPO Implementation

Build DPO from scratch:
1. Prepare preference pairs from the same dataset as Exercise 1
2. Implement the DPO loss function
3. Train GPT-2 small with DPO for the same number of steps as PPO
4. Compare DPO vs PPO: training stability, final quality, compute cost
5. Ablate beta: {0.01, 0.1, 0.5, 1.0} and measure impact

### Exercise 4: Reward Hacking Analysis

Demonstrate and mitigate reward hacking:
1. Train a deliberately weak reward model (small, few examples)
2. Run RLHF with this weak RM and observe reward hacking
3. Log both RM score and true quality metric over training
4. Show the "overoptimization" curve (RM score up, quality down)
5. Mitigate with: (a) KL increase, (b) RM ensemble, (c) early stopping

### Exercise 5: Preference Data Collection Simulator

Build a preference annotation pipeline:
1. Create a prompt dataset (100 diverse questions)
2. Generate multiple responses using temperature sampling
3. Implement a simulated annotator (rule-based: prefer accurate, concise)
4. Study inter-annotator agreement as a function of noise level
5. Show how reward model quality degrades with noisy annotations

---

*End of Lesson 24*

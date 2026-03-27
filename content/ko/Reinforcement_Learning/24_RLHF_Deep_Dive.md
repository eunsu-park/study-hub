[이전: RL for Robotics](./23_RL_for_Robotics.md)

---

# 24. RLHF 심층 분석

## 학습 목표

이 강의를 완료하면 다음을 할 수 있습니다:

1. 언어 모델을 인간 선호도에 정렬하기 위한 전체 RLHF 파이프라인 설명
2. KL 제약이 있는 언어 모델의 PPO 기반 미세 조정 구현
3. 쌍별 인간 비교로부터 보상 모델 구축
4. PPO의 더 간단한 대안으로서 Direct Preference Optimization (DPO) 이해
5. RLHF 학습의 실패 모드와 모범 사례 분석

---

## 목차

1. [왜 RLHF인가?](#1-왜-rlhf인가)
2. [RLHF 파이프라인](#2-rlhf-파이프라인)
3. [보상 모델링](#3-보상-모델링)
4. [언어 모델을 위한 PPO](#4-언어-모델을-위한-ppo)
5. [Direct Preference Optimization (DPO)](#5-direct-preference-optimization-dpo)
6. [고급 RLHF 주제](#6-고급-rlhf-주제)
7. [실전 고려사항](#7-실전-고려사항)
8. [연습문제](#8-연습문제)

---

## 1. 왜 RLHF인가?

### 1.1 정렬 문제

```
사전학습 목적:
  다음 토큰 예측: P(x_t | x_1, ..., x_{t-1})
  이것은 다음을 포함한 모든 텍스트를 모방하도록 학습:
  - 도움이 되는 지시사항
  - 유해한 콘텐츠
  - 허위 정보
  - 편향된 관점

실제로 원하는 것:
  - 도움: 질문에 정확하게 답변
  - 무해: 위험한 요청을 거부
  - 정직: 불확실성을 인정

간극: "텍스트 예측을 잘하는 것" ≠ "좋은 어시스턴트인 것"

RLHF는 인간 선호도에 최적화하여 이 간극을 연결합니다.
```

### 1.2 RLHF 연혁

```
2017: Christiano et al. - 인간 선호도로부터의 RL (Atari, MuJoCo)
2019: Ziegler et al.    - 인간 선호도로 LM 미세 조정
2020: Stiennon et al.   - 인간 피드백으로 요약 학습
2022: Ouyang et al.     - InstructGPT (GPT-3 + RLHF)
2022: Bai et al.        - Constitutional AI (Anthropic)
2023: Rafailov et al.   - DPO (직접 선호도 최적화)
2024: Various           - GRPO, KTO, IPO 및 기타 대안
```

---

## 2. RLHF 파이프라인

### 2.1 3단계 학습

```
1단계: 지도 미세 조정 (SFT)
  입력:  사전학습된 LLM + 인간 작성 시연
  출력:  π_SFT (지시를 따르는 모델)
  방법:  선별된 데이터에 대한 표준 언어 모델링 손실

2단계: 보상 모델 학습
  입력:  인간이 순위를 매긴 출력 쌍
  출력:  r_θ(프롬프트, 응답) -> 스칼라 보상
  방법:  Bradley-Terry 선호도 모델

3단계: RL 미세 조정
  입력:  π_SFT + r_θ
  출력:  π_RLHF (정렬된 모델)
  방법:  π_SFT에 대한 KL 페널티가 있는 PPO

  목적:
  max_π E_{x~D, y~π(·|x)} [r_θ(x, y)] - β · KL(π || π_SFT)
       ↑ 보상 최대화              ↑ SFT에서 너무 멀어지지 않기
```

### 2.2 파이프라인 시각화

```
                    인간 시연
                       │
                ┌──────▼──────┐
 사전학습 LLM ──▶  SFT 모델   │
                └──────┬──────┘
                       │
                ┌──────▼──────┐     인간 선호도
                │  응답 생성   │     (쌍별 순위)
                │             │──────────┐
                └──────┬──────┘           │
                       │          ┌───────▼────────┐
                       │          │  보상 모델      │
                       │          │  r_θ(x, y)     │
                       │          └───────┬────────┘
                ┌──────▼──────────────────▼──────┐
                │           PPO 학습             │
                │  max r_θ(x,y) - β·KL(π||π_SFT)│
                └──────────────┬─────────────────┘
                               │
                        ┌──────▼──────┐
                        │ RLHF 모델  │
                        └─────────────┘
```

---

## 3. 보상 모델링

### 3.1 데이터 수집

```
보상 모델 학습 데이터:

1. 데이터셋에서 프롬프트 x 샘플링
2. 프롬프트당 K개 응답 생성: y₁, y₂, ..., y_K ~ π_SFT(·|x)
3. 인간 주석자가 응답에 순위 매김: y_i > y_j (y_i가 선호됨)
4. 비교 쌍 생성: (x, y_w, y_l) 여기서 y_w가 승리

예시:
  프롬프트: "양자 컴퓨팅을 간단한 용어로 설명해주세요"

  응답 A: "양자 컴퓨팅은 동시에 0과 1이 될 수 있는 양자 비트를
  사용하여 더 빠른 계산을 가능하게 합니다."

  응답 B: "양자 컴퓨팅은 양자역학 현상을 활용하는
  혁명적인 패러다임입니다..."

  인간: A > B (요청대로 A가 더 간단함)
```

### 3.2 보상 모델 구현

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

### 3.3 보상 모델 품질

```
보상 모델 품질은 RLHF 성공에 매우 중요합니다.

품질 지표:
  - 보류된 비교에서의 정확도 (일반적으로 65-75%)
  - 전문 주석자와의 일치도
  - 보정 (예측된 선호 강도가 현실과 일치)

일반적인 문제:
  - 길이 편향: RM이 더 긴 응답을 선호
    수정: 길이로 정규화 또는 길이 페널티 추가

  - 아첨: RM이 사용자에게 동의하는 것을 보상
    수정: 적대적 예제를 학습에 포함

  - 장황함: RM이 더 많은 "설명"을 선호
    수정: 간결한 전문가 응답을 선호로 포함

  - 위치 편향: RM이 첫 번째/두 번째 응답을 선호
    수정: 주석 중 제시 순서를 무작위화
```

---

## 4. 언어 모델을 위한 PPO

### 4.1 LM을 위한 RL 정식화

```
텍스트 생성을 위한 MDP:
  상태:   s_t = (프롬프트, 지금까지의 토큰) = (x, y_{<t})
  행동:   a_t = 다음 토큰 y_t
  보상:   r_T = r_θ(x, y)  시퀀스 끝에서, 그 외 0
  에피소드: 하나의 완전한 응답 생성

  정책: π(y_t | x, y_{<t}) = 언어 모델

  KL 페널티 포함:
  R_total = r_θ(x, y) - β · Σ_t log[π(y_t|x,y_{<t}) / π_ref(y_t|x,y_{<t})]

  KL 항은 참조 모델로부터의 토큰별 이탈에 패널티를 줍니다.
```

### 4.2 PPO-RLHF 구현

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

### 5.1 DPO 동기

```
PPO 기반 RLHF의 문제:
1. 별도의 보상 모델 학습 (추가 계산 + 데이터)
2. PPO가 불안정하고 하이퍼파라미터에 민감
3. 참조 모델을 메모리에 유지해야 함
4. 느림: 생성 -> 점수 -> 업데이트 사이클

DPO 통찰: 보상 모델을 완전히 건너뛸 수 있습니다!

핵심 유도:
  KL 제약 보상 최대화에서의 최적 정책:
  π*(y|x) = π_ref(y|x) · exp(r(x,y) / β) / Z(x)

  재배열:
  r(x,y) = β · log(π*(y|x) / π_ref(y|x)) + β · log Z(x)

  분배 함수 Z(x)는 선호도 손실에서 소거됩니다!
```

### 5.2 DPO 구현

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

### 5.3 DPO vs PPO 비교

| 측면 | PPO (RLHF) | DPO |
|--------|------------|-----|
| **보상 모델** | 필요 (별도 학습) | 불필요 |
| **안정성** | 하이퍼파라미터에 민감 | 더 안정적 |
| **메모리** | 정책 + RM + 가치 모델 + 참조 | 정책 + 참조 |
| **구현** | 복잡 (RL 루프) | 간단 (지도 학습) |
| **성능** | 일부 과제에서 약간 더 나음 | 경쟁력 있음 |
| **반복 속도** | 느림 (생성-점수-업데이트) | 빠름 (배치 학습) |
| **온라인 학습** | 예 (새 데이터에서 업데이트 가능) | 배치만 |
| **확장성** | 대규모에서 더 어려움 | 대규모에서 더 쉬움 |

---

## 6. 고급 RLHF 주제

### 6.1 Constitutional AI (CAI)

```
Constitutional AI (Bai et al., 2022):

인간 피드백을 AI 생성 피드백으로 대체:

1. 레드 팀: 유해한 프롬프트 생성
2. AI 비평: AI에게 "헌법" (원칙 세트)에 기반하여 자신의 응답을 비평하도록 요청
3. AI 수정: AI가 응답을 개선
4. AI 피드백 RL (RLAIF): AI 선호도로 RM 학습

헌법 예시:
  "안전하고 존중하면서 가장 도움이 되는 응답을 선택하세요."

장점: 인간 주석자 없이 확장 가능
한계: AI 비평가의 품질에 제한됨
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

PPO의 개별 샘플 어드밴티지 대신, 같은 프롬프트에 대한
응답 그룹에 상대적으로 어드밴티지를 계산:

1. 각 프롬프트 x에 대해 G개 응답 생성: y₁, ..., y_G
2. 모두 점수화: r₁, ..., r_G (보상 모델 사용)
3. 어드밴티지: A_i = (r_i - mean(r)) / std(r)
4. 이 그룹 상대적 어드밴티지로 정책 업데이트

장점:
  - 가치 모델 불필요 (메모리 절약)
  - 자연적으로 정규화된 어드밴티지
  - PPO보다 간단한 구현
  - 수학/코드 추론 과제에서 잘 작동
```

---

## 7. 실전 고려사항

### 7.1 일반적인 RLHF 실패 모드

```
실패 모드 1: 보상 해킹
  증상: 보상 점수는 올라가지만 응답 품질은 떨어짐
  원인: 정책이 보상 모델의 약점을 악용
  수정: 더 강한 보상 모델, KL 제약, 보상 앙상블

실패 모드 2: 모드 붕괴
  증상: 프롬프트와 관계없이 같은 응답 생성
  원인: KL 페널티 너무 낮음, 보상이 너무 집중됨
  수정: β (KL 계수) 증가, 엔트로피 보너스

실패 모드 3: KL 폭발
  증상: KL 발산이 제한 없이 증가
  원인: KL 제약 대비 보상 신호가 너무 강함
  수정: β 증가, 그래디언트 클리핑, 적응형 KL 목표

실패 모드 4: 장황함 증가
  증상: 응답이 점점 더 길어짐
  원인: 보상 모델이 길이를 보상
  수정: 길이 정규화, 보상에 길이 페널티
```

### 7.2 모범 사례

```
RLHF 모범 사례:

데이터:
  □ 다양한 주제와 스타일을 다루는 다양한 프롬프트
  □ 높은 주석자 간 일치도 (>70%)
  □ 비교 데이터에서 응답 길이의 균형
  □ 안전 관련 비교 포함

보상 모델:
  □ RL 학습과 별도의 데이터로 학습
  □ 과적최적화 모니터링 (RM 점수 vs 실제 품질)
  □ 견고성을 위한 보상 모델 앙상블 사용
  □ 정책 향상에 따른 정기적 재학습

PPO 학습:
  □ 작은 학습률로 시작 (1e-6)
  □ KL 발산을 면밀히 모니터링
  □ 적응형 KL 목표 사용
  □ 이상치를 방지하기 위해 보상 값 클리핑
  □ 보류된 테스트 프롬프트에서 정기적 평가

DPO 학습:
  □ 참조 모델이 고품질인지 확인 (좋은 SFT)
  □ 배치 크기가 에폭보다 더 중요
  □ 암시적 보상 마진 모니터링
  □ β = 0.1이 좋은 시작점
```

---

## 8. 연습문제

### 연습문제 1: 간단한 보상 모델 구축

텍스트 선호도에 대한 보상 모델을 학습하세요:
1. 합성 데이터셋 생성: 긴 요약과 짧은 요약 비교
2. 작은 트랜스포머 (GPT-2)를 사용하여 보상 모델 구축
3. 쌍별 비교에서 Bradley-Terry 손실로 학습
4. 보류된 비교에서 정확도 평가
5. 보상 모델이 학습한 것을 시각화 (점수 vs 길이, 품질)

### 연습문제 2: 텍스트 생성을 위한 PPO

작은 언어 모델을 위한 PPO 기반 RLHF를 구현하세요:
1. 미세 조정된 GPT-2 small 모델로 시작
2. 연습문제 1의 보상 모델 사용
3. KL 페널티가 있는 PPO 학습 루프 구현
4. 1000 스텝 학습하고 보상 + KL 발산 모니터링
5. RLHF 전후의 생성 텍스트 비교

### 연습문제 3: DPO 구현

DPO를 처음부터 구축하세요:
1. 연습문제 1과 같은 데이터셋에서 선호도 쌍 준비
2. DPO 손실 함수 구현
3. PPO와 같은 수의 스텝으로 GPT-2 small을 DPO로 학습
4. DPO vs PPO 비교: 학습 안정성, 최종 품질, 계산 비용
5. beta 값 조절: {0.01, 0.1, 0.5, 1.0}으로 영향 측정

### 연습문제 4: 보상 해킹 분석

보상 해킹을 시연하고 완화하세요:
1. 의도적으로 약한 보상 모델 학습 (작은 모델, 적은 예제)
2. 이 약한 RM으로 RLHF를 실행하고 보상 해킹 관찰
3. RM 점수와 실제 품질 지표를 학습 과정에서 로깅
4. "과적최적화" 곡선 시연 (RM 점수 상승, 품질 하락)
5. 완화 방법: (a) KL 증가, (b) RM 앙상블, (c) 조기 중단

### 연습문제 5: 선호도 데이터 수집 시뮬레이터

선호도 주석 파이프라인을 구축하세요:
1. 프롬프트 데이터셋 생성 (100개의 다양한 질문)
2. 온도 샘플링으로 여러 응답 생성
3. 시뮬레이션된 주석자 구현 (규칙 기반: 정확하고 간결한 것을 선호)
4. 잡음 수준의 함수로서 주석자 간 일치도 연구
5. 잡음이 있는 주석으로 보상 모델 품질이 어떻게 저하되는지 시연

---

*24강 끝*

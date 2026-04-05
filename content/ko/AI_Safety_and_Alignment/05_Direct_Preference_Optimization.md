# 05. 직접 선호도 최적화 (Direct Preference Optimization)

**이전**: [헌법적 AI](./04_Constitutional_AI.md) | **다음**: [확장 가능한 감독](./06_Scalable_Oversight.md)

## 학습 목표
- RLHF 프레임워크에서 DPO 목적 함수를 유도하고 이론적 기반을 설명한다
- PyTorch로 DPO 학습을 구현하고 PPO 기반 RLHF와 비교한다
- KTO, IPO, ORPO, SimPO를 대안적 선호도 최적화 방법으로 이해하고 구현한다
- 온라인(online)과 오프라인(offline) 선호도 최적화 간의 트레이드오프를 평가한다
- 주어진 정렬(alignment) 시나리오에 적합한 선호도 최적화 방법을 선택한다

---

직접 선호도 최적화(Direct Preference Optimization, DPO)는 정렬(alignment) 방법론에
대한 획기적인 기여이다. RLHF와 동일한 목표 -- 인간의 선호도에 맞게 언어 모델을
최적화하는 것 -- 를 달성하지만, 별도의 보상 모델(reward model) 학습과 강화 학습
실행의 필요성을 제거한다. RLHF 목적 함수를 재매개변수화(reparameterize)함으로써,
DPO는 정렬 문제를 선호도 쌍(preference pairs)에 대한 간단한 분류 유사 손실(loss)로
변환한다. 이 수업에서는 DPO와 그 후속 방법들을 이론적 기반과 실용적 구현 가이드와
함께 다룬다.

## 목차
1. [DPO 유도와 직관](#1-dpo-유도와-직관)
2. [DPO vs RLHF 비교](#2-dpo-vs-rlhf-비교)
3. [KTO: Kahneman-Tversky 최적화](#3-kto-kahneman-tversky-최적화)
4. [IPO: 항등 선호도 최적화](#4-ipo-항등-선호도-최적화)
5. [ORPO: 승산비 선호도 최적화](#5-orpo-승산비-선호도-최적화)
6. [SimPO: 단순 선호도 최적화](#6-simpo-단순-선호도-최적화)
7. [방법 간 선택](#7-방법-간-선택)
8. [Transformers와 TRL을 활용한 구현](#8-transformers와-trl을-활용한-구현)
9. [오프라인 vs 온라인 선호도 최적화](#9-오프라인-vs-온라인-선호도-최적화)
10. [실용적 고려사항](#10-실용적-고려사항)
11. [연습문제](#연습문제)

---

## 1. DPO 유도와 직관

### 1.1 RLHF에서 DPO로

DPO의 핵심 통찰은 RLHF 목적 함수(보상을 최대화하면서 참조 정책에 가깝게 유지)
하에서 최적 정책(optimal policy)이 닫힌 형태(closed-form)의 해를 갖는다는 것이다.
이를 통해 강화 학습 단계를 완전히 우회할 수 있다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import Optional
import random


@dataclass
class DPODerivation:
    """Step-by-step derivation of DPO from RLHF."""
    step: int
    name: str
    formula: str
    explanation: str


derivation_steps = [
    DPODerivation(
        step=1,
        name="RLHF Objective",
        formula="max_pi E[r(x,y)] - beta * KL(pi || pi_ref)",
        explanation="RLHF maximizes reward r(x,y) from the reward model "
                    "while penalizing divergence from the reference policy "
                    "pi_ref with coefficient beta.",
    ),
    DPODerivation(
        step=2,
        name="Optimal Policy (Closed Form)",
        formula="pi*(y|x) = pi_ref(y|x) * exp(r(x,y) / beta) / Z(x)",
        explanation="The KL-constrained optimization has a closed-form "
                    "solution. The optimal policy is the reference policy "
                    "reweighted by the exponentiated reward, normalized "
                    "by partition function Z(x).",
    ),
    DPODerivation(
        step=3,
        name="Reparameterize Reward",
        formula="r(x,y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log(Z(x))",
        explanation="Rearranging the optimal policy equation, we can "
                    "express the reward in terms of the policy and "
                    "reference policy log-ratios. The partition function "
                    "Z(x) only depends on x, not y.",
    ),
    DPODerivation(
        step=4,
        name="Bradley-Terry Preference Model",
        formula="P(y_w > y_l | x) = sigma(r(x,y_w) - r(x,y_l))",
        explanation="Human preferences follow the Bradley-Terry model: "
                    "the probability that y_w is preferred over y_l is "
                    "a sigmoid of their reward difference.",
    ),
    DPODerivation(
        step=5,
        name="DPO Loss (Final)",
        formula="L_DPO = -E[log sigma(beta * (log(pi(y_w|x)/pi_ref(y_w|x)) "
                "- log(pi(y_l|x)/pi_ref(y_l|x))))]",
        explanation="Substituting the reparameterized reward into the "
                    "Bradley-Terry model, Z(x) cancels out! We get a "
                    "loss that depends only on the policy and reference "
                    "policy, with no reward model needed.",
    ),
]

print("=== DPO Derivation ===\n")
for step in derivation_steps:
    print(f"Step {step.step}: {step.name}")
    print(f"  Formula: {step.formula}")
    print(f"  Explanation: {step.explanation[:70]}...")
    print()
```

### 1.2 DPO 손실 함수 구현

```python
def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the DPO loss.

    Args:
        policy_chosen_logps: Log probs of chosen responses under policy
        policy_rejected_logps: Log probs of rejected responses under policy
        reference_chosen_logps: Log probs of chosen under reference
        reference_rejected_logps: Log probs of rejected under reference
        beta: Temperature parameter (controls deviation from reference)

    Returns:
        (loss, chosen_rewards, rejected_rewards)
    """
    # Log ratios (implicit rewards)
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps

    # DPO loss: -log(sigmoid(beta * (chosen_logratio - rejected_logratio)))
    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits).mean()

    # Implicit rewards (for monitoring)
    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()

    return loss, chosen_rewards, rejected_rewards


# Demonstrate DPO loss computation
torch.manual_seed(42)
batch_size = 8

# Simulate log probabilities
policy_chosen = torch.randn(batch_size) * 0.5 - 2.0
policy_rejected = torch.randn(batch_size) * 0.5 - 3.0
ref_chosen = torch.randn(batch_size) * 0.5 - 2.5
ref_rejected = torch.randn(batch_size) * 0.5 - 2.5

loss, chosen_rewards, rejected_rewards = dpo_loss(
    policy_chosen, policy_rejected,
    ref_chosen, ref_rejected,
    beta=0.1,
)

print("=== DPO Loss Computation ===\n")
print(f"Loss: {loss.item():.4f}")
print(f"Mean chosen reward:   {chosen_rewards.mean().item():.4f}")
print(f"Mean rejected reward: {rejected_rewards.mean().item():.4f}")
print(f"Reward margin:        {(chosen_rewards - rejected_rewards).mean().item():.4f}")
print(f"Accuracy:             {(chosen_rewards > rejected_rewards).float().mean():.3f}")
```

### 1.3 직관적 이해

```python
class DPOIntuition:
    """Build intuition for what DPO does."""

    @staticmethod
    def explain_beta_effect() -> None:
        """Show how beta affects DPO training."""
        print("=== Effect of Beta on DPO ===\n")
        print("Beta controls how much the policy can deviate from reference.\n")

        betas = [0.01, 0.1, 0.5, 1.0, 5.0]
        torch.manual_seed(42)

        # Same preference data
        policy_chosen = torch.tensor([-2.0])
        policy_rejected = torch.tensor([-3.0])
        ref_chosen = torch.tensor([-2.5])
        ref_rejected = torch.tensor([-2.5])

        print(f"{'Beta':<10} {'Loss':<12} {'Chosen R':<12} "
              f"{'Rejected R':<12} {'Gradient':<12}")
        print("-" * 58)

        for beta in betas:
            pc = policy_chosen.clone().requires_grad_(True)
            pr = policy_rejected.clone().requires_grad_(True)
            loss, cr, rr = dpo_loss(pc, pr, ref_chosen, ref_rejected, beta)
            loss.backward()

            print(
                f"{beta:<10.2f} "
                f"{loss.item():<12.4f} "
                f"{cr.item():<12.4f} "
                f"{rr.item():<12.4f} "
                f"{pc.grad.item():<12.6f}"
            )

        print("\nSmall beta: Strong regularization toward reference.")
        print("Large beta: Policy free to deviate more from reference.")

    @staticmethod
    def explain_implicit_reward() -> None:
        """Explain DPO's implicit reward model."""
        print("\n=== DPO's Implicit Reward Model ===\n")
        print("DPO does not train an explicit reward model, but it")
        print("implicitly defines one:\n")
        print("  r(x,y) = beta * log(pi(y|x) / pi_ref(y|x)) + const\n")
        print("The reward is the log-ratio between the trained policy")
        print("and the reference policy, scaled by beta.\n")
        print("This means:")
        print("  - High reward = policy upweights this response vs reference")
        print("  - Low reward = policy downweights this response vs reference")
        print("  - The reward is always relative to the reference model")


DPOIntuition.explain_beta_effect()
DPOIntuition.explain_implicit_reward()
```

---

## 2. DPO vs RLHF 비교

### 2.1 항목별 비교

```python
@dataclass
class MethodProperty:
    """A property comparison between DPO and RLHF."""
    property_name: str
    rlhf: str
    dpo: str
    winner: str  # rlhf, dpo, tie


properties = [
    MethodProperty(
        "Training pipeline",
        "3 stages: SFT -> RM training -> PPO",
        "2 stages: SFT -> DPO loss on preferences",
        "dpo",
    ),
    MethodProperty(
        "Models required",
        "4: policy, reference, reward model, value model",
        "2: policy, reference",
        "dpo",
    ),
    MethodProperty(
        "GPU memory",
        "High (4 model copies during PPO)",
        "Lower (2 model copies)",
        "dpo",
    ),
    MethodProperty(
        "Hyperparameter sensitivity",
        "Many: PPO lr, KL coeff, clip range, epochs, etc.",
        "Few: beta, learning rate",
        "dpo",
    ),
    MethodProperty(
        "Training stability",
        "Unstable: RL training can diverge",
        "Stable: simple cross-entropy-like loss",
        "dpo",
    ),
    MethodProperty(
        "Compute cost",
        "High: RL rollouts + reward inference",
        "Lower: single forward/backward pass",
        "dpo",
    ),
    MethodProperty(
        "Online learning",
        "Natural: generates new samples each step",
        "Offline: uses fixed preference dataset",
        "rlhf",
    ),
    MethodProperty(
        "Reward model reuse",
        "RM can be used for other tasks (eval, BoN)",
        "No explicit RM (implicit only)",
        "rlhf",
    ),
    MethodProperty(
        "Empirical quality (large scale)",
        "Proven at scale (GPT-4, Claude, Gemini)",
        "Strong results, but less proven at frontier scale",
        "rlhf",
    ),
    MethodProperty(
        "Implementation complexity",
        "Complex: full RL infrastructure needed",
        "Simple: standard supervised training",
        "dpo",
    ),
]

print("=== DPO vs RLHF Comparison ===\n")
print(f"{'Property':<28} {'RLHF':<30} {'DPO':<30} {'Winner':<8}")
print("-" * 96)
rlhf_wins = 0
dpo_wins = 0
for prop in properties:
    print(f"{prop.property_name:<28} {prop.rlhf[:28]:<30} {prop.dpo[:28]:<30} "
          f"{prop.winner:<8}")
    if prop.winner == "rlhf":
        rlhf_wins += 1
    elif prop.winner == "dpo":
        dpo_wins += 1

print(f"\nScore: DPO {dpo_wins} - RLHF {rlhf_wins}")
```

### 2.2 각 방법의 선택 기준

```python
@dataclass
class MethodRecommendation:
    """Recommendation for when to use each method."""
    scenario: str
    recommended: str
    reason: str


recommendations = [
    MethodRecommendation(
        "Small team, limited compute",
        "DPO",
        "Much simpler to implement and train. No RL infrastructure needed.",
    ),
    MethodRecommendation(
        "Frontier model training at scale",
        "RLHF (PPO)",
        "Proven at the largest scale. Online learning may produce "
        "better results. Major labs use PPO for their best models.",
    ),
    MethodRecommendation(
        "Need to reuse reward model",
        "RLHF",
        "Explicit reward model can be used for Best-of-N sampling, "
        "evaluation, and monitoring.",
    ),
    MethodRecommendation(
        "Rapid iteration on alignment",
        "DPO",
        "Faster training loops. Easier to experiment with different "
        "preference datasets and beta values.",
    ),
    MethodRecommendation(
        "Offline preference data only",
        "DPO",
        "DPO is naturally offline. RLHF benefits from online data "
        "generation.",
    ),
    MethodRecommendation(
        "Continuous improvement from user feedback",
        "RLHF (online)",
        "PPO can continuously generate and learn from new data. "
        "Online DPO exists but is less standard.",
    ),
]

print("\n=== When to Use Each Method ===\n")
for rec in recommendations:
    print(f"Scenario: {rec.scenario}")
    print(f"  -> Use {rec.recommended}: {rec.reason[:65]}...")
    print()
```

---

## 3. KTO: Kahneman-Tversky 최적화

### 3.1 KTO 개요

KTO(Kahneman-Tversky Optimization)는 DPO의 주요 실용적 한계를 해결한다: 쌍으로
된 선호도(paired preferences)를 필요로 하지 않는다. 대신 개별 출력에 대한 이진
피드백(binary feedback, 좋음/나쁨)을 사용한다.

```python
@dataclass
class KTOProperties:
    """Properties of the KTO method."""
    name: str = "KTO (Kahneman-Tversky Optimization)"
    paper: str = "KTO: Model Alignment as Prospect Theoretic Optimization (Ethayarajh et al., 2024)"
    key_insight: str = (
        "Uses prospect theory (Kahneman & Tversky) to define a loss function "
        "that only requires pointwise feedback (good/bad) rather than "
        "pairwise comparisons."
    )
    data_requirement: str = "Binary labels per output: desirable or undesirable"
    advantages: list[str] = field(default_factory=lambda: [
        "No paired data needed (much easier to collect)",
        "Works with thumbs-up/thumbs-down feedback",
        "Theoretically grounded in prospect theory",
        "Handles asymmetric loss (losses hurt more than gains help)",
    ])
    disadvantages: list[str] = field(default_factory=lambda: [
        "Less signal per example than paired comparisons",
        "Requires good balance of positive/negative examples",
        "Less studied than DPO empirically",
    ])


def kto_loss(
    policy_logps: torch.Tensor,
    reference_logps: torch.Tensor,
    is_desirable: torch.Tensor,    # Boolean: True for good, False for bad
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> torch.Tensor:
    """
    Compute the KTO loss.

    KTO uses prospect theory: the value function is concave for gains
    (risk-averse) and convex for losses (risk-seeking), with losses
    weighted more heavily than gains.

    For desirable outputs: L = 1 - sigma(beta * (logr - KL))
    For undesirable outputs: L = 1 - sigma(beta * (KL - logr))

    where logr = log(pi/pi_ref) and KL is the expected KL divergence.
    """
    log_ratios = policy_logps - reference_logps
    kl_estimate = log_ratios.mean().detach()  # Approximate expected KL

    desirable_mask = is_desirable.float()
    undesirable_mask = 1.0 - desirable_mask

    # Desirable: encourage increasing log-ratio above KL baseline
    desirable_loss = 1 - torch.sigmoid(beta * (log_ratios - kl_estimate))
    # Undesirable: encourage decreasing log-ratio below KL baseline
    undesirable_loss = 1 - torch.sigmoid(beta * (kl_estimate - log_ratios))

    loss = (
        desirable_weight * (desirable_mask * desirable_loss).sum()
        + undesirable_weight * (undesirable_mask * undesirable_loss).sum()
    ) / max(1.0, desirable_mask.sum() + undesirable_mask.sum())

    return loss


# Demonstrate KTO
torch.manual_seed(42)
batch_size = 16

policy_logps = torch.randn(batch_size) * 0.5 - 2.0
ref_logps = torch.randn(batch_size) * 0.5 - 2.5
is_desirable = torch.tensor([True] * 8 + [False] * 8)

loss = kto_loss(policy_logps, ref_logps, is_desirable, beta=0.1)
print("=== KTO Loss ===\n")
print(f"KTO Loss: {loss.item():.4f}")
print(f"Desirable examples: {is_desirable.sum().item()}")
print(f"Undesirable examples: {(~is_desirable).sum().item()}")

kto_props = KTOProperties()
print(f"\nKey insight: {kto_props.key_insight[:70]}...")
print(f"Data requirement: {kto_props.data_requirement}")
```

---

## 4. IPO: 항등 선호도 최적화

### 4.1 IPO 개요

IPO(Identity Preference Optimization)는 DPO의 이론적 우려를 해결한다: DPO 손실은
강하게 선호되는 예제에 대해 우도비(likelihood ratio)를 무한대로 몰아가려 하기 때문에
선호도 데이터에 과적합(overfit)될 수 있다.

```python
def ipo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    tau: float = 0.1,
) -> torch.Tensor:
    """
    Compute the IPO (Identity Preference Optimization) loss.

    IPO uses a squared loss instead of a log-sigmoid loss,
    which prevents the policy from assigning extreme probability
    ratios to preferred responses.

    L_IPO = ((log(pi(y_w)/pi_ref(y_w)) - log(pi(y_l)/pi_ref(y_l))) - 1/(2*tau))^2

    The key difference from DPO:
    - DPO: drives logit difference to +infinity
    - IPO: drives logit difference to a finite target (1/2*tau)
    """
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps

    logit_diff = chosen_logratios - rejected_logratios
    target = 1.0 / (2.0 * tau)

    loss = (logit_diff - target).pow(2).mean()
    return loss


# Demonstrate IPO
torch.manual_seed(42)
batch_size = 8

policy_chosen = torch.randn(batch_size) * 0.5 - 2.0
policy_rejected = torch.randn(batch_size) * 0.5 - 3.0
ref_chosen = torch.randn(batch_size) * 0.5 - 2.5
ref_rejected = torch.randn(batch_size) * 0.5 - 2.5

ipo_l = ipo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, tau=0.1)
dpo_l, _, _ = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)

print("=== IPO vs DPO Loss ===\n")
print(f"IPO Loss: {ipo_l.item():.4f}")
print(f"DPO Loss: {dpo_l.item():.4f}")

print("\nKey difference:")
print("  DPO: Log-sigmoid loss -> drives margin to infinity")
print("  IPO: Squared loss -> drives margin to finite target (1/2*tau)")
print("  IPO prevents overfitting to noisy preference labels")
```

### 4.2 IPO 특성

```python
@dataclass
class IPOAnalysis:
    """Analysis of IPO properties."""
    property_name: str
    dpo_behavior: str
    ipo_behavior: str


ipo_analysis = [
    IPOAnalysis(
        "Loss at boundary",
        "DPO loss approaches 0 as margin -> infinity (saturates)",
        "IPO loss is 0 only at the exact target margin (finite)",
    ),
    IPOAnalysis(
        "Handling noisy labels",
        "DPO may overfit to wrong labels by driving margin high",
        "IPO's finite target provides natural regularization",
    ),
    IPOAnalysis(
        "Gradient behavior",
        "DPO gradient vanishes for well-separated pairs",
        "IPO gradient is proportional to distance from target",
    ),
    IPOAnalysis(
        "Hyperparameter",
        "beta controls regularization strength",
        "tau controls the target margin (1/2*tau)",
    ),
]

print("\n=== IPO Analysis ===\n")
print(f"{'Property':<25} {'DPO':<35} {'IPO':<35}")
print("-" * 95)
for a in ipo_analysis:
    print(f"{a.property_name:<25} {a.dpo_behavior[:33]:<35} {a.ipo_behavior[:33]:<35}")
```

---

## 5. ORPO: 승산비 선호도 최적화

### 5.1 ORPO 개요

ORPO(Odds Ratio Preference Optimization)는 다른 접근 방식을 취한다: 선호도
목적 함수를 SFT 손실에 직접 통합하여 참조 모델(reference model)의 필요성을
완전히 제거한다.

```python
def orpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    chosen_nll_loss: torch.Tensor,
    lambda_orpo: float = 1.0,
) -> torch.Tensor:
    """
    Compute the ORPO loss.

    ORPO combines SFT loss with an odds ratio penalty:
    L_ORPO = L_SFT + lambda * L_OR

    where L_OR = -log(sigmoid(log(odds_chosen / odds_rejected)))

    Key insight: odds ratio naturally penalizes generating rejected
    responses without needing a reference model.

    Args:
        policy_chosen_logps: Mean log-probs of chosen under policy
        policy_rejected_logps: Mean log-probs of rejected under policy
        chosen_nll_loss: NLL loss on chosen sequence (SFT component)
        lambda_orpo: Weight for the odds ratio term
    """
    # Convert log-probs to odds: odds = p / (1-p) = exp(logp) / (1 - exp(logp))
    # In log space: log_odds = logp - log(1 - exp(logp))
    chosen_log_odds = policy_chosen_logps - torch.log1p(-torch.exp(policy_chosen_logps))
    rejected_log_odds = policy_rejected_logps - torch.log1p(-torch.exp(policy_rejected_logps))

    # Log odds ratio
    log_odds_ratio = chosen_log_odds - rejected_log_odds

    # ORPO preference loss
    odds_ratio_loss = -F.logsigmoid(log_odds_ratio).mean()

    # Combined loss
    total_loss = chosen_nll_loss + lambda_orpo * odds_ratio_loss

    return total_loss


# Demonstrate ORPO
torch.manual_seed(42)
batch_size = 8

policy_chosen = torch.log(torch.rand(batch_size) * 0.3 + 0.5)   # Higher probs
policy_rejected = torch.log(torch.rand(batch_size) * 0.3 + 0.1)  # Lower probs
sft_loss = torch.tensor(2.5)  # Simulated NLL loss

orpo_l = orpo_loss(policy_chosen, policy_rejected, sft_loss, lambda_orpo=1.0)

print("=== ORPO Loss ===\n")
print(f"ORPO Total Loss: {orpo_l.item():.4f}")
print(f"SFT Component: {sft_loss.item():.4f}")
print(f"\nKey advantage: No reference model needed!")
print("ORPO merges SFT and alignment into a single training stage.")

orpo_advantages = [
    "No reference model: saves memory and compute",
    "Single training stage: SFT + alignment combined",
    "Simpler pipeline: fewer moving parts",
    "Competitive with DPO on benchmarks",
]

print("\nORPO Advantages:")
for adv in orpo_advantages:
    print(f"  - {adv}")
```

---

## 6. SimPO: 단순 선호도 최적화

### 6.1 SimPO 개요

SimPO(Simple Preference Optimization)는 길이 정규화(length-normalized) 로그 확률을
사용하고 참조 모델 의존성을 제거하여 DPO를 더욱 단순화한다.

```python
def simpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    chosen_lengths: torch.Tensor,
    rejected_lengths: torch.Tensor,
    beta: float = 2.0,
    gamma: float = 0.5,
) -> torch.Tensor:
    """
    Compute the SimPO loss.

    SimPO uses length-normalized log-probs as implicit rewards,
    eliminating the need for a reference model.

    r(x,y) = (1/|y|) * log pi(y|x)  (length-normalized)

    L_SimPO = -log sigma(beta * (r_w - r_l) - gamma)

    Args:
        policy_chosen_logps: Sum of log-probs for chosen
        policy_rejected_logps: Sum of log-probs for rejected
        chosen_lengths: Length of chosen sequences
        rejected_lengths: Length of rejected sequences
        beta: Scaling parameter
        gamma: Target margin (reward difference threshold)
    """
    # Length-normalized rewards
    chosen_rewards = policy_chosen_logps / chosen_lengths
    rejected_rewards = policy_rejected_logps / rejected_lengths

    # SimPO loss with margin
    logits = beta * (chosen_rewards - rejected_rewards) - gamma
    loss = -F.logsigmoid(logits).mean()

    return loss


# Demonstrate SimPO
torch.manual_seed(42)
batch_size = 8

policy_chosen = torch.randn(batch_size) * 2 - 5     # Sum of log-probs
policy_rejected = torch.randn(batch_size) * 2 - 7
chosen_lengths = torch.randint(10, 50, (batch_size,)).float()
rejected_lengths = torch.randint(10, 50, (batch_size,)).float()

simpo_l = simpo_loss(
    policy_chosen, policy_rejected,
    chosen_lengths, rejected_lengths,
    beta=2.0, gamma=0.5,
)

print("=== SimPO Loss ===\n")
print(f"SimPO Loss: {simpo_l.item():.4f}")

print("\nSimPO key innovations:")
innovations = [
    "Length normalization: prevents rewarding verbose responses",
    "No reference model: uses absolute log-probs as rewards",
    "Target margin (gamma): explicitly controls reward gap",
    "Simpler than DPO: fewer hyperparameters, no ref model forward pass",
]
for inn in innovations:
    print(f"  - {inn}")
```

---

## 7. 방법 간 선택

### 7.1 의사결정 프레임워크

```python
@dataclass
class MethodProfile:
    """Profile of a preference optimization method."""
    name: str
    reference_model_needed: bool
    paired_data_needed: bool
    training_stages: int
    memory_footprint: str    # low, medium, high
    empirical_quality: str   # strong, competitive, emerging
    best_for: str
    key_paper: str


method_profiles = [
    MethodProfile(
        name="RLHF (PPO)",
        reference_model_needed=True,
        paired_data_needed=True,
        training_stages=3,
        memory_footprint="high",
        empirical_quality="strong",
        best_for="Frontier model training where quality is paramount",
        key_paper="Ouyang et al., 2022 (InstructGPT)",
    ),
    MethodProfile(
        name="DPO",
        reference_model_needed=True,
        paired_data_needed=True,
        training_stages=2,
        memory_footprint="medium",
        empirical_quality="strong",
        best_for="Teams wanting RLHF-quality alignment without RL complexity",
        key_paper="Rafailov et al., 2023",
    ),
    MethodProfile(
        name="KTO",
        reference_model_needed=True,
        paired_data_needed=False,
        training_stages=2,
        memory_footprint="medium",
        empirical_quality="competitive",
        best_for="When only binary feedback is available (thumbs up/down)",
        key_paper="Ethayarajh et al., 2024",
    ),
    MethodProfile(
        name="IPO",
        reference_model_needed=True,
        paired_data_needed=True,
        training_stages=2,
        memory_footprint="medium",
        empirical_quality="competitive",
        best_for="When preference data is noisy with label errors",
        key_paper="Azar et al., 2023",
    ),
    MethodProfile(
        name="ORPO",
        reference_model_needed=False,
        paired_data_needed=True,
        training_stages=1,
        memory_footprint="low",
        empirical_quality="competitive",
        best_for="Memory-constrained settings; combined SFT+alignment",
        key_paper="Hong et al., 2024",
    ),
    MethodProfile(
        name="SimPO",
        reference_model_needed=False,
        paired_data_needed=True,
        training_stages=2,
        memory_footprint="low",
        empirical_quality="competitive",
        best_for="When length bias is a concern and compute is limited",
        key_paper="Meng et al., 2024",
    ),
]

print("=== Method Selection Guide ===\n")
print(f"{'Method':<12} {'Ref?':<6} {'Paired?':<9} {'Stages':<8} "
      f"{'Memory':<8} {'Quality':<12} {'Best For':<35}")
print("-" * 90)

for mp in method_profiles:
    ref = "Yes" if mp.reference_model_needed else "No"
    paired = "Yes" if mp.paired_data_needed else "No"
    print(
        f"{mp.name:<12} {ref:<6} {paired:<9} {mp.training_stages:<8} "
        f"{mp.memory_footprint:<8} {mp.empirical_quality:<12} "
        f"{mp.best_for[:33]:<35}"
    )
```

### 7.2 의사결정 트리

```python
class MethodDecisionTree:
    """Help users choose the right preference optimization method."""

    @staticmethod
    def recommend(
        has_paired_data: bool,
        compute_budget: str,     # low, medium, high
        data_quality: str,       # clean, noisy
        need_reference_model: bool,
        length_bias_concern: bool,
    ) -> str:
        """Recommend a method based on constraints."""
        recommendations = []

        if not has_paired_data:
            recommendations.append(("KTO", 0.9,
                                    "Only method that works with binary feedback"))
        else:
            if compute_budget == "low":
                if not need_reference_model:
                    recommendations.append(("ORPO", 0.85,
                                            "No ref model, single stage, low memory"))
                    if length_bias_concern:
                        recommendations.append(("SimPO", 0.80,
                                                "No ref model + length normalization"))
                else:
                    recommendations.append(("DPO", 0.75,
                                            "Simple but needs ref model"))

            elif compute_budget == "medium":
                if data_quality == "noisy":
                    recommendations.append(("IPO", 0.85,
                                            "Handles noisy labels well"))
                elif length_bias_concern:
                    recommendations.append(("SimPO", 0.85,
                                            "Length normalization addresses bias"))
                else:
                    recommendations.append(("DPO", 0.90,
                                            "Strong default choice"))

            else:  # high compute
                recommendations.append(("RLHF (PPO)", 0.85,
                                        "Best proven quality at scale"))
                recommendations.append(("DPO", 0.80,
                                        "Simpler alternative with strong results"))

        # Sort by confidence
        recommendations.sort(key=lambda x: -x[1])
        return recommendations

    @staticmethod
    def display_decision_examples() -> None:
        """Show example decisions."""
        print("=== Method Selection Examples ===\n")

        scenarios = [
            {
                "desc": "Startup with limited GPUs, clean paired data",
                "args": (True, "low", "clean", False, False),
            },
            {
                "desc": "Only have thumbs-up/down user feedback",
                "args": (False, "medium", "noisy", True, False),
            },
            {
                "desc": "Large lab training frontier model",
                "args": (True, "high", "clean", True, False),
            },
            {
                "desc": "Noisy crowdsourced preferences, medium compute",
                "args": (True, "medium", "noisy", True, False),
            },
            {
                "desc": "Concern about model being too verbose",
                "args": (True, "medium", "clean", False, True),
            },
        ]

        for scenario in scenarios:
            recs = MethodDecisionTree.recommend(*scenario["args"])
            print(f"Scenario: {scenario['desc']}")
            for method, conf, reason in recs:
                print(f"  -> {method} (confidence: {conf:.0%}): {reason}")
            print()


MethodDecisionTree.display_decision_examples()
```

### 7.3 방법별 한눈에 비교

아래 표는 빠른 참조를 위해 각 방법의 주요 속성을 요약합니다:

| 방법 (Method) | 참조 모델 필요 (Reference Model Required) | 쌍 데이터 필요 (Paired Data Required) | 계산 비용 (Computational Cost) | 주요 장점 (Key Advantage) | 주요 한계 (Key Limitation) |
|--------------|------------------------------------------|--------------------------------------|-------------------------------|--------------------------|--------------------------|
| DPO | 예 | 예 | 낮음 | 단순하고 안정적 | 오프라인 전용, 분포 이동 |
| KTO | 예 | 아니요 (이진 신호) | 낮음 | 좋아요/싫어요로 작동 | 쌍 데이터보다 정밀도 낮음 |
| IPO | 예 | 예 | 낮음 | 이론적 근거 탄탄 | 실제로는 DPO와 유사 |
| ORPO | 아니요 | 예 | 매우 낮음 | 참조 모델 불필요 | 최신, 검증 부족 |
| SimPO | 아니요 | 예 | 매우 낮음 | 길이 정규화, 단순함 | 매우 최신, 연구 제한적 |
| Online DPO | 예 | 예 (반복적) | 중간 | 분포 이동 해결 | 더 복잡한 파이프라인 |

---

## 8. Transformers와 TRL을 활용한 구현

### 8.1 TRL을 사용한 DPO

```python
class DPOImplementationGuide:
    """
    Guide for implementing DPO using the TRL (Transformer
    Reinforcement Learning) library by Hugging Face.
    """

    @staticmethod
    def show_trl_config() -> None:
        """Show TRL DPOConfig and DPOTrainer setup."""
        # This is pseudo-code showing how TRL is used in practice
        config_template = """
# === TRL DPO Training Configuration ===
#
# Install: pip install trl transformers datasets
#
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from trl import DPOTrainer, DPOConfig
# from datasets import load_dataset
#
# # 1. Load models
# model = AutoModelForCausalLM.from_pretrained("your-sft-model")
# ref_model = AutoModelForCausalLM.from_pretrained("your-sft-model")
# tokenizer = AutoTokenizer.from_pretrained("your-sft-model")
#
# # 2. Configure DPO training
# dpo_config = DPOConfig(
#     beta=0.1,                    # KL penalty coefficient
#     learning_rate=5e-7,          # Lower than SFT
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=4,
#     max_length=512,
#     max_prompt_length=256,
#     num_train_epochs=1,          # Usually 1 epoch
#     loss_type="sigmoid",         # "sigmoid" (DPO) or "ipo" or "kto"
#     warmup_ratio=0.1,
#     bf16=True,
#     logging_steps=10,
#     output_dir="./dpo-output",
# )
#
# # 3. Prepare dataset (must have: prompt, chosen, rejected)
# dataset = load_dataset("your-preference-dataset")
# # Format: {"prompt": str, "chosen": str, "rejected": str}
#
# # 4. Create trainer and train
# trainer = DPOTrainer(
#     model=model,
#     ref_model=ref_model,
#     args=dpo_config,
#     train_dataset=dataset["train"],
#     tokenizer=tokenizer,
# )
# trainer.train()
"""
        print(config_template)

    @staticmethod
    def show_dataset_format() -> None:
        """Show the expected dataset format for DPO training."""
        example_data = [
            {
                "prompt": "What is the capital of France?",
                "chosen": "The capital of France is Paris. Paris is located "
                          "in northern France along the Seine River.",
                "rejected": "France's capital is Lyon. It is a major city "
                            "in southeastern France.",
            },
            {
                "prompt": "Explain photosynthesis briefly.",
                "chosen": "Photosynthesis is the process by which plants "
                          "convert sunlight, water, and CO2 into glucose "
                          "and oxygen using chlorophyll.",
                "rejected": "Photosynthesis is when plants eat sunlight and "
                            "turn it into food. It's basically plant eating.",
            },
        ]

        print("=== DPO Dataset Format ===\n")
        for i, example in enumerate(example_data):
            print(f"Example {i+1}:")
            print(f"  Prompt:   {example['prompt']}")
            print(f"  Chosen:   {example['chosen'][:60]}...")
            print(f"  Rejected: {example['rejected'][:60]}...")
            print()


DPOImplementationGuide.show_trl_config()
DPOImplementationGuide.show_dataset_format()
```

### 8.2 커스텀 DPO 학습 루프

```python
class SimpleDPOTrainer:
    """
    Minimal DPO trainer implementation with PyTorch.

    This shows the core training logic without the full
    TRL infrastructure.
    """

    def __init__(
        self,
        policy: nn.Module,
        reference: nn.Module,
        beta: float = 0.1,
        lr: float = 1e-6,
        max_grad_norm: float = 1.0,
    ):
        self.policy = policy
        self.reference = reference
        self.beta = beta
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

        # Freeze reference
        for p in self.reference.parameters():
            p.requires_grad = False

    def get_logps(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Get log-probabilities from a model."""
        logits = model(inputs)
        return F.log_softmax(logits, dim=-1).mean(dim=-1)

    def train_step(
        self,
        chosen_inputs: torch.Tensor,
        rejected_inputs: torch.Tensor,
    ) -> dict:
        """Execute one DPO training step."""
        self.policy.train()

        # Policy log-probs
        policy_chosen_logps = self.get_logps(self.policy, chosen_inputs)
        policy_rejected_logps = self.get_logps(self.policy, rejected_inputs)

        # Reference log-probs (frozen)
        with torch.no_grad():
            ref_chosen_logps = self.get_logps(self.reference, chosen_inputs)
            ref_rejected_logps = self.get_logps(self.reference, rejected_inputs)

        # DPO loss
        loss, chosen_rewards, rejected_rewards = dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            beta=self.beta,
        )

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.max_grad_norm
        )
        self.optimizer.step()

        # Metrics
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        reward_margin = (chosen_rewards - rejected_rewards).mean()

        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "reward_margin": reward_margin.item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
        }


# Run a DPO training loop
torch.manual_seed(42)
input_dim = 32

policy = nn.Sequential(
    nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 16)
)
reference = nn.Sequential(
    nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 16)
)
reference.load_state_dict(policy.state_dict())

trainer = SimpleDPOTrainer(policy, reference, beta=0.1, lr=1e-4)

print("=== Custom DPO Training Loop ===\n")
print(f"{'Step':<6} {'Loss':<10} {'Accuracy':<10} {'Margin':<10} "
      f"{'Chosen R':<10} {'Rejected R':<12}")
print("-" * 58)

for step in range(20):
    chosen = torch.randn(16, input_dim) + 0.3    # Slightly higher signal
    rejected = torch.randn(16, input_dim) - 0.3

    metrics = trainer.train_step(chosen, rejected)

    if step % 2 == 0:
        print(
            f"{step:<6} "
            f"{metrics['loss']:<10.4f} "
            f"{metrics['accuracy']:<10.3f} "
            f"{metrics['reward_margin']:<10.4f} "
            f"{metrics['chosen_reward']:<10.4f} "
            f"{metrics['rejected_reward']:<12.4f}"
        )
```

---

## 9. 오프라인 vs 온라인 선호도 최적화

### 9.1 온라인-오프라인 스펙트럼

```python
@dataclass
class OptimizationMode:
    """A preference optimization mode on the online-offline spectrum."""
    name: str
    description: str
    data_source: str
    advantages: list[str]
    disadvantages: list[str]
    examples: list[str]


modes = [
    OptimizationMode(
        name="Fully Offline",
        description="Train on a fixed, pre-collected dataset of preferences. "
                    "No new data is generated during training.",
        data_source="Static dataset collected before training",
        advantages=[
            "Simple: standard supervised training pipeline",
            "Reproducible: same data every run",
            "Cheap: no generation during training",
        ],
        disadvantages=[
            "Distribution mismatch: data may not match current policy",
            "No self-improvement: cannot discover new good behaviors",
            "Data staleness: preferences may shift over time",
        ],
        examples=["DPO", "IPO", "ORPO", "SimPO"],
    ),
    OptimizationMode(
        name="Online (On-Policy)",
        description="Generate new responses from the current policy, "
                    "collect preferences on these, and train immediately.",
        data_source="Responses from current policy, evaluated in real-time",
        advantages=[
            "On-policy: data matches current model behavior",
            "Self-improvement: discovers new good behaviors",
            "Adapts to changing preferences",
        ],
        disadvantages=[
            "Expensive: requires generation + evaluation each step",
            "Infrastructure: needs reward model or human evaluators online",
            "Unstable: RL training challenges",
        ],
        examples=["RLHF (PPO)", "Online DPO", "RLHF with PPO"],
    ),
    OptimizationMode(
        name="Hybrid (Iterative Offline)",
        description="Alternate between generating new data with the current "
                    "policy and training offline on the collected data.",
        data_source="Periodically generated from current policy",
        advantages=[
            "Better coverage than pure offline",
            "More stable than pure online",
            "Practical compromise",
        ],
        disadvantages=[
            "Still requires periodic generation",
            "May not converge as well as pure online",
            "Scheduling generation rounds is an extra hyperparameter",
        ],
        examples=["Iterative DPO", "SPIN", "Self-Play methods"],
    ),
]

print("=== Online vs Offline Preference Optimization ===\n")
for mode in modes:
    print(f"Mode: {mode.name}")
    print(f"  {mode.description[:70]}...")
    print(f"  Methods: {', '.join(mode.examples)}")
    print(f"  Advantages:")
    for a in mode.advantages:
        print(f"    + {a}")
    print(f"  Disadvantages:")
    for d in mode.disadvantages:
        print(f"    - {d}")
    print()
```

### 9.2 분포 이동 분석

```python
class DistributionShiftAnalysis:
    """Analyze the impact of distribution shift on offline methods."""

    def __init__(self, seed: int = 42):
        torch.manual_seed(seed)

    def simulate_shift(
        self,
        n_steps: int = 50,
        shift_rate: float = 0.1,
    ) -> dict:
        """
        Simulate how offline training performance degrades as
        the policy diverges from the data-generating policy.
        """
        offline_quality = []
        online_quality = []
        policy_kl = []

        for step in range(n_steps):
            kl = step * shift_rate

            # Offline quality degrades as policy shifts
            offline_q = 0.8 * math.exp(-0.1 * kl) + 0.1

            # Online quality stays constant (on-policy data)
            online_q = 0.75 + random.gauss(0, 0.02)

            offline_quality.append(offline_q)
            online_quality.append(online_q)
            policy_kl.append(kl)

        return {
            "kl": policy_kl,
            "offline": offline_quality,
            "online": online_quality,
        }

    def display(self) -> None:
        """Display distribution shift analysis."""
        results = self.simulate_shift()

        print("=== Distribution Shift Impact ===\n")
        print(f"{'Step':<8} {'KL':<10} {'Offline':<12} {'Online':<12} {'Gap':<10}")
        print("-" * 52)

        for i in range(0, len(results["kl"]), 5):
            kl = results["kl"][i]
            off = results["offline"][i]
            on = results["online"][i]
            gap = on - off
            print(f"{i:<8} {kl:<10.2f} {off:<12.3f} {on:<12.3f} {gap:<10.3f}")

        print("\nConclusion: Offline methods degrade as the trained policy")
        print("diverges from the data-generating policy. This is the key")
        print("motivation for online/iterative DPO approaches.")


DistributionShiftAnalysis().display()
```

---

## 10. 실용적 고려사항

### 10.1 흔한 함정과 해결책

```python
@dataclass
class Pitfall:
    """A common pitfall in preference optimization."""
    name: str
    symptoms: list[str]
    cause: str
    solution: str
    applies_to: list[str]   # Which methods are affected


pitfalls = [
    Pitfall(
        name="Chosen/Rejected Reward Collapse",
        symptoms=[
            "Both chosen and rejected rewards converge to same value",
            "Loss plateaus at log(2) (random chance)",
            "Accuracy stuck near 50%",
        ],
        cause="Beta too low or learning rate too high. Policy changes "
              "too fast and loses the ability to distinguish preferences.",
        solution="Increase beta (more regularization). Decrease learning rate. "
                 "Check that reference model is correct.",
        applies_to=["DPO", "IPO", "SimPO"],
    ),
    Pitfall(
        name="Length Exploitation",
        symptoms=[
            "Model outputs get progressively longer",
            "Chosen responses are consistently longer than rejected",
            "Quality per token decreases",
        ],
        cause="Longer responses tend to have higher total log-probability, "
              "creating a bias toward verbosity.",
        solution="Use length-normalized log-probs (SimPO approach). "
                 "Filter training data for length balance. Add length penalty.",
        applies_to=["DPO", "RLHF", "ORPO"],
    ),
    Pitfall(
        name="Catastrophic Forgetting",
        symptoms=[
            "Model loses general capabilities after DPO training",
            "Perplexity on general text increases sharply",
            "Model becomes worse at tasks not in preference data",
        ],
        cause="DPO training overfits to the preference data distribution, "
              "degrading performance on other tasks.",
        solution="Use very low learning rate. Train for 1 epoch only. "
                 "Mix in SFT data during DPO training. Monitor general "
                 "benchmarks alongside preference metrics.",
        applies_to=["DPO", "IPO", "KTO", "SimPO", "ORPO"],
    ),
    Pitfall(
        name="Beta Too High",
        symptoms=[
            "Loss barely decreases",
            "Chosen/rejected rewards barely move",
            "Model behavior almost identical to reference",
        ],
        cause="High beta strongly regularizes toward reference, preventing "
              "the policy from learning new behavior.",
        solution="Decrease beta. Start with beta=0.1 and adjust based on "
                 "reward margins and accuracy.",
        applies_to=["DPO", "KTO"],
    ),
    Pitfall(
        name="Data Quality Issues",
        symptoms=[
            "Training loss oscillates or increases",
            "Accuracy decreases over training",
            "Model outputs degrade despite loss decreasing",
        ],
        cause="Noisy, inconsistent, or incorrectly labeled preference data.",
        solution="Audit data quality. Use IPO for robustness to label noise. "
                 "Filter low-confidence comparisons. Check for position bias.",
        applies_to=["DPO", "RLHF", "IPO", "KTO", "ORPO", "SimPO"],
    ),
]

print("=== Common Pitfalls in Preference Optimization ===\n")
for pitfall in pitfalls:
    print(f"Pitfall: {pitfall.name}")
    print(f"  Applies to: {', '.join(pitfall.applies_to)}")
    print(f"  Symptoms:")
    for s in pitfall.symptoms[:2]:
        print(f"    - {s}")
    print(f"  Cause: {pitfall.cause[:65]}...")
    print(f"  Solution: {pitfall.solution[:65]}...")
    print()
```

### 10.2 모니터링 체크리스트

```python
@dataclass
class MonitoringMetric:
    """A metric to monitor during preference optimization."""
    name: str
    what_to_track: str
    healthy_range: str
    red_flag: str


monitoring_checklist = [
    MonitoringMetric(
        "Loss",
        "Training loss should decrease steadily",
        "Decreasing from ~0.693 (log 2) toward 0.3-0.5",
        "Loss stuck at 0.693 (not learning) or increasing",
    ),
    MonitoringMetric(
        "Chosen/Rejected Accuracy",
        "Fraction of pairs where chosen_reward > rejected_reward",
        "70-90% (not too high, not too low)",
        "<55% (not separating) or >95% (possible overfitting)",
    ),
    MonitoringMetric(
        "Reward Margin",
        "Average (chosen_reward - rejected_reward)",
        "Steadily increasing, positive",
        "Negative, decreasing, or oscillating wildly",
    ),
    MonitoringMetric(
        "Implicit KL",
        "KL divergence between policy and reference",
        "Moderate growth (0-10 nats typical)",
        ">20 nats (policy has diverged too far)",
    ),
    MonitoringMetric(
        "Response Length",
        "Average length of generated responses",
        "Stable or slight increase",
        "Monotonically increasing (length exploitation)",
    ),
    MonitoringMetric(
        "General Benchmarks",
        "Performance on held-out tasks (MMLU, etc.)",
        "No significant degradation",
        "Degradation >5% (catastrophic forgetting)",
    ),
]

print("=== Preference Optimization Monitoring Checklist ===\n")
for metric in monitoring_checklist:
    print(f"Metric: {metric.name}")
    print(f"  Track: {metric.what_to_track}")
    print(f"  Healthy: {metric.healthy_range}")
    print(f"  Red flag: {metric.red_flag}")
    print()
```

### 10.3 분포 붕괴와 공변량 이동

오프라인 선호도 최적화에서 특별한 주의가 필요한 두 가지 실패 모드는 분포 붕괴(distribution collapse)와 공변량 이동(covariate shift)입니다. 이를 이해하면 실무자가 성능이 저하된 모델을 진단하고 적절한 완화 전략을 선택하는 데 도움이 됩니다.

**분포 붕괴(Distribution collapse)**는 DPO 훈련이 정책으로 하여금 좁은 응답 부분 집합을 제외한 모든 응답에 거의 0에 가까운 확률을 할당하게 할 때 발생합니다. 모델이 출력 공간의 작은 영역으로 효과적으로 "붕괴"됩니다. 관찰 가능한 징후는 다음과 같습니다:

- 출력 다양성 감소: 다양한 프롬프트에 걸쳐 유사하거나 동일한 응답
- 어휘나 표현 변화가 제한된 반복적인 생성
- 선호도 데이터셋에 표현되지 않은 작업에서의 능력 상실

분포 붕괴는 베타(β)가 낮을 때(약한 정규화), 선호도 데이터셋의 커버리지가 좁을 때, 또는 너무 많은 스텝 동안 훈련할 때 가장 발생하기 쉽습니다.

**공변량 이동(Covariate shift)**은 더 미묘한 문제입니다. 오프라인 DPO는 어떤 이전 정책(일반적으로 SFT 모델)에 의해 생성된 고정 데이터셋으로 훈련됩니다. 훈련이 진행되고 정책이 그 참조 지점에서 이탈함에 따라, 훈련 분포는 더 이상 현재 정책이 생성할 분포와 일치하지 않습니다. 모델이 자신에게 점점 더 분포 밖이 되는 데이터에 대해 최적화되고 있는 것입니다. 이로 인해 초기 개선 이후, 특히 장기 훈련에서 성능이 저하됩니다.

**오프라인 vs. 온라인 DPO — 전환 시점:**

- 고품질로 잘 선별된 선호도 데이터셋이 있고, 계산 자원이 제한되며, 작업 분포가 비교적 좁을 때 **오프라인 DPO**를 사용하십시오.
- 분포 붕괴 또는 공변량 이동 증상이 관찰되거나, 작업 분포가 넓거나 개방형이거나, 모델이 개선됨에 따라 지속적인 개선이 필요할 때 **온라인 또는 반복적 DPO**로 전환하십시오.

**소규모 데이터셋 지침:** 약 5,000개 미만의 선호도 쌍으로는 DPO가 일반화 가능한 선호도 신호를 학습하는 대신 훈련 예시에 과적합되기 쉽습니다. 실용적인 완화 방법은 다음과 같습니다:

- 베타(β)를 높이십시오(예: 0.1에서 0.3–0.5로). 참조 정책 방향으로 더 강한 정규화를 적용하고 정책 이동을 늦춥니다.
- 패러프레이징을 통해 데이터를 증강하십시오: 새로운 주석 없이 커버리지를 확장하기 위해 기존 선택/거부 쌍의 의미론적으로 동등한 변형을 생성합니다.
- KTO로 전환하십시오. KTO는 쌍이 없는 이진 피드백으로 작동하며, 매칭된 쌍을 요구하는 대신 더 큰 단일 응답 평가 풀을 활용할 수 있습니다.

---

## 연습문제

### 연습문제 1: DPO 처음부터 구현하기

모델 학습과 평가를 포함한 완전한 DPO 학습 파이프라인을 구현하라.

<details><summary>정답 보기</summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random


class PreferencePairDataset(Dataset):
    """Synthetic preference pair dataset."""

    def __init__(self, n_samples: int = 1000, dim: int = 32, seed: int = 42):
        torch.manual_seed(seed)
        # Chosen responses: systematically higher in first dimensions
        self.chosen = torch.randn(n_samples, dim)
        self.chosen[:, :dim // 4] += 1.0

        # Rejected responses: systematically lower
        self.rejected = torch.randn(n_samples, dim)
        self.rejected[:, :dim // 4] -= 0.5

        # Add 10% label noise
        n_noise = n_samples // 10
        noise_idx = torch.randperm(n_samples)[:n_noise]
        temp = self.chosen[noise_idx].clone()
        self.chosen[noise_idx] = self.rejected[noise_idx]
        self.rejected[noise_idx] = temp

    def __len__(self) -> int:
        return self.chosen.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.chosen[idx], self.rejected[idx]


class PolicyModel(nn.Module):
    """Simple policy model that outputs log-probabilities."""

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FullDPOTrainer:
    """Complete DPO trainer with training, evaluation, and monitoring."""

    def __init__(
        self,
        input_dim: int = 32,
        beta: float = 0.1,
        lr: float = 1e-4,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        self.beta = beta

        self.policy = PolicyModel(input_dim)
        self.reference = PolicyModel(input_dim)
        self.reference.load_state_dict(self.policy.state_dict())
        for p in self.reference.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)
        self.history: list[dict] = []

    def compute_dpo_loss(
        self,
        chosen: torch.Tensor,
        rejected: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute DPO loss and metrics."""
        pi_chosen = self.policy(chosen)
        pi_rejected = self.policy(rejected)

        with torch.no_grad():
            ref_chosen = self.reference(chosen)
            ref_rejected = self.reference(rejected)

        chosen_logratios = pi_chosen - ref_chosen
        rejected_logratios = pi_rejected - ref_rejected

        logits = self.beta * (chosen_logratios - rejected_logratios)
        loss = -F.logsigmoid(logits).mean()

        # Metrics
        with torch.no_grad():
            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            margin = (chosen_rewards - rejected_rewards).mean()
            implicit_kl = chosen_logratios.mean()

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "margin": margin.item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
            "implicit_kl": implicit_kl.item(),
        }

        return loss, metrics

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch."""
        self.policy.train()
        epoch_metrics: dict[str, list[float]] = {}

        for chosen, rejected in dataloader:
            loss, metrics = self.compute_dpo_loss(chosen, rejected)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            for k, v in metrics.items():
                epoch_metrics.setdefault(k, []).append(v)

        avg_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
        self.history.append(avg_metrics)
        return avg_metrics

    def evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluate on a dataset."""
        self.policy.eval()
        all_metrics: dict[str, list[float]] = {}

        with torch.no_grad():
            for chosen, rejected in dataloader:
                _, metrics = self.compute_dpo_loss(chosen, rejected)
                for k, v in metrics.items():
                    all_metrics.setdefault(k, []).append(v)

        return {k: sum(v) / len(v) for k, v in all_metrics.items()}

    def training_report(self) -> None:
        """Generate training report."""
        print("=== DPO Training Report ===\n")
        print(f"{'Epoch':<8} {'Loss':<10} {'Accuracy':<10} {'Margin':<10} "
              f"{'KL':<10}")
        print("-" * 48)
        for i, metrics in enumerate(self.history):
            print(
                f"{i+1:<8} "
                f"{metrics['loss']:<10.4f} "
                f"{metrics['accuracy']:<10.3f} "
                f"{metrics['margin']:<10.4f} "
                f"{metrics['implicit_kl']:<10.4f}"
            )


# Full training run
train_data = PreferencePairDataset(n_samples=800, seed=42)
val_data = PreferencePairDataset(n_samples=200, seed=123)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

trainer = FullDPOTrainer(input_dim=32, beta=0.1, lr=1e-3)

for epoch in range(10):
    train_metrics = trainer.train_epoch(train_loader)
    val_metrics = trainer.evaluate(val_loader)

    if epoch % 2 == 0:
        print(f"Epoch {epoch+1}: train_acc={train_metrics['accuracy']:.3f}, "
              f"val_acc={val_metrics['accuracy']:.3f}")

print()
trainer.training_report()
```

</details>

### 연습문제 2: 다중 방법 비교

DPO, IPO, KTO 손실을 구현하고 동일한 데이터에서의 동작을 비교하라.

<details><summary>정답 보기</summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiMethodComparison:
    """Compare DPO, IPO, and KTO on the same preference data."""

    def __init__(self, input_dim: int = 32, seed: int = 42):
        torch.manual_seed(seed)
        self.input_dim = input_dim

    def dpo_loss(
        self, pi_c: torch.Tensor, pi_r: torch.Tensor,
        ref_c: torch.Tensor, ref_r: torch.Tensor, beta: float
    ) -> torch.Tensor:
        logits = beta * ((pi_c - ref_c) - (pi_r - ref_r))
        return -F.logsigmoid(logits).mean()

    def ipo_loss(
        self, pi_c: torch.Tensor, pi_r: torch.Tensor,
        ref_c: torch.Tensor, ref_r: torch.Tensor, tau: float
    ) -> torch.Tensor:
        diff = (pi_c - ref_c) - (pi_r - ref_r)
        target = 1.0 / (2.0 * tau)
        return (diff - target).pow(2).mean()

    def kto_loss(
        self, pi: torch.Tensor, ref: torch.Tensor,
        is_desirable: torch.Tensor, beta: float
    ) -> torch.Tensor:
        log_ratios = pi - ref
        kl = log_ratios.mean().detach()
        desirable_loss = 1 - torch.sigmoid(beta * (log_ratios - kl))
        undesirable_loss = 1 - torch.sigmoid(beta * (kl - log_ratios))
        mask = is_desirable.float()
        loss = (mask * desirable_loss + (1 - mask) * undesirable_loss).mean()
        return loss

    def run_comparison(self, n_steps: int = 50, batch_size: int = 32) -> dict:
        """Train all three methods and compare."""
        results = {"dpo": [], "ipo": [], "kto": []}

        # Shared initial model weights
        init_weights = nn.Linear(self.input_dim, 1)

        for method_name in ["dpo", "ipo", "kto"]:
            model = nn.Sequential(
                nn.Linear(self.input_dim, 64), nn.ReLU(), nn.Linear(64, 1)
            )
            ref = nn.Sequential(
                nn.Linear(self.input_dim, 64), nn.ReLU(), nn.Linear(64, 1)
            )
            ref.load_state_dict(model.state_dict())
            for p in ref.parameters():
                p.requires_grad = False

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            for step in range(n_steps):
                chosen = torch.randn(batch_size, self.input_dim) + 0.3
                rejected = torch.randn(batch_size, self.input_dim) - 0.3

                pi_c = model(chosen).squeeze(-1)
                pi_r = model(rejected).squeeze(-1)
                with torch.no_grad():
                    ref_c = ref(chosen).squeeze(-1)
                    ref_r = ref(rejected).squeeze(-1)

                if method_name == "dpo":
                    loss = self.dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1)
                elif method_name == "ipo":
                    loss = self.ipo_loss(pi_c, pi_r, ref_c, ref_r, tau=0.1)
                else:  # kto
                    all_pi = torch.cat([pi_c, pi_r])
                    all_ref = torch.cat([ref_c, ref_r])
                    is_good = torch.cat([
                        torch.ones(batch_size, dtype=torch.bool),
                        torch.zeros(batch_size, dtype=torch.bool),
                    ])
                    loss = self.kto_loss(all_pi, all_ref, is_good, beta=0.1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Evaluate accuracy
                with torch.no_grad():
                    test_c = torch.randn(100, self.input_dim) + 0.3
                    test_r = torch.randn(100, self.input_dim) - 0.3
                    acc = (model(test_c) > model(test_r)).float().mean().item()

                results[method_name].append({
                    "step": step,
                    "loss": loss.item(),
                    "accuracy": acc,
                })

        return results

    def display_results(self, results: dict) -> None:
        """Display comparison results."""
        print("=== Multi-Method Comparison ===\n")
        print(f"{'Step':<8}", end="")
        for method in results:
            print(f"  {method}_loss  {method}_acc ", end="")
        print()
        print("-" * 70)

        for step in range(0, 50, 5):
            print(f"{step:<8}", end="")
            for method in results:
                data = results[method][step]
                print(f"  {data['loss']:<10.4f} {data['accuracy']:<10.3f}", end="")
            print()

        # Final comparison
        print(f"\n--- Final Results (step 49) ---")
        for method in results:
            final = results[method][-1]
            print(f"  {method.upper()}: loss={final['loss']:.4f}, "
                  f"accuracy={final['accuracy']:.3f}")


comp = MultiMethodComparison()
results = comp.run_comparison()
comp.display_results(results)
```

</details>

### 연습문제 3: Beta 민감도 분석

DPO의 beta 매개변수가 학습 역학과 최종 모델 품질에 미치는 영향을 분석하라.

<details><summary>정답 보기</summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaSensitivityAnalysis:
    """Analyze the sensitivity of DPO to the beta parameter."""

    def __init__(self, input_dim: int = 32, seed: int = 42):
        torch.manual_seed(seed)
        self.input_dim = input_dim

    def train_with_beta(
        self,
        beta: float,
        n_steps: int = 100,
        batch_size: int = 32,
    ) -> dict:
        """Train DPO with a specific beta and record metrics."""
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Linear(self.input_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        ref = nn.Sequential(
            nn.Linear(self.input_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        ref.load_state_dict(model.state_dict())
        for p in ref.parameters():
            p.requires_grad = False

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        history = {
            "loss": [], "accuracy": [], "margin": [],
            "kl": [], "chosen_reward": [], "rejected_reward": [],
        }

        for step in range(n_steps):
            chosen = torch.randn(batch_size, self.input_dim) + 0.3
            rejected = torch.randn(batch_size, self.input_dim) - 0.3

            pi_c = model(chosen).squeeze(-1)
            pi_r = model(rejected).squeeze(-1)
            with torch.no_grad():
                ref_c = ref(chosen).squeeze(-1)
                ref_r = ref(rejected).squeeze(-1)

            logratios_c = pi_c - ref_c
            logratios_r = pi_r - ref_r
            logits = beta * (logratios_c - logratios_r)
            loss = -F.logsigmoid(logits).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                rewards_c = beta * logratios_c
                rewards_r = beta * logratios_r
                accuracy = (rewards_c > rewards_r).float().mean()
                margin = (rewards_c - rewards_r).mean()
                kl = logratios_c.mean()

            history["loss"].append(loss.item())
            history["accuracy"].append(accuracy.item())
            history["margin"].append(margin.item())
            history["kl"].append(kl.item())
            history["chosen_reward"].append(rewards_c.mean().item())
            history["rejected_reward"].append(rewards_r.mean().item())

        return history

    def run_analysis(self) -> None:
        """Run analysis across multiple beta values."""
        betas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        all_results = {}

        for beta in betas:
            all_results[beta] = self.train_with_beta(beta)

        print("=== Beta Sensitivity Analysis ===\n")
        print(f"{'Beta':<8} {'Final Loss':<12} {'Final Acc':<12} "
              f"{'Final Margin':<14} {'Final KL':<12}")
        print("-" * 58)

        for beta in betas:
            h = all_results[beta]
            print(
                f"{beta:<8.2f} "
                f"{h['loss'][-1]:<12.4f} "
                f"{h['accuracy'][-1]:<12.3f} "
                f"{h['margin'][-1]:<14.4f} "
                f"{h['kl'][-1]:<12.4f}"
            )

        # Convergence speed analysis
        print(f"\n--- Convergence Speed (steps to reach 70% accuracy) ---")
        for beta in betas:
            h = all_results[beta]
            steps_to_70 = None
            for i, acc in enumerate(h["accuracy"]):
                if acc >= 0.70:
                    steps_to_70 = i
                    break
            status = f"{steps_to_70} steps" if steps_to_70 else "Never"
            print(f"  beta={beta:.2f}: {status}")

        print("\n--- Recommendation ---")
        print("  beta=0.1 is a good starting point for most cases.")
        print("  Decrease beta if policy is not learning (too regularized).")
        print("  Increase beta if KL is growing too fast (overfitting).")


analysis = BetaSensitivityAnalysis()
analysis.run_analysis()
```

</details>

### 연습문제 4: 선호도 데이터 품질 감사기

선호도 데이터 품질을 감사하고 개선점을 제안하는 도구를 구축하라.

<details><summary>정답 보기</summary>

```python
from dataclasses import dataclass, field
import random
from collections import Counter


@dataclass
class PreferencePair:
    """A single preference pair."""
    prompt: str
    chosen: str
    rejected: str
    chosen_length: int
    rejected_length: int
    annotator_id: str
    confidence: float
    category: str


class PreferenceDataAuditor:
    """Audit preference data quality for DPO/RLHF training."""

    def __init__(self) -> None:
        self.data: list[PreferencePair] = []
        self.issues: list[dict] = []

    def load_data(self, data: list[PreferencePair]) -> None:
        self.data = data

    def check_length_bias(self) -> dict:
        """Check if chosen responses are systematically longer."""
        length_diffs = [
            d.chosen_length - d.rejected_length for d in self.data
        ]
        avg_diff = sum(length_diffs) / len(length_diffs)
        positive_pct = sum(1 for d in length_diffs if d > 0) / len(length_diffs)

        has_bias = positive_pct > 0.7 or positive_pct < 0.3

        if has_bias:
            self.issues.append({
                "type": "length_bias",
                "severity": "high",
                "detail": f"Chosen is longer in {positive_pct:.0%} of cases "
                          f"(avg diff: {avg_diff:.1f} tokens)",
            })

        return {
            "avg_length_diff": avg_diff,
            "chosen_longer_pct": positive_pct,
            "has_bias": has_bias,
        }

    def check_annotator_agreement(self) -> dict:
        """Check for annotator consistency."""
        by_prompt: dict[str, list[str]] = {}
        for d in self.data:
            by_prompt.setdefault(d.prompt, []).append(d.annotator_id)

        # Check annotator workload distribution
        annotator_counts = Counter(d.annotator_id for d in self.data)
        max_count = max(annotator_counts.values())
        min_count = min(annotator_counts.values())

        if max_count > 3 * min_count:
            self.issues.append({
                "type": "annotator_imbalance",
                "severity": "medium",
                "detail": f"Workload imbalance: {min_count}-{max_count} "
                          f"pairs per annotator",
            })

        return {
            "n_annotators": len(annotator_counts),
            "annotator_counts": dict(annotator_counts),
            "workload_balanced": max_count <= 3 * min_count,
        }

    def check_confidence_distribution(self) -> dict:
        """Check the distribution of annotator confidence."""
        confidences = [d.confidence for d in self.data]
        avg_conf = sum(confidences) / len(confidences)
        low_conf = sum(1 for c in confidences if c < 0.3) / len(confidences)

        if low_conf > 0.3:
            self.issues.append({
                "type": "low_confidence",
                "severity": "high",
                "detail": f"{low_conf:.0%} of pairs have confidence < 0.3",
            })

        return {
            "avg_confidence": avg_conf,
            "low_confidence_pct": low_conf,
            "has_issue": low_conf > 0.3,
        }

    def check_category_balance(self) -> dict:
        """Check if categories are balanced in the dataset."""
        category_counts = Counter(d.category for d in self.data)
        total = len(self.data)
        proportions = {k: v / total for k, v in category_counts.items()}

        max_prop = max(proportions.values())
        min_prop = min(proportions.values())

        if max_prop > 5 * min_prop:
            self.issues.append({
                "type": "category_imbalance",
                "severity": "medium",
                "detail": f"Category imbalance: {dict(category_counts)}",
            })

        return {
            "category_counts": dict(category_counts),
            "proportions": proportions,
            "balanced": max_prop <= 5 * min_prop,
        }

    def full_audit(self) -> None:
        """Run full data quality audit."""
        print("=== Preference Data Quality Audit ===\n")
        print(f"Total pairs: {len(self.data)}\n")

        # Run all checks
        length_result = self.check_length_bias()
        annotator_result = self.check_annotator_agreement()
        confidence_result = self.check_confidence_distribution()
        category_result = self.check_category_balance()

        # Display results
        print("--- Length Bias Check ---")
        print(f"  Chosen longer in: {length_result['chosen_longer_pct']:.0%} of cases")
        print(f"  Average length diff: {length_result['avg_length_diff']:.1f} tokens")
        status = "FAIL" if length_result['has_bias'] else "PASS"
        print(f"  Status: {status}\n")

        print("--- Annotator Check ---")
        print(f"  Annotators: {annotator_result['n_annotators']}")
        status = "PASS" if annotator_result['workload_balanced'] else "FAIL"
        print(f"  Workload balanced: {status}\n")

        print("--- Confidence Check ---")
        print(f"  Average confidence: {confidence_result['avg_confidence']:.3f}")
        print(f"  Low confidence: {confidence_result['low_confidence_pct']:.0%}")
        status = "FAIL" if confidence_result['has_issue'] else "PASS"
        print(f"  Status: {status}\n")

        print("--- Category Balance ---")
        for cat, count in category_result['category_counts'].items():
            print(f"  {cat}: {count}")
        status = "PASS" if category_result['balanced'] else "FAIL"
        print(f"  Status: {status}\n")

        # Summary
        print("--- Issues Found ---")
        if self.issues:
            for issue in self.issues:
                print(f"  [{issue['severity'].upper()}] {issue['type']}: "
                      f"{issue['detail']}")
        else:
            print("  No issues found. Data quality looks good!")


# Generate synthetic data and audit
random.seed(42)
data = []
for i in range(200):
    chosen_len = random.randint(50, 500)
    rejected_len = random.randint(30, 300)
    # Introduce systematic length bias
    if random.random() < 0.75:
        chosen_len = max(chosen_len, rejected_len + 50)

    data.append(PreferencePair(
        prompt=f"prompt_{i % 20}",
        chosen=f"chosen_{i}",
        rejected=f"rejected_{i}",
        chosen_length=chosen_len,
        rejected_length=rejected_len,
        annotator_id=f"ann_{random.randint(1, 5)}",
        confidence=random.uniform(0.1, 1.0),
        category=random.choice(["general", "general", "general",
                                "safety", "coding"]),
    ))

auditor = PreferenceDataAuditor()
auditor.load_data(data)
auditor.full_audit()
```

</details>

### 연습문제 5: 방법 절삭 연구

동일한 데이터셋에서 제어된 변수로 DPO, IPO, KTO, SimPO를 비교하는 절삭 연구
(ablation study)를 수행하라.

<details><summary>정답 보기</summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dataclasses import dataclass


@dataclass
class AblationResult:
    """Result of one ablation experiment."""
    method: str
    final_accuracy: float
    final_loss: float
    convergence_step: int  # Step where accuracy first exceeds 80%
    max_accuracy: float
    training_stability: float  # Std of accuracy in last 20 steps


class AblationStudy:
    """Run controlled ablation study across methods."""

    def __init__(self, input_dim: int = 32, seed: int = 42):
        self.input_dim = input_dim
        self.seed = seed

    def create_model_pair(self) -> tuple[nn.Module, nn.Module]:
        """Create identical model and reference pair."""
        torch.manual_seed(self.seed)
        model = nn.Sequential(
            nn.Linear(self.input_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        ref = nn.Sequential(
            nn.Linear(self.input_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        ref.load_state_dict(model.state_dict())
        for p in ref.parameters():
            p.requires_grad = False
        return model, ref

    def generate_batch(self, batch_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of preference data."""
        chosen = torch.randn(batch_size, self.input_dim) + 0.3
        rejected = torch.randn(batch_size, self.input_dim) - 0.3
        return chosen, rejected

    def train_method(
        self,
        method: str,
        n_steps: int = 200,
        batch_size: int = 32,
    ) -> AblationResult:
        """Train with a specific method and record results."""
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        model, ref = self.create_model_pair()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        accuracies = []
        losses = []

        for step in range(n_steps):
            chosen, rejected = self.generate_batch(batch_size)

            pi_c = model(chosen).squeeze(-1)
            pi_r = model(rejected).squeeze(-1)
            with torch.no_grad():
                ref_c = ref(chosen).squeeze(-1)
                ref_r = ref(rejected).squeeze(-1)

            # Compute loss based on method
            if method == "dpo":
                logits = 0.1 * ((pi_c - ref_c) - (pi_r - ref_r))
                loss = -F.logsigmoid(logits).mean()

            elif method == "ipo":
                diff = (pi_c - ref_c) - (pi_r - ref_r)
                target = 1.0 / 0.2  # 1/(2*tau) with tau=0.1
                loss = (diff - target).pow(2).mean()

            elif method == "kto":
                all_pi = torch.cat([pi_c, pi_r])
                all_ref = torch.cat([ref_c, ref_r])
                log_ratios = all_pi - all_ref
                kl = log_ratios.mean().detach()
                n = len(pi_c)
                good_loss = 1 - torch.sigmoid(0.1 * (log_ratios[:n] - kl))
                bad_loss = 1 - torch.sigmoid(0.1 * (kl - log_ratios[n:]))
                loss = (good_loss.mean() + bad_loss.mean()) / 2

            elif method == "simpo":
                chosen_len = torch.ones_like(pi_c) * 20
                rejected_len = torch.ones_like(pi_r) * 20
                r_c = pi_c / chosen_len
                r_r = pi_r / rejected_len
                logits = 2.0 * (r_c - r_r) - 0.5
                loss = -F.logsigmoid(logits).mean()

            else:
                raise ValueError(f"Unknown method: {method}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Evaluate
            with torch.no_grad():
                test_c, test_r = self.generate_batch(100)
                acc = (model(test_c) > model(test_r)).float().mean().item()
            accuracies.append(acc)
            losses.append(loss.item())

        # Compute convergence step
        convergence = n_steps  # Default: never converged
        for i, acc in enumerate(accuracies):
            if acc >= 0.80:
                convergence = i
                break

        # Stability: std of last 20 accuracies
        stability = (
            torch.tensor(accuracies[-20:]).std().item()
            if len(accuracies) >= 20 else 1.0
        )

        return AblationResult(
            method=method,
            final_accuracy=accuracies[-1],
            final_loss=losses[-1],
            convergence_step=convergence,
            max_accuracy=max(accuracies),
            training_stability=stability,
        )

    def run_full_study(self) -> None:
        """Run ablation study across all methods."""
        methods = ["dpo", "ipo", "kto", "simpo"]
        results: list[AblationResult] = []

        print("=== Ablation Study: Preference Optimization Methods ===\n")
        print("Training all methods with identical data and initialization...\n")

        for method in methods:
            result = self.train_method(method)
            results.append(result)
            print(f"  {method.upper()} complete: acc={result.final_accuracy:.3f}")

        # Comparison table
        print(f"\n{'Method':<10} {'Final Acc':<12} {'Max Acc':<10} "
              f"{'Convergence':<14} {'Stability':<12} {'Final Loss':<12}")
        print("-" * 70)

        for r in results:
            conv_str = f"{r.convergence_step} steps" if r.convergence_step < 200 else "N/A"
            print(
                f"{r.method:<10} "
                f"{r.final_accuracy:<12.3f} "
                f"{r.max_accuracy:<10.3f} "
                f"{conv_str:<14} "
                f"{r.training_stability:<12.4f} "
                f"{r.final_loss:<12.4f}"
            )

        # Rankings
        print("\n--- Rankings ---")
        by_accuracy = sorted(results, key=lambda r: -r.final_accuracy)
        print(f"  By accuracy: {' > '.join(r.method for r in by_accuracy)}")

        by_speed = sorted(results, key=lambda r: r.convergence_step)
        print(f"  By convergence: {' > '.join(r.method for r in by_speed)}")

        by_stability = sorted(results, key=lambda r: r.training_stability)
        print(f"  By stability: {' > '.join(r.method for r in by_stability)}")

        # Overall winner
        print("\n--- Summary ---")
        print("All methods achieve comparable final accuracy on clean data.")
        print("Differences become more pronounced with noisy labels and")
        print("different data characteristics.")


study = AblationStudy()
study.run_full_study()
```

</details>

---

**이전**: [헌법적 AI](./04_Constitutional_AI.md) | **다음**: [확장 가능한 감독](./06_Scalable_Oversight.md)

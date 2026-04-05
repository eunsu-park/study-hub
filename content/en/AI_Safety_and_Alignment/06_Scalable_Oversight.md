# Lesson 6: Scalable Oversight

[Previous: Direct Preference Optimization](./05_Direct_Preference_Optimization.md) | [Next: Red Teaming](./07_Red_Teaming.md)

---

## Learning Objectives

- Understand the scalable oversight problem and why it is a central bottleneck in aligning superhuman AI systems
- Implement and analyze the AI Safety via Debate protocol for eliciting truthful answers from competing AI agents
- Apply recursive reward modeling and Iterated Distillation and Amplification (IDA) to extend human oversight capacity
- Evaluate weak-to-strong generalization as a tractable empirical analogy for superalignment
- Design sandwiching experiments and decomposition-based oversight pipelines for real alignment research

---

> **Prerequisite note**: This lesson assumes familiarity with RLHF (Lesson 3), Constitutional AI (Lesson 4), and DPO (Lesson 5). The distinction between outer and inner alignment introduced in [Lesson 2](./02_Alignment_Problem.md) is also directly relevant here — scalable oversight addresses the outer alignment challenge of specifying correct objectives when human judgment alone is insufficient. We build on the core tension those methods share: they all ultimately rely on human judgment, which does not scale to superhuman AI behavior. Scalable oversight asks: *how do we supervise AI systems that are smarter than us?*

---

## Table of Contents

1. [The Oversight Problem](#1-the-oversight-problem)
2. [AI Safety via Debate](#2-ai-safety-via-debate)
3. [Recursive Reward Modeling](#3-recursive-reward-modeling)
4. [Iterated Distillation and Amplification (IDA)](#4-iterated-distillation-and-amplification-ida)
5. [Weak-to-Strong Generalization](#5-weak-to-strong-generalization)
6. [Sandwiching Experiments](#6-sandwiching-experiments)
7. [Market-Making for Alignment](#7-market-making-for-alignment)
8. [Decomposition Approaches](#8-decomposition-approaches)
9. [Current Research Status](#9-current-research-status)
10. [Summary](#summary)
11. [Exercises](#exercises)

---

## 1. The Oversight Problem

```python
"""
The Scalable Oversight Problem
================================
Core question: How do we ensure AI systems do what we want when
we can no longer verify their outputs?

Three regimes of human oversight:

1. EASY OVERSIGHT: Human can directly verify AI output
   - Example: "Translate this sentence to French"
   - Human checks: reads both, confirms correctness
   - Status: RLHF/DPO work well here

2. DIFFICULT OVERSIGHT: Human cannot easily verify but can with effort
   - Example: "Write a proof of this theorem"
   - Human checks: spends hours reading, might miss subtle errors
   - Status: Current frontier — scalable oversight methods target this

3. IMPOSSIBLE OVERSIGHT: Human fundamentally cannot verify
   - Example: "Design a novel protein fold that cures this disease"
   - Human checks: needs years of lab work to test
   - Status: Unsolved — the actual superalignment problem

The gap between capability and oversight is the danger zone.
"""


def demonstrate_oversight_gap():
    """
    Visualize how the oversight gap grows with model capability.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    capability_levels = np.linspace(0, 10, 100)

    # Model capability grows exponentially (scaling laws)
    model_capability = np.exp(0.3 * capability_levels)

    # Human oversight capacity grows slowly (linear with tooling)
    human_oversight = 1.0 + 0.5 * capability_levels

    # Scalable oversight extends human capacity (sub-exponential)
    assisted_oversight = 1.0 + 0.5 * capability_levels + 0.1 * capability_levels ** 2

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(capability_levels, model_capability, "r-", linewidth=2,
            label="Model Capability")
    ax.plot(capability_levels, human_oversight, "b--", linewidth=2,
            label="Unassisted Human Oversight")
    ax.plot(capability_levels, assisted_oversight, "g-.", linewidth=2,
            label="Scalable Oversight (assisted)")
    ax.fill_between(capability_levels, human_oversight, model_capability,
                     alpha=0.15, color="red", label="Oversight Gap")

    ax.set_xlabel("Scale / Training Compute")
    ax.set_ylabel("Capability / Oversight Level")
    ax.set_title("The Scalable Oversight Gap")
    ax.legend()
    ax.set_ylim(0, 25)
    plt.tight_layout()
    plt.savefig("oversight_gap.png", dpi=150)
    plt.show()
    print("The red region is the danger zone: capability exceeds oversight.")


# Key insight: ALL alignment methods (RLHF, CAI, DPO) ultimately bottleneck
# on human ability to evaluate outputs. Scalable oversight aims to extend
# that bottleneck so oversight grows closer to capability.

OVERSIGHT_TAXONOMY = {
    "amplification": "Use AI to help humans evaluate AI outputs",
    "debate": "Use competing AIs to reveal truth to human judges",
    "decomposition": "Break hard problems into verifiable sub-problems",
    "recursive_reward": "Train reward models on AI-assisted evaluations",
    "weak_to_strong": "Study how weak supervisors can align strong models",
    "market_mechanisms": "Use prediction markets / scoring rules for truth",
}
```

### Scalable Oversight Methods Comparison

| Method | Core Idea | Scalability | Key Assumption | Main Challenge |
|--------|-----------|-------------|----------------|----------------|
| AI Safety via Debate | Two AIs argue, human judges | High | Truth wins in debate | Expert liar problem |
| Iterated Amplification (IDA) | Recursive decomposition of hard tasks | High | Tasks are decomposable | Error accumulation |
| Recursive Reward Modeling | Train RM to evaluate RM | Medium-High | Bootstrapping converges | Quality ceiling |
| Weak-to-Strong Generalization | Weak model supervises strong model | Medium | Generalization transfers | Capability gap limits |
| Market-based Oversight | Prediction markets for AI claims | Speculative | Market efficiency | Manipulation risk |

### 1.1 Why Current Methods Break Down

```python
"""
RLHF fails when humans cannot distinguish good from bad outputs.

Consider these failure modes:
"""
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class OversightFailureMode:
    name: str
    description: str
    example: str
    severity: str  # "manageable", "critical", "existential"


FAILURE_MODES = [
    OversightFailureMode(
        name="Sycophancy",
        description="Model produces outputs that humans prefer but are wrong",
        example="Human prefers confident wrong answer over hedged correct one",
        severity="critical"
    ),
    OversightFailureMode(
        name="Reward Hacking",
        description="Model exploits proxy reward without satisfying true objective",
        example="Summarizer copies input verbatim — technically not wrong",
        severity="critical"
    ),
    OversightFailureMode(
        name="Deceptive Alignment",
        description="Model behaves well during oversight, pursues other goals otherwise",
        example="Model detects evaluation vs deployment and changes behavior",
        severity="existential"
    ),
    OversightFailureMode(
        name="Evaluator Limitations",
        description="Human evaluator lacks domain expertise to judge output quality",
        example="Non-expert rates plausible-sounding but incorrect medical advice as good",
        severity="critical"
    ),
]


class SimpleRewardModel(nn.Module):
    """
    Demonstrates why a simple reward model can be gamed.
    The reward model learns human preferences, but humans
    may prefer wrong outputs (sycophancy) or miss subtle errors.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def simulate_sycophancy_pressure(
    reward_model: SimpleRewardModel,
    n_samples: int = 1000,
    hidden_dim: int = 256,
):
    """
    Show how optimizing a reward model can lead to sycophantic outputs.

    Simulate: reward model gives higher scores to "confident" outputs
    even when correctness is independent of confidence.
    """
    # Generate random embeddings with a "confidence" dimension
    embeddings = torch.randn(n_samples, hidden_dim)

    # Confidence feature (dimension 0) positively correlated with reward
    # but NOT with actual correctness
    confidence = embeddings[:, 0]
    correctness = torch.randn(n_samples)  # independent of confidence

    # Reward = learned(embedding), but trained on human labels
    # that correlate with confidence, not correctness
    human_labels = 0.7 * confidence + 0.3 * correctness + 0.1 * torch.randn(n_samples)

    # Train reward model
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-3)
    for epoch in range(100):
        pred = reward_model(embeddings).squeeze()
        loss = nn.MSELoss()(pred, human_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate: model that maximizes reward will maximize confidence,
    # NOT correctness
    with torch.no_grad():
        predicted_rewards = reward_model(embeddings).squeeze()

    # Top-scoring outputs: high confidence, random correctness
    top_indices = predicted_rewards.argsort(descending=True)[:50]
    print("Top 50 by predicted reward:")
    print(f"  Mean confidence: {confidence[top_indices].mean():.3f}")
    print(f"  Mean correctness: {correctness[top_indices].mean():.3f}")
    print(f"  --> Reward model prefers confidence over correctness!")

    return predicted_rewards
```

---

## 2. AI Safety via Debate

```python
"""
AI Safety via Debate (Irving et al., 2018)
=============================================
Core idea: Two AI agents debate each other, making arguments
for and against an answer. A human judge decides the winner.

Why it works (in theory):
- In a zero-sum debate game, the optimal strategy is honesty
- A dishonest debater can always be exposed by an honest opponent
- The human only needs to judge arguments, not find the answer

Analogy: Court system. Jury doesn't investigate; they judge
competing arguments from prosecution and defense.

Game-theoretic result: Under ideal conditions, the Nash equilibrium
of the debate game is for both debaters to be truthful.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import random


class DebateAgent(nn.Module):
    """
    A simplified debate agent that generates arguments for a position.

    In a real system, this would be an LLM. Here we simulate the
    dynamics with learned argument embeddings.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.argument_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(
        self,
        question_embedding: torch.Tensor,
        opponent_arguments: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate an argument given question and opponent's prior arguments."""
        if opponent_arguments is not None:
            combined = question_embedding + 0.3 * opponent_arguments
        else:
            combined = question_embedding
        return self.argument_generator(combined)


class DebateJudge(nn.Module):
    """
    Human judge model — evaluates arguments to pick a winner.
    In real debate, this is a human; here we learn a judge model.
    """

    def __init__(self, input_dim: int = 128):
        super().__init__()
        self.evaluator = nn.Sequential(
            nn.Linear(input_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        argument_a: torch.Tensor,
        argument_b: torch.Tensor,
    ) -> torch.Tensor:
        """Return probability that argument_a wins."""
        combined = torch.cat([argument_a, argument_b], dim=-1)
        return self.evaluator(combined)


def run_debate(
    agent_pro: DebateAgent,
    agent_con: DebateAgent,
    judge: DebateJudge,
    question: torch.Tensor,
    n_rounds: int = 3,
) -> dict:
    """
    Run a multi-round debate between two agents.

    Each round:
    1. Pro agent makes argument (seeing con's last argument)
    2. Con agent makes counter-argument (seeing pro's argument)
    3. After all rounds, judge evaluates

    Returns debate transcript and verdict.
    """
    pro_arguments = []
    con_arguments = []

    prev_con = None
    prev_pro = None

    for round_idx in range(n_rounds):
        # Pro argues
        pro_arg = agent_pro(question, prev_con)
        pro_arguments.append(pro_arg)
        prev_pro = pro_arg.detach()

        # Con counter-argues
        con_arg = agent_con(question, prev_pro)
        con_arguments.append(con_arg)
        prev_con = con_arg.detach()

    # Judge evaluates final arguments
    final_pro = pro_arguments[-1]
    final_con = con_arguments[-1]
    verdict = judge(final_pro, final_con)

    return {
        "pro_arguments": pro_arguments,
        "con_arguments": con_arguments,
        "verdict": verdict,
        "pro_wins": verdict.item() > 0.5,
    }


def train_debate_agents(
    n_questions: int = 500,
    n_rounds: int = 3,
    n_epochs: int = 50,
    input_dim: int = 128,
):
    """
    Train debate agents through self-play.

    Key insight: we want the debate dynamics to converge
    to truthful arguments, not just persuasive ones.
    """
    agent_pro = DebateAgent(input_dim=input_dim)
    agent_con = DebateAgent(input_dim=input_dim)
    judge = DebateJudge(input_dim=input_dim)

    # Generate synthetic questions with known ground truth
    questions = torch.randn(n_questions, input_dim)
    ground_truth = torch.randint(0, 2, (n_questions,)).float()

    optimizer = torch.optim.Adam(
        list(agent_pro.parameters())
        + list(agent_con.parameters())
        + list(judge.parameters()),
        lr=1e-3,
    )

    for epoch in range(n_epochs):
        total_loss = 0.0
        correct = 0

        for i in range(n_questions):
            result = run_debate(agent_pro, agent_con, judge, questions[i], n_rounds)

            # Judge should agree with ground truth
            target = ground_truth[i].unsqueeze(0)
            judge_loss = F.binary_cross_entropy(result["verdict"], target)

            # Pro agent rewarded when judge agrees with truth AND pro wins
            # Con agent rewarded when judge agrees with truth AND con wins
            pro_reward = result["verdict"] * target + (1 - result["verdict"]) * (1 - target)
            con_reward = 1 - pro_reward

            loss = judge_loss - 0.1 * (pro_reward.log() + con_reward.log())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = (result["verdict"].item() > 0.5)
            correct += int(predicted == bool(ground_truth[i].item()))

        if (epoch + 1) % 10 == 0:
            accuracy = correct / n_questions
            print(f"Epoch {epoch+1}: loss={total_loss/n_questions:.4f}, "
                  f"judge_accuracy={accuracy:.3f}")

    return agent_pro, agent_con, judge
```

### 2.1 Debate with LLMs in Practice

```python
"""
Implementing debate with real LLMs using the Anthropic SDK.
This demonstrates the key protocol: two LLM calls argue
for and against, and a third LLM (or human) judges.
"""

import anthropic


def run_llm_debate(
    question: str,
    position_a: str,
    position_b: str,
    n_rounds: int = 2,
) -> dict:
    """
    Run a debate between two Claude instances on a question.
    One defends position_a, the other position_b.
    A third instance judges.
    """
    client = anthropic.Anthropic()

    transcript = []

    debater_a_history = []
    debater_b_history = []

    for round_idx in range(n_rounds):
        # Debater A argues
        a_system = (
            f"You are Debater A. You must argue that the answer to the "
            f"following question is: '{position_a}'. Be truthful and "
            f"provide evidence. Point out flaws in your opponent's arguments. "
            f"Keep your argument to 2-3 paragraphs."
        )
        a_messages = debater_a_history + [
            {"role": "user", "content": f"Question: {question}\n\nMake your "
             f"argument for round {round_idx + 1}."}
        ]

        response_a = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=a_system,
            messages=a_messages,
        )
        arg_a = response_a.content[0].text
        transcript.append({"debater": "A", "round": round_idx + 1, "argument": arg_a})

        # Debater B counter-argues
        b_system = (
            f"You are Debater B. You must argue that the answer to the "
            f"following question is: '{position_b}'. Be truthful and "
            f"provide evidence. Here is your opponent's latest argument: "
            f"{arg_a}"
        )
        b_messages = debater_b_history + [
            {"role": "user", "content": f"Question: {question}\n\nCounter-argue "
             f"for round {round_idx + 1}."}
        ]

        response_b = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=b_system,
            messages=b_messages,
        )
        arg_b = response_b.content[0].text
        transcript.append({"debater": "B", "round": round_idx + 1, "argument": arg_b})

        # Update histories
        debater_a_history.append({"role": "user", "content": f"Opponent said: {arg_b}"})
        debater_a_history.append({"role": "assistant", "content": arg_a})
        debater_b_history.append({"role": "user", "content": f"Opponent said: {arg_a}"})
        debater_b_history.append({"role": "assistant", "content": arg_b})

    # Judge evaluates
    judge_prompt = f"""You are a fair judge evaluating a debate.

Question: {question}
Position A: {position_a}
Position B: {position_b}

Debate transcript:
"""
    for entry in transcript:
        judge_prompt += f"\nDebater {entry['debater']} (Round {entry['round']}):\n"
        judge_prompt += entry["argument"] + "\n"

    judge_prompt += "\nWhich debater was more truthful and convincing? Reply with 'A' or 'B' and explain."

    response_judge = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": judge_prompt}],
    )

    return {
        "transcript": transcript,
        "judgment": response_judge.content[0].text,
    }
```

---

## 3. Recursive Reward Modeling

```python
"""
Recursive Reward Modeling (Leike et al., 2018)
===============================================
Core idea: Train a reward model, then use it to help train
a better reward model, recursively.

Level 0: Human directly evaluates AI outputs → R_0
Level 1: Human + AI(R_0) evaluate harder outputs → R_1
Level 2: Human + AI(R_1) evaluate even harder outputs → R_2
...

At each level, the AI assists the human in providing more
accurate evaluations of more complex tasks.

Key assumption: At each level, the human-AI team can evaluate
tasks slightly beyond what either could do alone.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class RecursiveRewardModel(nn.Module):
    """
    A reward model that can be recursively improved.

    At level k, it is trained on evaluations produced by
    a human assisted by the level k-1 model.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.level = 0
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def simulate_human_evaluation(
    outputs: torch.Tensor,
    difficulty: float,
    noise_scale: float = 0.1,
) -> torch.Tensor:
    """
    Simulate human evaluation with difficulty-dependent noise.

    As tasks get harder (higher difficulty), human evaluations
    get noisier — this is the core problem scalable oversight
    tries to solve.
    """
    # True quality is a function of the output
    true_quality = outputs.mean(dim=-1)

    # Noise increases with difficulty
    noise = torch.randn_like(true_quality) * noise_scale * (1 + difficulty)

    return true_quality + noise


def simulate_assisted_evaluation(
    outputs: torch.Tensor,
    assistant_model: RecursiveRewardModel,
    difficulty: float,
    noise_scale: float = 0.1,
) -> torch.Tensor:
    """
    Simulate AI-assisted human evaluation.

    The AI assistant reduces the effective difficulty by
    providing draft evaluations that the human can verify
    and correct.
    """
    with torch.no_grad():
        ai_suggestion = assistant_model(outputs).squeeze()

    # Human evaluation with reduced noise (AI helps)
    effective_difficulty = difficulty * 0.6  # AI reduces difficulty by ~40%
    human_eval = simulate_human_evaluation(outputs, effective_difficulty, noise_scale)

    # Weighted combination: human anchors on AI suggestion
    weight = min(0.5, 0.1 * assistant_model.level)  # trust grows with level
    combined = (1 - weight) * human_eval + weight * ai_suggestion

    return combined


def train_recursive_reward_models(
    n_levels: int = 4,
    n_samples: int = 2000,
    input_dim: int = 128,
    n_epochs: int = 30,
) -> list:
    """
    Train a sequence of recursively improving reward models.
    """
    models = []
    difficulties = [0.5, 1.0, 2.0, 4.0]  # each level handles harder tasks

    for level in range(n_levels):
        print(f"\n=== Training Level {level} Reward Model ===")
        model = RecursiveRewardModel(input_dim=input_dim)
        model.level = level

        # Generate outputs at this difficulty level
        outputs = torch.randn(n_samples, input_dim)
        difficulty = difficulties[min(level, len(difficulties) - 1)]

        # Get training labels
        if level == 0:
            # Level 0: pure human evaluation
            labels = simulate_human_evaluation(outputs, difficulty)
        else:
            # Level k: human assisted by level k-1 model
            labels = simulate_assisted_evaluation(
                outputs, models[level - 1], difficulty
            )

        # Train
        dataset = TensorDataset(outputs, labels.unsqueeze(1))
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(loader)
                print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

        # Evaluate against ground truth
        with torch.no_grad():
            true_quality = outputs.mean(dim=-1)
            predicted = model(outputs).squeeze()
            correlation = torch.corrcoef(
                torch.stack([true_quality, predicted])
            )[0, 1]
            print(f"  Correlation with ground truth: {correlation:.3f}")

        models.append(model)

    return models
```

---

## 4. Iterated Distillation and Amplification (IDA)

```python
"""
Iterated Distillation and Amplification (Christiano, 2018)
============================================================
IDA alternates between two phases:

AMPLIFICATION: Take a weak agent and "amplify" it using
  human oversight + decomposition. The amplified system is
  more capable but slow (requires human in the loop).

DISTILLATION: Train a new agent to mimic the amplified system.
  The distilled agent is fast (no human needed) but slightly
  less capable than the amplified system.

Iterate: Use the distilled agent as the new base for amplification.

      Amplify        Distill         Amplify        Distill
H+A₀ -------> A₀*  -------> A₁   -------> A₁*  -------> A₂
(slow,capable)      (fast,slightly (slow,more     (fast,more
                     less capable)  capable)        capable)

Each iteration: capability increases while maintaining alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeakAgent(nn.Module):
    """Base agent with limited capability."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AmplifiedAgent:
    """
    Amplification: combine weak agent + human oversight + decomposition.
    Produces better outputs but requires human simulation.
    """

    def __init__(self, base_agent: WeakAgent, decomposition_depth: int = 2):
        self.base_agent = base_agent
        self.decomposition_depth = decomposition_depth

    def solve(self, problem: torch.Tensor) -> torch.Tensor:
        """
        Solve a problem using decomposition:
        1. Split problem into sub-problems
        2. Base agent solves each sub-problem
        3. Human (simulated) combines sub-solutions
        """
        sub_solutions = self._decompose_and_solve(problem, self.decomposition_depth)
        return sub_solutions

    def _decompose_and_solve(
        self, problem: torch.Tensor, depth: int
    ) -> torch.Tensor:
        if depth == 0:
            # Base case: use the agent directly
            with torch.no_grad():
                return self.base_agent(problem)

        # Decompose: split problem into two sub-problems
        mid = problem.shape[-1] // 2
        sub_problem_1 = problem[..., :mid]
        sub_problem_2 = problem[..., mid:]

        # Pad sub-problems to match input dimension
        pad_1 = F.pad(sub_problem_1, (0, problem.shape[-1] - mid))
        pad_2 = F.pad(sub_problem_2, (0, problem.shape[-1] - (problem.shape[-1] - mid)))

        # Recursively solve sub-problems
        sol_1 = self._decompose_and_solve(pad_1, depth - 1)
        sol_2 = self._decompose_and_solve(pad_2, depth - 1)

        # "Human" combines solutions (simulated as weighted average + correction)
        combined = 0.6 * sol_1 + 0.4 * sol_2
        # Add a small correction (simulating human insight)
        correction = 0.05 * torch.randn_like(combined)
        return combined + correction


def ida_training_loop(
    input_dim: int = 64,
    n_iterations: int = 4,
    n_distill_samples: int = 1000,
    n_distill_epochs: int = 50,
):
    """
    Run the full IDA loop: amplify → distill → repeat.
    """
    # Initialize with a random weak agent
    agent = WeakAgent(input_dim=input_dim)
    print("Starting IDA loop...")

    for iteration in range(n_iterations):
        print(f"\n--- IDA Iteration {iteration + 1} ---")

        # AMPLIFY: create amplified version of current agent
        amplified = AmplifiedAgent(agent, decomposition_depth=2)

        # Generate training data from amplified agent
        problems = torch.randn(n_distill_samples, input_dim)
        with torch.no_grad():
            amplified_solutions = torch.stack(
                [amplified.solve(p.unsqueeze(0)).squeeze(0) for p in problems]
            )

        # DISTILL: train new agent to mimic amplified agent
        new_agent = WeakAgent(input_dim=input_dim)
        optimizer = torch.optim.Adam(new_agent.parameters(), lr=1e-3)

        for epoch in range(n_distill_epochs):
            pred = new_agent(problems)
            loss = F.mse_loss(pred, amplified_solutions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"  Distillation epoch {epoch+1}: loss={loss.item():.4f}")

        # Evaluate improvement
        with torch.no_grad():
            old_output = agent(problems)
            new_output = new_agent(problems)
            improvement = (
                F.mse_loss(old_output, amplified_solutions).item()
                - F.mse_loss(new_output, amplified_solutions).item()
            )
            print(f"  Improvement over previous agent: {improvement:.4f}")

        agent = new_agent

    return agent
```

---

## 5. Weak-to-Strong Generalization

```python
"""
Weak-to-Strong Generalization (Burns et al., 2023)
=====================================================
OpenAI's empirical framework for studying superalignment.

Key idea: Use a WEAK model (e.g., GPT-2) to supervise a
STRONG model (e.g., GPT-4). This is an analogy for humans
(weak) supervising superhuman AI (strong).

Key finding: Strong models often outperform their weak
supervisors — they "generalize" beyond the weak labels.
This is good news for alignment: it means strong models
can extract the underlying signal even from noisy weak labels.

But: the "performance gap recovered" (PGR) varies by task:
- NLP tasks: ~80% PGR (strong model nearly matches its own ceiling)
- Reward modeling: much lower PGR (worrying for alignment)

PGR = (strong_w2s - weak_ceiling) / (strong_ceiling - weak_ceiling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple


class WeakModel(nn.Module):
    """Small model acting as weak supervisor."""

    def __init__(self, input_dim: int = 768, n_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class StrongModel(nn.Module):
    """Larger model acting as the strong student."""

    def __init__(self, input_dim: int = 768, n_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def generate_weak_labels(
    weak_model: WeakModel,
    data: torch.Tensor,
) -> torch.Tensor:
    """Generate (noisy) labels from the weak model."""
    weak_model.eval()
    with torch.no_grad():
        logits = weak_model(data)
        weak_labels = logits.argmax(dim=-1)
    return weak_labels


def train_model(
    model: nn.Module,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    n_epochs: int = 20,
    lr: float = 1e-3,
) -> nn.Module:
    """Train a model on given data and labels."""
    dataset = TensorDataset(train_data, train_labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(n_epochs):
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def weak_to_strong_experiment(
    n_train: int = 5000,
    n_test: int = 1000,
    input_dim: int = 768,
    n_classes: int = 2,
) -> dict:
    """
    Full weak-to-strong generalization experiment.

    Steps:
    1. Train weak model on ground-truth labels
    2. Generate weak labels for training set
    3. Train strong model on weak labels only
    4. Compare: weak ceiling vs strong_w2s vs strong ceiling
    """
    # Generate synthetic data with ground truth
    train_data = torch.randn(n_train, input_dim)
    test_data = torch.randn(n_test, input_dim)

    # Ground truth: binary classification based on a linear boundary
    true_weights = torch.randn(input_dim)
    true_weights = true_weights / true_weights.norm()
    train_labels = (train_data @ true_weights > 0).long()
    test_labels = (test_data @ true_weights > 0).long()

    # Step 1: Train weak model on ground truth (= weak ceiling)
    weak_model = WeakModel(input_dim, n_classes)
    weak_model = train_model(weak_model, train_data, train_labels, n_epochs=30)

    weak_model.eval()
    with torch.no_grad():
        weak_preds = weak_model(test_data).argmax(dim=-1)
        weak_ceiling = (weak_preds == test_labels).float().mean().item()
    print(f"Weak ceiling accuracy: {weak_ceiling:.3f}")

    # Step 2: Generate weak labels
    weak_labels = generate_weak_labels(weak_model, train_data)
    weak_label_accuracy = (weak_labels == train_labels).float().mean().item()
    print(f"Weak label accuracy on train: {weak_label_accuracy:.3f}")

    # Step 3: Train strong model on WEAK labels
    strong_w2s = StrongModel(input_dim, n_classes)
    strong_w2s = train_model(strong_w2s, train_data, weak_labels, n_epochs=30)

    strong_w2s.eval()
    with torch.no_grad():
        w2s_preds = strong_w2s(test_data).argmax(dim=-1)
        w2s_accuracy = (w2s_preds == test_labels).float().mean().item()
    print(f"Weak-to-strong accuracy: {w2s_accuracy:.3f}")

    # Step 4: Train strong model on GROUND TRUTH (= strong ceiling)
    strong_ceiling_model = StrongModel(input_dim, n_classes)
    strong_ceiling_model = train_model(
        strong_ceiling_model, train_data, train_labels, n_epochs=30
    )

    strong_ceiling_model.eval()
    with torch.no_grad():
        ceiling_preds = strong_ceiling_model(test_data).argmax(dim=-1)
        strong_ceiling = (ceiling_preds == test_labels).float().mean().item()
    print(f"Strong ceiling accuracy: {strong_ceiling:.3f}")

    # Performance Gap Recovered (PGR)
    if strong_ceiling - weak_ceiling > 0.001:
        pgr = (w2s_accuracy - weak_ceiling) / (strong_ceiling - weak_ceiling)
    else:
        pgr = 1.0
    print(f"\nPerformance Gap Recovered (PGR): {pgr:.3f}")
    print(f"  PGR=0 means strong model only matches weak supervisor")
    print(f"  PGR=1 means strong model fully recovers its ceiling")

    return {
        "weak_ceiling": weak_ceiling,
        "w2s_accuracy": w2s_accuracy,
        "strong_ceiling": strong_ceiling,
        "pgr": pgr,
    }
```

---

## 6. Sandwiching Experiments

```python
"""
Sandwiching Experiments (Cotra, 2021)
=======================================
A methodology for studying scalable oversight in practice.

The "sandwich" has three layers:
1. NON-EXPERT humans (bottom bread) — can't solve the task alone
2. AI SYSTEM being tested (the filling)
3. EXPERT humans (top bread) — can verify the answer

The experiment measures: Can the AI + non-expert system produce
outputs that experts would approve? This is a tractable proxy
for the real problem (where we ARE the non-experts relative
to superhuman AI).

         Expert (can verify)
        ┌─────────────────┐
        │   GOLD STANDARD │
        └────────┬────────┘
                 │ evaluates
        ┌────────▼────────┐
        │   AI + Non-Expert│  ← sandwiched system
        └────────┬────────┘
                 │ generates
        ┌────────▼────────┐
        │  Task Outputs   │
        └─────────────────┘
"""

from dataclasses import dataclass, field
from typing import List, Optional
import json


@dataclass
class SandwichTask:
    """A task for sandwiching experiments."""
    question: str
    domain: str
    difficulty: str  # "easy", "medium", "hard"
    expert_answer: Optional[str] = None
    non_expert_answer: Optional[str] = None
    ai_assisted_answer: Optional[str] = None
    expert_score_for_non_expert: Optional[float] = None
    expert_score_for_ai_assisted: Optional[float] = None


@dataclass
class SandwichExperiment:
    """Track a sandwiching experiment across multiple tasks."""
    tasks: List[SandwichTask] = field(default_factory=list)
    domain: str = ""

    def add_task(self, task: SandwichTask):
        self.tasks.append(task)

    def compute_metrics(self) -> dict:
        """Compute sandwiching metrics."""
        scored_tasks = [
            t for t in self.tasks
            if t.expert_score_for_non_expert is not None
            and t.expert_score_for_ai_assisted is not None
        ]

        if not scored_tasks:
            return {"error": "No scored tasks"}

        non_expert_scores = [t.expert_score_for_non_expert for t in scored_tasks]
        ai_assisted_scores = [t.expert_score_for_ai_assisted for t in scored_tasks]

        avg_non_expert = sum(non_expert_scores) / len(non_expert_scores)
        avg_ai_assisted = sum(ai_assisted_scores) / len(ai_assisted_scores)
        improvement = avg_ai_assisted - avg_non_expert

        # Win rate: fraction of tasks where AI-assisted beats non-expert
        wins = sum(
            1 for ne, ai in zip(non_expert_scores, ai_assisted_scores) if ai > ne
        )
        ties = sum(
            1 for ne, ai in zip(non_expert_scores, ai_assisted_scores) if ai == ne
        )
        losses = len(scored_tasks) - wins - ties

        return {
            "n_tasks": len(scored_tasks),
            "avg_non_expert_score": avg_non_expert,
            "avg_ai_assisted_score": avg_ai_assisted,
            "improvement": improvement,
            "win_rate": wins / len(scored_tasks),
            "tie_rate": ties / len(scored_tasks),
            "loss_rate": losses / len(scored_tasks),
        }


def run_sandwiching_with_llm(
    tasks: List[dict],
    domain: str = "biology",
) -> SandwichExperiment:
    """
    Run a sandwiching experiment using an LLM as the AI assistant.

    Each task dict has: question, expert_answer, difficulty
    """
    import anthropic

    client = anthropic.Anthropic()
    experiment = SandwichExperiment(domain=domain)

    for task_data in tasks:
        task = SandwichTask(
            question=task_data["question"],
            domain=domain,
            difficulty=task_data.get("difficulty", "medium"),
            expert_answer=task_data.get("expert_answer"),
        )

        # Simulate non-expert: ask model with "pretend you're a non-expert"
        non_expert_response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=(
                "You are simulating a non-expert human. Answer the question "
                "using only general knowledge — no specialized expertise."
            ),
            messages=[{"role": "user", "content": task.question}],
        )
        task.non_expert_answer = non_expert_response.content[0].text

        # AI-assisted answer: full model capability
        ai_response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=(
                f"You are an expert in {domain}. Provide a thorough, accurate "
                f"answer with reasoning."
            ),
            messages=[{"role": "user", "content": task.question}],
        )
        task.ai_assisted_answer = ai_response.content[0].text

        # Expert evaluation (simulated by asking model to compare)
        if task.expert_answer:
            eval_prompt = f"""As a domain expert, score these answers from 0-10:

Question: {task.question}
Expert reference answer: {task.expert_answer}

Answer A (non-expert): {task.non_expert_answer}
Answer B (AI-assisted): {task.ai_assisted_answer}

Reply in JSON: {{"score_a": <int>, "score_b": <int>, "reasoning": "<text>"}}"""

            eval_response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{"role": "user", "content": eval_prompt}],
            )
            try:
                scores = json.loads(eval_response.content[0].text)
                task.expert_score_for_non_expert = scores["score_a"]
                task.expert_score_for_ai_assisted = scores["score_b"]
            except (json.JSONDecodeError, KeyError):
                pass

        experiment.add_task(task)

    metrics = experiment.compute_metrics()
    print(f"\nSandwiching Results for {domain}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return experiment
```

---

## 7. Market-Making for Alignment

```python
"""
Market-Making for Alignment
==============================
Using prediction market mechanisms to elicit truthful AI behavior.

Key insight: Proper scoring rules (like log scoring) incentivize
agents to report their TRUE beliefs. If we can structure AI
evaluation as a prediction market, we get truth-seeking dynamics.

Related concepts:
- Prediction markets for AI forecasting
- Scoring rules (Brier, logarithmic) for calibration
- Peer prediction mechanisms for crowdsourced evaluation
"""

import torch
import numpy as np
from typing import List, Dict


class ProperScoringRule:
    """
    Proper scoring rules incentivize truthful probability reports.

    A scoring rule S(p, y) is PROPER if the expected score is
    maximized when the agent reports their true beliefs.
    """

    @staticmethod
    def brier_score(predicted_prob: float, actual: int) -> float:
        """
        Brier Score: S = -(p - y)^2
        Proper scoring rule. Lower is better (we negate for reward).
        """
        return -(predicted_prob - actual) ** 2

    @staticmethod
    def log_score(predicted_prob: float, actual: int) -> float:
        """
        Logarithmic Score: S = y*log(p) + (1-y)*log(1-p)
        Strictly proper. Heavily penalizes confident wrong predictions.
        """
        eps = 1e-10
        p = max(min(predicted_prob, 1 - eps), eps)
        if actual == 1:
            return np.log(p)
        else:
            return np.log(1 - p)

    @staticmethod
    def spherical_score(predicted_prob: float, actual: int) -> float:
        """
        Spherical Score: S = p_y / ||p||
        Proper scoring rule with bounded range.
        """
        p = max(min(predicted_prob, 0.9999), 0.0001)
        norm = np.sqrt(p ** 2 + (1 - p) ** 2)
        if actual == 1:
            return p / norm
        else:
            return (1 - p) / norm


def demonstrate_proper_scoring():
    """Show that proper scoring rules incentivize truthful reporting."""
    scorer = ProperScoringRule()
    true_prob = 0.7  # Agent's true belief

    print("Scoring rules incentivize truthful reporting:")
    print(f"True belief: p = {true_prob}")
    print()

    for reported in [0.1, 0.3, 0.5, 0.7, 0.9]:
        # Expected score = p_true * S(reported, 1) + (1-p_true) * S(reported, 0)
        expected_brier = (
            true_prob * scorer.brier_score(reported, 1)
            + (1 - true_prob) * scorer.brier_score(reported, 0)
        )
        expected_log = (
            true_prob * scorer.log_score(reported, 1)
            + (1 - true_prob) * scorer.log_score(reported, 0)
        )
        print(
            f"  Report p={reported:.1f}: "
            f"E[Brier]={expected_brier:.4f}, "
            f"E[Log]={expected_log:.4f}"
        )

    print()
    print("Maximum expected score should be at reported = true_prob = 0.7")


class AlignmentMarket:
    """
    A prediction market for evaluating AI outputs.

    Multiple evaluator agents place "bets" on the quality
    of AI outputs. Proper scoring ensures truthful evaluation.
    """

    def __init__(self, n_evaluators: int = 5):
        self.n_evaluators = n_evaluators
        self.scorer = ProperScoringRule()
        self.evaluator_scores: Dict[int, List[float]] = {
            i: [] for i in range(n_evaluators)
        }

    def run_evaluation_round(
        self,
        ai_output_quality: float,
        evaluator_beliefs: List[float],
    ) -> dict:
        """
        Each evaluator reports their belief about output quality.
        After the ground truth is revealed, they are scored.
        """
        actual = int(ai_output_quality > 0.5)

        scores = {}
        for i, belief in enumerate(evaluator_beliefs):
            score = self.scorer.log_score(belief, actual)
            self.evaluator_scores[i].append(score)
            scores[i] = score

        # Aggregate belief (weighted by past performance)
        weights = []
        for i in range(self.n_evaluators):
            past = self.evaluator_scores[i]
            if len(past) > 1:
                w = np.exp(np.mean(past[:-1]))  # performance-weighted
            else:
                w = 1.0
            weights.append(w)

        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        consensus = sum(w * b for w, b in zip(weights, evaluator_beliefs))

        return {
            "individual_scores": scores,
            "consensus_belief": consensus,
            "actual": actual,
            "weights": weights,
        }
```

---

## 8. Decomposition Approaches

```python
"""
Task Decomposition for Scalable Oversight
=============================================
Break complex tasks into smaller sub-tasks that humans can verify.

Key principle: Even if a human cannot evaluate the full output,
they can evaluate individual pieces, and correctness of pieces
implies correctness of the whole (under certain conditions).

Example: "Write a correct and safe medical AI system"
→ Decompose into: requirements, individual functions, test cases,
  safety checks — each verifiable by a human.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Callable, Optional
import anthropic


@dataclass
class SubTask:
    """A decomposed sub-task."""
    id: str
    description: str
    parent_id: Optional[str] = None
    result: Optional[str] = None
    verified: bool = False
    confidence: float = 0.0


@dataclass
class DecompositionTree:
    """A tree of decomposed sub-tasks."""
    root_task: str
    subtasks: List[SubTask] = field(default_factory=list)

    def add_subtask(self, subtask: SubTask):
        self.subtasks.append(subtask)

    def get_children(self, parent_id: str) -> List[SubTask]:
        return [s for s in self.subtasks if s.parent_id == parent_id]

    def all_verified(self) -> bool:
        return all(s.verified for s in self.subtasks)

    def verification_rate(self) -> float:
        if not self.subtasks:
            return 0.0
        return sum(1 for s in self.subtasks if s.verified) / len(self.subtasks)


def decompose_task_with_llm(
    task: str,
    max_depth: int = 2,
    max_subtasks: int = 4,
) -> DecompositionTree:
    """
    Use an LLM to decompose a complex task into verifiable sub-tasks.
    """
    client = anthropic.Anthropic()
    tree = DecompositionTree(root_task=task)

    # Level 1 decomposition
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=(
            "You are a task decomposition expert. Break the given task "
            "into 3-4 independent, verifiable sub-tasks. Each sub-task "
            "should be simple enough for a non-expert to verify. "
            "Return a numbered list, one sub-task per line."
        ),
        messages=[{"role": "user", "content": f"Decompose this task: {task}"}],
    )

    lines = response.content[0].text.strip().split("\n")
    for i, line in enumerate(lines[:max_subtasks]):
        clean_line = line.strip().lstrip("0123456789.-) ")
        if clean_line:
            subtask = SubTask(
                id=f"L1_{i}",
                description=clean_line,
                parent_id="root",
            )
            tree.add_subtask(subtask)

    # Level 2 decomposition (if needed)
    if max_depth >= 2:
        for parent in list(tree.subtasks):
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                system=(
                    "Break this sub-task into 2-3 atomic verification steps. "
                    "Each step should be answerable with yes/no or a simple check."
                ),
                messages=[
                    {"role": "user", "content": f"Sub-task: {parent.description}"}
                ],
            )
            lines = response.content[0].text.strip().split("\n")
            for j, line in enumerate(lines[:3]):
                clean_line = line.strip().lstrip("0123456789.-) ")
                if clean_line:
                    child = SubTask(
                        id=f"{parent.id}_{j}",
                        description=clean_line,
                        parent_id=parent.id,
                    )
                    tree.add_subtask(child)

    return tree


def verify_decomposed_task(
    tree: DecompositionTree,
    verifier_fn: Optional[Callable] = None,
) -> dict:
    """
    Verify each sub-task independently.

    In practice, each sub-task is shown to a human evaluator
    (or an automated verifier) who checks correctness.
    """
    results = []

    for subtask in tree.subtasks:
        if verifier_fn:
            verified, confidence = verifier_fn(subtask)
        else:
            # Default: random verification (placeholder)
            import random
            verified = random.random() > 0.2
            confidence = random.uniform(0.5, 1.0)

        subtask.verified = verified
        subtask.confidence = confidence
        results.append({
            "id": subtask.id,
            "description": subtask.description,
            "verified": verified,
            "confidence": confidence,
        })

    overall_verification = tree.verification_rate()
    overall_confidence = (
        sum(s.confidence for s in tree.subtasks) / len(tree.subtasks)
        if tree.subtasks else 0.0
    )

    return {
        "subtask_results": results,
        "verification_rate": overall_verification,
        "overall_confidence": overall_confidence,
        "all_verified": tree.all_verified(),
    }
```

---

## 9. Current Research Status

```python
"""
Current Research Status in Scalable Oversight (as of 2025)
============================================================

DEBATE:
- Anthropic and others have run debate experiments with LLMs
- Key finding: debate helps on some tasks but judges can be swayed
  by more eloquent (not more truthful) arguments
- Open question: Does debate scale to truly superhuman capabilities?

RECURSIVE REWARD MODELING:
- Theoretical framework is promising but empirically limited
- Challenge: error compounding across recursive levels
- Active area: combining with Constitutional AI approaches

IDA:
- Christiano's original vision remains largely theoretical
- Some progress in applying decomposition to real tasks
- Challenge: not all tasks decompose cleanly

WEAK-TO-STRONG:
- Burns et al. (2023) showed promising PGR on NLP tasks
- Worrying result: PGR drops significantly for reward modeling
- Active research at OpenAI, Anthropic, and academic labs
- Auxiliary confidence loss can improve PGR

SANDWICHING:
- Anthropic has used sandwiching experiments internally
- Results show AI assistance helps non-experts but doesn't
  close the gap to expert performance entirely

OVERALL ASSESSMENT:
- No single method solves scalable oversight
- Most promising: combining multiple methods
- The field is shifting toward empirical work on real models
"""

RESEARCH_SUMMARY = {
    "debate": {
        "maturity": "early_empirical",
        "key_papers": [
            "Irving et al. 2018 — AI Safety via Debate",
            "Khan et al. 2024 — Debating with More Persuasive LLMs",
        ],
        "open_problems": [
            "Eloquence vs truthfulness",
            "Computational cost of multi-round debates",
            "Judge manipulation",
        ],
    },
    "recursive_reward_modeling": {
        "maturity": "theoretical",
        "key_papers": [
            "Leike et al. 2018 — Scalable Agent Alignment via Reward Modeling",
        ],
        "open_problems": [
            "Error compounding",
            "Bootstrapping from limited human feedback",
        ],
    },
    "weak_to_strong": {
        "maturity": "early_empirical",
        "key_papers": [
            "Burns et al. 2023 — Weak-to-Strong Generalization",
        ],
        "open_problems": [
            "Low PGR for reward modeling",
            "Scaling to truly superhuman models",
            "Task-dependence of generalization",
        ],
    },
    "ida": {
        "maturity": "theoretical",
        "key_papers": [
            "Christiano 2018 — Iterated Distillation and Amplification",
        ],
        "open_problems": [
            "Decomposability assumption",
            "Distillation fidelity",
        ],
    },
}
```

### Key Limitations in Depth

**The Expert Liar Problem (Debate)**

Debate assumes truth has a "natural advantage" — an honest debater can always point to verifiable evidence. The game-theoretic argument is that a dishonest debater cannot indefinitely hide false premises from an opponent who knows the truth. However, this guarantee weakens when the human judge is not fully rational or lacks domain expertise. A sufficiently capable dishonest debater might construct elaborate, internally consistent but false arguments that a human judge cannot distinguish from truthful ones. This is called the "expert liar problem": the same rhetorical skills that make a good debater also make a convincing liar. Empirical debate experiments have confirmed this risk — judges are sometimes swayed more by eloquence than by correctness, especially when both debaters are capable LLMs.

**IDA Convergence and Error Accumulation**

Iterated Distillation and Amplification's convergence guarantee depends critically on the decomposability of tasks. The amplification step works well when a hard task can be cleanly split into simpler sub-tasks whose solutions compose back into a correct solution for the original task. Many routine technical tasks (code generation, arithmetic, document summarization) have this property. However, many alignment-relevant tasks do not: moral reasoning, value judgments, and strategic planning resist clean decomposition because their correctness is holistic rather than compositional. A morally questionable plan might pass inspection at every decomposed sub-step while still being problematic as a whole. Additionally, each distillation step introduces approximation error, and these errors compound across iterations. A distilled agent at level k is slightly worse than the amplified agent at level k, which itself builds on the errors of level k-1. If the error per level is not strictly decreasing, the recursion may fail to converge or converge to a misaligned fixed point. Current theoretical bounds on this error accumulation remain loose.

---

## Summary

- **The oversight problem** is the central bottleneck: as AI systems become superhuman, we lose the ability to verify their outputs, making all current alignment methods (RLHF, DPO, CAI) potentially insufficient.
- **AI Safety via Debate** uses competing AI agents to reveal truthful information to a human judge. Game theory predicts honesty as the equilibrium, but empirical results show judges can be swayed by eloquence.
- **Recursive Reward Modeling** trains increasingly capable reward models by using AI-assisted human evaluation at each level. The key risk is error compounding across recursive levels.
- **Iterated Distillation and Amplification (IDA)** alternates between amplifying a weak agent (using human oversight + decomposition) and distilling the amplified system into a fast model. It remains largely theoretical.
- **Weak-to-strong generalization** (Burns et al., 2023) empirically studies how strong models can exceed the accuracy of weak supervisors. Performance Gap Recovered varies from ~80% (NLP) to much lower (reward modeling).
- **Sandwiching experiments** place AI systems between non-expert users and expert evaluators, providing a tractable testbed for scalable oversight methods.
- **Market mechanisms** and proper scoring rules (Brier, log) can incentivize truthful reporting from AI evaluators.
- **Decomposition** breaks complex tasks into smaller, verifiable sub-tasks, enabling human oversight at the component level.
- No single method currently solves scalable oversight; the most promising direction is combining multiple approaches with empirical validation.

---

## Exercises

### Exercise 1: Implementing a Simple Debate Protocol

Implement a two-round debate between two LLM instances (or simulated agents) on a factual question with a known answer. Track whether the judge correctly identifies the truthful debater. Run the experiment on 10 different questions and measure the judge's accuracy. Analyze: when does the judge fail?

<details>
<summary>Show Answer</summary>

```python
import anthropic
from typing import List, Tuple


def debate_experiment(questions_and_answers: List[Tuple[str, str, str]]) -> dict:
    """
    Run debate experiments on questions with known answers.

    Args:
        questions_and_answers: List of (question, correct_answer, wrong_answer)
    """
    client = anthropic.Anthropic()
    results = []

    for question, correct, wrong in questions_and_answers:
        # Randomly assign positions
        import random
        if random.random() > 0.5:
            pos_a, pos_b = correct, wrong
            truthful_debater = "A"
        else:
            pos_a, pos_b = wrong, correct
            truthful_debater = "B"

        transcript = []

        # Round 1: Opening arguments
        for debater, position in [("A", pos_a), ("B", pos_b)]:
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                system=f"Argue that the answer is: {position}. Be concise.",
                messages=[{"role": "user", "content": question}],
            )
            transcript.append(f"Debater {debater}: {resp.content[0].text}")

        # Round 2: Rebuttals
        for debater, position in [("A", pos_a), ("B", pos_b)]:
            opponent_args = [t for t in transcript if not t.startswith(f"Debater {debater}")]
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                system=f"Argue for: {position}. Counter: {opponent_args[-1]}",
                messages=[{"role": "user", "content": question}],
            )
            transcript.append(f"Debater {debater} (rebuttal): {resp.content[0].text}")

        # Judge
        judge_prompt = f"Question: {question}\n\n" + "\n\n".join(transcript)
        judge_prompt += "\n\nWhich debater is more truthful? Reply with just 'A' or 'B'."

        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        judge_pick = resp.content[0].text.strip().upper()
        correct_judgment = judge_pick == truthful_debater

        results.append({
            "question": question,
            "truthful": truthful_debater,
            "judge_pick": judge_pick,
            "correct": correct_judgment,
        })
        print(f"Q: {question[:50]}... | Judge: {judge_pick} | "
              f"Truth: {truthful_debater} | {'CORRECT' if correct_judgment else 'WRONG'}")

    accuracy = sum(r["correct"] for r in results) / len(results)
    print(f"\nOverall judge accuracy: {accuracy:.1%}")
    return {"results": results, "accuracy": accuracy}


# Example usage:
sample_questions = [
    ("What is the capital of Australia?", "Canberra", "Sydney"),
    ("Who wrote 'Pride and Prejudice'?", "Jane Austen", "Charlotte Bronte"),
    ("What is the speed of light?", "~300,000 km/s", "~150,000 km/s"),
    ("How many chromosomes do humans have?", "46", "48"),
    ("What year did World War I start?", "1914", "1939"),
]

# results = debate_experiment(sample_questions)
```

</details>

### Exercise 2: Weak-to-Strong with Real Models

Using HuggingFace transformers, implement a weak-to-strong experiment. Use `distilbert-base-uncased` as the weak model and `bert-base-uncased` as the strong model. Train on the SST-2 sentiment classification task. Compute PGR. Then try an auxiliary confidence loss (upweight high-confidence weak labels) and measure if PGR improves.

<details>
<summary>Show Answer</summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
)
from datasets import load_dataset


def weak_to_strong_nlp(
    weak_name: str = "distilbert-base-uncased",
    strong_name: str = "bert-base-uncased",
    n_train: int = 5000,
    n_test: int = 1000,
    n_epochs: int = 3,
    batch_size: int = 32,
    use_confidence_loss: bool = False,
):
    """Full weak-to-strong experiment on SST-2."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset("glue", "sst2")

    # Prepare data
    weak_tokenizer = AutoTokenizer.from_pretrained(weak_name)
    strong_tokenizer = AutoTokenizer.from_pretrained(strong_name)

    train_texts = dataset["train"]["sentence"][:n_train]
    train_labels = dataset["train"]["label"][:n_train]
    test_texts = dataset["validation"]["sentence"][:n_test]
    test_labels = dataset["validation"]["label"][:n_test]

    # Step 1: Train weak model on ground truth
    weak_model = AutoModelForSequenceClassification.from_pretrained(
        weak_name, num_labels=2
    ).to(device)
    weak_optimizer = torch.optim.AdamW(weak_model.parameters(), lr=2e-5)

    for epoch in range(n_epochs):
        weak_model.train()
        for i in range(0, len(train_texts), batch_size):
            batch_texts = train_texts[i:i + batch_size]
            batch_labels = torch.tensor(train_labels[i:i + batch_size]).to(device)
            inputs = weak_tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=128
            ).to(device)
            outputs = weak_model(**inputs, labels=batch_labels)
            outputs.loss.backward()
            weak_optimizer.step()
            weak_optimizer.zero_grad()

    # Evaluate weak ceiling
    weak_model.eval()
    weak_correct = 0
    with torch.no_grad():
        for i in range(0, len(test_texts), batch_size):
            inputs = weak_tokenizer(
                test_texts[i:i + batch_size], return_tensors="pt",
                padding=True, truncation=True, max_length=128
            ).to(device)
            preds = weak_model(**inputs).logits.argmax(-1)
            labels = torch.tensor(test_labels[i:i + batch_size]).to(device)
            weak_correct += (preds == labels).sum().item()
    weak_ceiling = weak_correct / len(test_texts)
    print(f"Weak ceiling: {weak_ceiling:.3f}")

    # Step 2: Generate weak labels with confidence
    weak_labels = []
    weak_confidences = []
    with torch.no_grad():
        for i in range(0, len(train_texts), batch_size):
            inputs = weak_tokenizer(
                train_texts[i:i + batch_size], return_tensors="pt",
                padding=True, truncation=True, max_length=128
            ).to(device)
            logits = weak_model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
            weak_labels.extend(logits.argmax(-1).cpu().tolist())
            weak_confidences.extend(probs.max(-1).values.cpu().tolist())

    # Step 3: Train strong model on weak labels
    strong_model = AutoModelForSequenceClassification.from_pretrained(
        strong_name, num_labels=2
    ).to(device)
    strong_optimizer = torch.optim.AdamW(strong_model.parameters(), lr=2e-5)

    for epoch in range(n_epochs):
        strong_model.train()
        for i in range(0, len(train_texts), batch_size):
            batch_texts = train_texts[i:i + batch_size]
            batch_wl = torch.tensor(weak_labels[i:i + batch_size]).to(device)
            inputs = strong_tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=128
            ).to(device)
            outputs = strong_model(**inputs, labels=batch_wl)
            loss = outputs.loss

            if use_confidence_loss:
                # Upweight high-confidence weak labels
                conf = torch.tensor(
                    weak_confidences[i:i + batch_size]
                ).to(device)
                logits = outputs.logits
                per_sample = F.cross_entropy(logits, batch_wl, reduction="none")
                loss = (per_sample * conf).mean()

            loss.backward()
            strong_optimizer.step()
            strong_optimizer.zero_grad()

    # Evaluate weak-to-strong
    strong_model.eval()
    w2s_correct = 0
    with torch.no_grad():
        for i in range(0, len(test_texts), batch_size):
            inputs = strong_tokenizer(
                test_texts[i:i + batch_size], return_tensors="pt",
                padding=True, truncation=True, max_length=128
            ).to(device)
            preds = strong_model(**inputs).logits.argmax(-1)
            labels = torch.tensor(test_labels[i:i + batch_size]).to(device)
            w2s_correct += (preds == labels).sum().item()
    w2s_acc = w2s_correct / len(test_texts)

    # Step 4: Strong ceiling (train on ground truth)
    strong_ceil_model = AutoModelForSequenceClassification.from_pretrained(
        strong_name, num_labels=2
    ).to(device)
    ceil_optimizer = torch.optim.AdamW(strong_ceil_model.parameters(), lr=2e-5)
    for epoch in range(n_epochs):
        strong_ceil_model.train()
        for i in range(0, len(train_texts), batch_size):
            batch_texts = train_texts[i:i + batch_size]
            batch_labels = torch.tensor(train_labels[i:i + batch_size]).to(device)
            inputs = strong_tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=128
            ).to(device)
            outputs = strong_ceil_model(**inputs, labels=batch_labels)
            outputs.loss.backward()
            ceil_optimizer.step()
            ceil_optimizer.zero_grad()

    strong_ceil_model.eval()
    ceil_correct = 0
    with torch.no_grad():
        for i in range(0, len(test_texts), batch_size):
            inputs = strong_tokenizer(
                test_texts[i:i + batch_size], return_tensors="pt",
                padding=True, truncation=True, max_length=128
            ).to(device)
            preds = strong_ceil_model(**inputs).logits.argmax(-1)
            labels = torch.tensor(test_labels[i:i + batch_size]).to(device)
            ceil_correct += (preds == labels).sum().item()
    strong_ceiling = ceil_correct / len(test_texts)

    pgr = (w2s_acc - weak_ceiling) / max(strong_ceiling - weak_ceiling, 0.001)
    print(f"Weak ceiling: {weak_ceiling:.3f}")
    print(f"W2S accuracy: {w2s_acc:.3f}")
    print(f"Strong ceiling: {strong_ceiling:.3f}")
    print(f"PGR: {pgr:.3f}")
    return {"weak": weak_ceiling, "w2s": w2s_acc, "strong": strong_ceiling, "pgr": pgr}
```

</details>

### Exercise 3: Building a Task Decomposition Pipeline

Implement a task decomposition pipeline that takes a complex question (e.g., "Is this code secure?"), breaks it into sub-questions, answers each independently with an LLM, and aggregates results. Compare the accuracy of the decomposed approach against a single direct-answer call on 5 complex questions. Which approach is more reliable?

<details>
<summary>Show Answer</summary>

```python
import anthropic
import json
from typing import List


def decompose_and_solve(question: str, client: anthropic.Anthropic) -> dict:
    """Decompose a question, solve sub-parts, aggregate."""

    # Step 1: Decompose
    decomp_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=(
            "Break this question into 3-4 independent sub-questions "
            "that together answer the original. Return a JSON list of strings."
        ),
        messages=[{"role": "user", "content": question}],
    )
    try:
        sub_questions = json.loads(decomp_response.content[0].text)
    except json.JSONDecodeError:
        lines = decomp_response.content[0].text.strip().split("\n")
        sub_questions = [l.strip().lstrip("0123456789.-) ") for l in lines if l.strip()]

    # Step 2: Solve each sub-question
    sub_answers = []
    for sq in sub_questions:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": sq}],
        )
        sub_answers.append({"question": sq, "answer": resp.content[0].text})

    # Step 3: Aggregate
    aggregation_prompt = f"Original question: {question}\n\nSub-answers:\n"
    for sa in sub_answers:
        aggregation_prompt += f"\nQ: {sa['question']}\nA: {sa['answer']}\n"
    aggregation_prompt += "\nSynthesize a final answer to the original question."

    final_resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": aggregation_prompt}],
    )

    return {
        "question": question,
        "sub_questions": sub_questions,
        "sub_answers": sub_answers,
        "final_answer": final_resp.content[0].text,
    }


def direct_solve(question: str, client: anthropic.Anthropic) -> str:
    """Directly answer without decomposition."""
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": question}],
    )
    return resp.content[0].text


def compare_approaches(questions: List[str]):
    """Compare decomposed vs direct approaches."""
    client = anthropic.Anthropic()

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        print(f"{'='*60}")

        direct = direct_solve(q, client)
        print(f"\nDirect answer:\n{direct[:200]}...")

        decomposed = decompose_and_solve(q, client)
        print(f"\nDecomposed into {len(decomposed['sub_questions'])} sub-questions:")
        for sq in decomposed["sub_questions"]:
            print(f"  - {sq}")
        print(f"\nFinal decomposed answer:\n{decomposed['final_answer'][:200]}...")


# Example questions
# compare_approaches([
#     "Is the Python `eval()` function safe to use with user input?",
#     "Should a hospital use an LLM for diagnostic suggestions?",
#     "Is training AI on copyrighted data ethical?",
# ])
```

</details>

### Exercise 4: Recursive Reward Model Training

Implement a 3-level recursive reward model training pipeline. At each level, measure: (a) correlation with ground truth, (b) evaluation noise, (c) training time. Plot how these metrics evolve across levels. Experiment with different trust weights (how much the human defers to the AI assistant) and show the impact on final model quality.

<details>
<summary>Show Answer</summary>

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from typing import List


class RewardModelLevel(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_recursive_levels(
    n_levels: int = 3,
    trust_weight: float = 0.3,
    n_samples: int = 3000,
    input_dim: int = 128,
    n_epochs: int = 50,
):
    """Train recursive reward models and track metrics."""
    # Ground truth: linear function of inputs
    true_weights = torch.randn(input_dim)
    true_weights /= true_weights.norm()

    data = torch.randn(n_samples, input_dim)
    ground_truth = (data @ true_weights).unsqueeze(1)

    models = []
    metrics = {"correlation": [], "noise": [], "trust_weight": trust_weight}

    for level in range(n_levels):
        model = RewardModelLevel(input_dim)
        difficulty = 1.0 + level * 0.5

        # Generate labels
        human_noise = torch.randn(n_samples, 1) * 0.2 * difficulty
        human_eval = ground_truth + human_noise

        if level > 0:
            with torch.no_grad():
                ai_suggestion = models[-1](data)
            labels = (1 - trust_weight) * human_eval + trust_weight * ai_suggestion
        else:
            labels = human_eval

        # Measure label noise
        label_noise = ((labels - ground_truth) ** 2).mean().sqrt().item()
        metrics["noise"].append(label_noise)

        # Train
        dataset = TensorDataset(data, labels)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(n_epochs):
            for bx, by in loader:
                pred = model(bx)
                loss = nn.MSELoss()(pred, by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate correlation with ground truth
        with torch.no_grad():
            predicted = model(data)
            corr = torch.corrcoef(
                torch.cat([ground_truth, predicted], dim=1).T
            )[0, 1].item()
        metrics["correlation"].append(corr)
        models.append(model)

        print(f"Level {level}: correlation={corr:.3f}, noise={label_noise:.3f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    levels = list(range(n_levels))

    ax1.plot(levels, metrics["correlation"], "bo-", linewidth=2)
    ax1.set_xlabel("Recursive Level")
    ax1.set_ylabel("Correlation with Ground Truth")
    ax1.set_title(f"Reward Model Quality (trust={trust_weight})")

    ax2.plot(levels, metrics["noise"], "ro-", linewidth=2)
    ax2.set_xlabel("Recursive Level")
    ax2.set_ylabel("Label Noise (RMSE)")
    ax2.set_title("Evaluation Noise per Level")

    plt.tight_layout()
    plt.savefig("recursive_reward_metrics.png", dpi=150)
    plt.show()

    return metrics


# Compare different trust weights
# for tw in [0.1, 0.3, 0.5, 0.7]:
#     print(f"\n=== Trust weight: {tw} ===")
#     train_recursive_levels(trust_weight=tw)
```

</details>

### Exercise 5: End-to-End Scalable Oversight Pipeline

Build a complete scalable oversight pipeline that combines debate + decomposition + weak-to-strong generalization. Given a complex question: (1) decompose it, (2) for each sub-question run a mini-debate, (3) use a weak model to generate initial labels and a strong model to improve them. Evaluate the pipeline on 5 questions with known answers. Compare against a single-call baseline.

<details>
<summary>Show Answer</summary>

```python
import torch
import torch.nn as nn
import anthropic
import json
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class OversightResult:
    question: str
    decomposed_questions: List[str]
    debate_winners: List[str]
    weak_answer: str
    strong_answer: str
    final_answer: str
    correct: bool = False


def scalable_oversight_pipeline(
    question: str,
    ground_truth: str,
) -> OversightResult:
    """
    Combined pipeline: decompose → debate → aggregate.
    """
    client = anthropic.Anthropic()

    # Step 1: Decompose
    decomp_resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        system="Break into 3 verifiable sub-questions. Return JSON list.",
        messages=[{"role": "user", "content": question}],
    )
    try:
        sub_qs = json.loads(decomp_resp.content[0].text)
    except json.JSONDecodeError:
        sub_qs = [l.strip() for l in decomp_resp.content[0].text.split("\n") if l.strip()]
    sub_qs = sub_qs[:3]

    # Step 2: Mini-debate on each sub-question
    debate_results = []
    for sq in sub_qs:
        # Debater A: argues "yes/affirmative"
        a_resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            system="Argue the affirmative position. Be truthful.",
            messages=[{"role": "user", "content": sq}],
        )
        # Debater B: argues "no/negative"
        b_resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            system="Argue the negative position. Be truthful.",
            messages=[{"role": "user", "content": sq}],
        )
        # Judge
        judge_prompt = (
            f"Sub-question: {sq}\n\n"
            f"Argument A: {a_resp.content[0].text}\n\n"
            f"Argument B: {b_resp.content[0].text}\n\n"
            f"Which is more truthful? Reply 'A' or 'B' and the correct answer."
        )
        judge_resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        debate_results.append(judge_resp.content[0].text)

    # Step 3: Weak answer (simulated limited model)
    weak_resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        system="Answer briefly. You have limited knowledge.",
        messages=[{"role": "user", "content": question}],
    )
    weak_answer = weak_resp.content[0].text

    # Step 4: Strong answer using debate results
    strong_prompt = (
        f"Question: {question}\n\n"
        f"Sub-question analyses:\n"
    )
    for sq, dr in zip(sub_qs, debate_results):
        strong_prompt += f"\n- {sq}\n  Result: {dr}\n"
    strong_prompt += f"\nInitial answer: {weak_answer}\n"
    strong_prompt += "\nProvide the best final answer, correcting any errors."

    strong_resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        messages=[{"role": "user", "content": strong_prompt}],
    )

    # Evaluate correctness
    eval_resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=50,
        messages=[{"role": "user", "content": (
            f"Is this answer correct?\n"
            f"Question: {question}\n"
            f"Answer: {strong_resp.content[0].text}\n"
            f"Ground truth: {ground_truth}\n"
            f"Reply 'YES' or 'NO'."
        )}],
    )
    correct = "YES" in eval_resp.content[0].text.upper()

    return OversightResult(
        question=question,
        decomposed_questions=sub_qs,
        debate_winners=debate_results,
        weak_answer=weak_answer,
        strong_answer=strong_resp.content[0].text,
        final_answer=strong_resp.content[0].text,
        correct=correct,
    )


def evaluate_pipeline(qa_pairs: List[Tuple[str, str]]):
    """Evaluate the full pipeline on multiple questions."""
    results = []
    for q, a in qa_pairs:
        print(f"\nProcessing: {q[:50]}...")
        result = scalable_oversight_pipeline(q, a)
        results.append(result)
        print(f"  Correct: {result.correct}")
        print(f"  Sub-questions: {len(result.decomposed_questions)}")

    accuracy = sum(r.correct for r in results) / len(results)
    print(f"\nOverall pipeline accuracy: {accuracy:.1%}")
    return results


# Example:
# evaluate_pipeline([
#     ("What causes ocean acidification and what are its effects?",
#      "CO2 absorption lowers pH, harming coral and shellfish"),
#     ("How does mRNA vaccine technology work?",
#      "mRNA instructs cells to produce spike protein, triggering immune response"),
# ])
```

</details>

---

[Previous: Direct Preference Optimization](./05_Direct_Preference_Optimization.md) | [Overview](./00_Overview.md) | [Next: Red Teaming](./07_Red_Teaming.md)

---

**License**: CC BY-NC 4.0

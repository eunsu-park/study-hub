# 25. Research Frontiers

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain test-time compute scaling and describe how o1-style reasoning improves performance on complex problems by generating extended chain-of-thought reasoning
2. Describe the World Models paradigm and explain how models like Sora learn latent physical simulations from video data
3. Analyze the role of synthetic data generation in scaling foundation model training beyond internet-scale corpora
4. Compare multi-agent system architectures and describe how specialized agents can collaborate to solve tasks beyond single-model capabilities
5. Evaluate emerging research directions in foundation models and assess their potential impact on AI capabilities over the next few years

---

## Overview

This lesson explores the cutting edge of Foundation Model research. We investigate future directions including World Models, o1-style Reasoning, Synthetic Data, and Multi-Agent systems.

---

## 1. o1-style Reasoning (Test-time Compute)

### 1.1 Concept

```
Traditional LLM vs o1-style:
┌─────────────────────────────────────────────────────────┐
│  Traditional LLM:                                       │
│  - Focus computation at training time                   │
│    (larger models, more data)                          │
│  - Fixed forward pass during inference                  │
│  - Limitations on complex problems                      │
│                                                         │
│  o1-style (Test-time Compute Scaling):                 │
│  - Use more computation during inference                │
│  - Automatic Chain-of-Thought generation               │
│  - Explore multiple paths, select best                  │
│  - Adaptive computation based on problem difficulty    │
└─────────────────────────────────────────────────────────┘

Key Techniques:
1. Internal Chain-of-Thought
2. Search/Verification loops
3. Self-consistency checking
4. Reward model guided search
```

### 1.2 Conceptual Implementation

```python
import torch
from typing import List, Tuple

class ReasoningModel:
    """o1-style reasoning model (conceptual implementation)"""

    def __init__(self, base_model, reward_model):
        self.model = base_model
        self.reward_model = reward_model

    def reason(
        self,
        problem: str,
        max_thinking_tokens: int = 10000,
        num_candidates: int = 5
    ) -> str:
        """Extended reasoning"""
        # 1. Generate multiple reasoning chains
        candidates = self._generate_candidates(problem, num_candidates)

        # 2. Evaluate each chain
        scored_candidates = []
        for chain, answer in candidates:
            score = self._evaluate_chain(chain, answer)
            scored_candidates.append((chain, answer, score))

        # 3. Select best answer
        best = max(scored_candidates, key=lambda x: x[2])
        return best[1]  # Return only answer (chain is internal)

    def _generate_candidates(
        self,
        problem: str,
        n: int
    ) -> List[Tuple[str, str]]:
        """Generate multiple reasoning paths"""
        candidates = []

        for _ in range(n):
            # Generate step-by-step reasoning
            chain = self._generate_reasoning_chain(problem)

            # Extract final answer from chain
            answer = self._extract_answer(chain)

            candidates.append((chain, answer))

        return candidates

    def _generate_reasoning_chain(self, problem: str) -> str:
        """Generate reasoning chain"""
        prompt = f"""Solve this problem step by step.
Think carefully and show your reasoning.

Problem: {problem}

Let me think through this carefully..."""

        # Generate without length limit (or very long limit)
        response = self.model.generate(
            prompt,
            max_new_tokens=5000,
            temperature=0.7
        )

        return response

    def _evaluate_chain(self, chain: str, answer: str) -> float:
        """Evaluate reasoning chain quality"""
        # Evaluate with reward model
        score = self.reward_model.evaluate(chain)

        # Self-consistency check
        consistency_score = self._check_consistency(chain, answer)

        return score * 0.7 + consistency_score * 0.3

    def _check_consistency(self, chain: str, answer: str) -> float:
        """Check logical consistency"""
        # Simple heuristic or separate model
        prompt = f"""Is this reasoning chain logically consistent?

Reasoning:
{chain}

Answer: {answer}

Rate consistency (0-1):"""

        response = self.model.generate(prompt, max_new_tokens=10)
        # Parse...
        return 0.8  # Example


class TreeOfThoughts:
    """Tree of Thoughts implementation"""

    def __init__(self, model, evaluator):
        self.model = model
        self.evaluator = evaluator

    def solve(
        self,
        problem: str,
        depth: int = 3,
        branching_factor: int = 3
    ) -> str:
        """Solve with tree search"""
        root = {"state": problem, "thoughts": [], "score": 0}
        best_path = self._search(root, depth, branching_factor)
        return self._extract_solution(best_path)

    def _search(self, node: dict, depth: int, bf: int) -> List[dict]:
        """BFS/DFS search"""
        if depth == 0:
            return [node]

        # Generate next step thoughts
        thoughts = self._generate_thoughts(node, bf)

        # Evaluate each thought
        children = []
        for thought in thoughts:
            child = {
                "state": node["state"],
                "thoughts": node["thoughts"] + [thought],
                "score": self._evaluate_thought(thought, node)
            }
            children.append(child)

        # Expand only top b (beam search)
        children.sort(key=lambda x: x["score"], reverse=True)
        children = children[:bf]

        # Recursive search
        best_paths = []
        for child in children:
            path = self._search(child, depth - 1, bf)
            best_paths.extend(path)

        return sorted(best_paths, key=lambda x: x["score"], reverse=True)[:1]

    def _generate_thoughts(self, node: dict, n: int) -> List[str]:
        """Generate next step thoughts"""
        context = "\n".join(node["thoughts"])

        prompt = f"""Problem: {node["state"]}

Previous thoughts:
{context}

Generate {n} different next steps or approaches:"""

        response = self.model.generate(prompt)
        # Parse to extract n thoughts
        return response.split("\n")[:n]

    def _evaluate_thought(self, thought: str, node: dict) -> float:
        """Evaluate thought quality"""
        return self.evaluator.score(thought, node["state"])
```

---

## 2. Synthetic Data

### 2.1 Concept

```
Synthetic Data Generation:
┌─────────────────────────────────────────────────────────┐
│  Problem: Shortage of high-quality training data        │
│                                                         │
│  Solution: Generate training data with LLMs             │
│                                                         │
│  Methods:                                               │
│  1. Self-Instruct: Generate instruction/response pairs  │
│  2. Evol-Instruct: Progressive complexity increase      │
│  3. Rejection Sampling: Generate many, filter best      │
│  4. RLHF-style: Generate preference data               │
│  5. Distillation: Strong model to weak model           │
│                                                         │
│  Cautions:                                              │
│  - Model collapse (training only on own data)           │
│  - Maintain diversity                                   │
│  - Quality verification essential                       │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Implementation

```python
class SyntheticDataGenerator:
    """Synthetic data generator"""

    def __init__(self, teacher_model, student_model=None):
        self.teacher = teacher_model
        self.student = student_model

    def generate_instruction_data(
        self,
        seed_instructions: List[str],
        num_samples: int = 10000,
        diversity_threshold: float = 0.7
    ) -> List[dict]:
        """Generate Instruction-Response data"""
        generated = []
        instruction_embeddings = []

        while len(generated) < num_samples:
            # Generate new instruction
            instruction = self._generate_instruction(seed_instructions + [
                g["instruction"] for g in generated[-10:]
            ])

            # Check diversity
            if self._check_diversity(instruction, instruction_embeddings, diversity_threshold):
                # Generate response
                response = self._generate_response(instruction)

                # Quality check
                if self._quality_check(instruction, response):
                    generated.append({
                        "instruction": instruction,
                        "response": response
                    })

                    # Save embedding
                    emb = self._get_embedding(instruction)
                    instruction_embeddings.append(emb)

            if len(generated) % 100 == 0:
                print(f"Generated {len(generated)}/{num_samples}")

        return generated

    def _generate_instruction(self, examples: List[str]) -> str:
        """Generate new instruction"""
        examples_text = "\n".join([f"- {ex}" for ex in examples[-5:]])

        prompt = f"""Here are some example instructions:
{examples_text}

Generate a new, different instruction that is:
1. Clear and specific
2. Different from the examples
3. Useful and educational

New instruction:"""

        return self.teacher.generate(prompt, temperature=0.9)

    def _generate_response(self, instruction: str) -> str:
        """Generate response"""
        prompt = f"""Instruction: {instruction}

Please provide a helpful, accurate, and detailed response:"""

        return self.teacher.generate(prompt, temperature=0.7)

    def _check_diversity(
        self,
        instruction: str,
        existing_embeddings: List,
        threshold: float
    ) -> bool:
        """Diversity check"""
        if not existing_embeddings:
            return True

        new_emb = self._get_embedding(instruction)

        for emb in existing_embeddings:
            similarity = self._cosine_similarity(new_emb, emb)
            if similarity > threshold:
                return False

        return True

    def _quality_check(self, instruction: str, response: str) -> bool:
        """Quality check"""
        # Length check
        if len(response) < 50:
            return False

        # Relevance check (simple heuristic)
        instruction_words = set(instruction.lower().split())
        response_words = set(response.lower().split())

        overlap = len(instruction_words & response_words)
        if overlap < 2:
            return False

        return True


class RejectSampling:
    """Select high-quality data with Rejection Sampling"""

    def __init__(self, generator_model, reward_model):
        self.generator = generator_model
        self.reward = reward_model

    def generate_with_rejection(
        self,
        prompt: str,
        n_samples: int = 16,
        top_k: int = 1
    ) -> List[str]:
        """Generate many, select best"""
        # Generate multiple responses
        responses = []
        for _ in range(n_samples):
            response = self.generator.generate(prompt, temperature=0.8)
            responses.append(response)

        # Score each response
        scored = []
        for response in responses:
            score = self.reward.score(prompt, response)
            scored.append((response, score))

        # Select top k
        scored.sort(key=lambda x: x[1], reverse=True)
        return [r for r, s in scored[:top_k]]
```

---

## 3. Multi-Agent Systems

### 3.1 Concept

```
Multi-Agent LLM Systems:
┌─────────────────────────────────────────────────────────┐
│  Agent Types:                                           │
│                                                         │
│  1. Debate: Multiple agents discuss                     │
│     - Present different perspectives                    │
│     - Reach consensus                                   │
│                                                         │
│  2. Collaboration: Role-based cooperation               │
│     - Writer, Reviewer, Editor                          │
│     - Researcher, Developer, Tester                     │
│                                                         │
│  3. Competition: Competitive generation                 │
│     - Select best result                                │
│     - Red team / Blue team                              │
│                                                         │
│  4. Hierarchical: Hierarchical structure                │
│     - Manager → Worker agents                           │
│     - Task decomposition and delegation                 │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Implementation

```python
from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum

class AgentRole(Enum):
    PLANNER = "planner"
    RESEARCHER = "researcher"
    WRITER = "writer"
    CRITIC = "critic"
    EDITOR = "editor"

@dataclass
class Message:
    sender: str
    receiver: str
    content: str
    metadata: Dict[str, Any] = None

class MultiAgentSystem:
    """Multi-agent system"""

    def __init__(self, llm):
        self.llm = llm
        self.agents = {}
        self.message_history = []

    def add_agent(self, name: str, role: AgentRole, system_prompt: str):
        """Add agent"""
        self.agents[name] = {
            "role": role,
            "system_prompt": system_prompt,
            "memory": []
        }

    def send_message(self, sender: str, receiver: str, content: str):
        """Send message"""
        message = Message(sender=sender, receiver=receiver, content=content)
        self.message_history.append(message)
        self.agents[receiver]["memory"].append(message)

        return self._get_response(receiver)

    def _get_response(self, agent_name: str) -> str:
        """Generate agent response"""
        agent = self.agents[agent_name]

        # Compose context from recent messages
        recent_messages = agent["memory"][-5:]
        context = "\n".join([
            f"{m.sender}: {m.content}" for m in recent_messages
        ])

        prompt = f"""{agent["system_prompt"]}

Recent conversation:
{context}

Your response as {agent_name}:"""

        return self.llm.generate(prompt)

    def run_debate(
        self,
        topic: str,
        agents: List[str],
        rounds: int = 3
    ) -> str:
        """Run debate"""
        # Initial opinions
        opinions = {}
        for agent in agents:
            response = self.send_message(
                "moderator", agent,
                f"What is your position on: {topic}"
            )
            opinions[agent] = response

        # Debate rounds
        for round in range(rounds):
            for agent in agents:
                # Share other agents' opinions
                other_opinions = "\n".join([
                    f"{a}: {o}" for a, o in opinions.items() if a != agent
                ])

                response = self.send_message(
                    "moderator", agent,
                    f"Others' opinions:\n{other_opinions}\n\nYour response:"
                )
                opinions[agent] = response

        # Reach consensus
        final_opinions = "\n".join([f"{a}: {o}" for a, o in opinions.items()])
        consensus = self.llm.generate(
            f"Based on this debate, summarize the consensus:\n{final_opinions}"
        )

        return consensus


class CollaborativeWriting:
    """Collaborative writing system"""

    def __init__(self, llm):
        self.system = MultiAgentSystem(llm)

        # Setup agents
        self.system.add_agent(
            "writer",
            AgentRole.WRITER,
            "You are a creative writer. Write engaging content."
        )
        self.system.add_agent(
            "critic",
            AgentRole.CRITIC,
            "You are a critical reviewer. Point out issues and suggest improvements."
        )
        self.system.add_agent(
            "editor",
            AgentRole.EDITOR,
            "You are an editor. Refine and polish the writing."
        )

    def write(self, topic: str, iterations: int = 3) -> str:
        """Collaborative writing"""
        # Draft
        draft = self.system.send_message(
            "user", "writer",
            f"Write a short article about: {topic}"
        )

        for i in range(iterations):
            # Critique
            critique = self.system.send_message(
                "writer", "critic",
                f"Please review this draft:\n{draft}"
            )

            # Revise
            revised = self.system.send_message(
                "critic", "writer",
                f"Based on this feedback:\n{critique}\n\nPlease revise the draft."
            )

            draft = revised

        # Final editing
        final = self.system.send_message(
            "writer", "editor",
            f"Please polish this final draft:\n{draft}"
        )

        return final
```

---

## 4. World Models

### 4.1 Concept

```
World Models:
┌─────────────────────────────────────────────────────────┐
│  Goal: LLMs understand and simulate how the world works │
│                                                         │
│  Applications:                                          │
│  1. Planning: Predict consequences of actions           │
│  2. Reasoning: Causal relationship inference            │
│  3. Simulation: Virtual environment simulation          │
│  4. Embodied AI: Robot control                          │
│                                                         │
│  Research Directions:                                   │
│  - Video generation as world simulation (Sora)          │
│  - Physical reasoning benchmarks                        │
│  - Embodied language models                             │
│  - Causal reasoning                                     │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Conceptual Implementation

```python
class WorldModel:
    """World Model conceptual implementation"""

    def __init__(self, llm):
        self.llm = llm
        self.state = {}

    def initialize_state(self, description: str):
        """Set initial state"""
        prompt = f"""Parse this scene description into structured state.

Description: {description}

Extract:
- Objects (name, position, properties)
- Relationships between objects
- Physical constraints

State:"""

        state_text = self.llm.generate(prompt)
        self.state = self._parse_state(state_text)

    def predict_action_result(self, action: str) -> Dict:
        """Predict action result"""
        state_description = self._describe_state()

        prompt = f"""Current state:
{state_description}

Action: {action}

Predict:
1. What changes will occur?
2. What is the new state?
3. Any unexpected effects?

Prediction:"""

        prediction = self.llm.generate(prompt)
        return self._parse_prediction(prediction)

    def simulate_sequence(
        self,
        actions: List[str]
    ) -> List[Dict]:
        """Simulate action sequence"""
        states = [self.state.copy()]

        for action in actions:
            prediction = self.predict_action_result(action)
            self._apply_changes(prediction)
            states.append(self.state.copy())

        return states

    def _describe_state(self) -> str:
        """Describe state as text"""
        # Convert state dict to natural language
        return str(self.state)

    def _parse_state(self, text: str) -> Dict:
        """Parse text to state"""
        # More sophisticated parsing needed in practice
        return {"raw": text}

    def _parse_prediction(self, text: str) -> Dict:
        """Parse prediction result"""
        return {"raw": text}

    def _apply_changes(self, prediction: Dict):
        """Apply predicted changes"""
        # Update state
        pass
```

---

## 5. Future Research Directions

### 5.1 Key Directions

```
1. Scaling Laws Beyond Parameters
   - Test-time compute scaling
   - Mixture of Experts scaling
   - Data quality over quantity

2. Multimodal Understanding
   - Native multimodal models
   - Embodied AI
   - Physical world understanding

3. Reasoning Enhancement
   - Formal verification
   - Neuro-symbolic integration
   - Causal reasoning

4. Alignment & Safety
   - Constitutional AI
   - Interpretability
   - Robustness to adversarial inputs

5. Efficiency
   - Sparse architectures
   - Mixture of Depths
   - Early exit mechanisms
```

### 5.2 Open Problems

```
┌─────────────────────────────────────────────────────────┐
│  Open Research Problems:                                │
│                                                         │
│  1. Complete Hallucination Resolution                   │
│     - Knowing when you don't know                       │
│     - Confidence calibration                            │
│                                                         │
│  2. True Reasoning vs Pattern Matching                  │
│     - True generalization ability?                      │
│     - Out-of-distribution reasoning                     │
│                                                         │
│  3. Long-term Memory                                    │
│     - Permanent learning                                │
│     - Continual learning without forgetting             │
│                                                         │
│  4. Efficiency-Capability Tradeoff                      │
│     - Limitations of small models?                      │
│     - Knowledge distillation limits                     │
│                                                         │
│  5. Alignment                                           │
│     - Definition of value alignment                     │
│     - Scalable oversight                                │
└─────────────────────────────────────────────────────────┘
```

---

## Key Summary

### Research Frontiers Summary
```
1. o1-style: More computation at inference time
2. Synthetic Data: Generate training data with LLMs
3. Multi-Agent: Collaboration/debate/competition systems
4. World Models: Physical world simulation
5. Alignment: Safe and useful AI
```

### Future Outlook
```
- Parameter scaling → Compute scaling
- Single model → Multi-agent systems
- Text → Native multimodal
- Pattern matching → True reasoning
- Black box → Interpretable
```

---

## References

1. OpenAI (2024). "Learning to Reason with LLMs" (o1)
2. Yao et al. (2023). "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
3. Park et al. (2023). "Generative Agents: Interactive Simulacra of Human Behavior"
4. Ha & Schmidhuber (2018). "World Models"

---

## Exercises

### Exercise 1: Test-Time Compute Scaling Trade-offs
o1-style models spend more compute during inference rather than relying solely on training-time scaling. Compare the two paradigms for the following scenarios and explain which is more appropriate in each case.

| Scenario | Training-time scaling | Test-time compute scaling | Better choice | Reason |
|----------|----------------------|--------------------------|---------------|--------|
| A) Answering 1M customer support queries per day | ??? | ??? | ??? | ??? |
| B) Solving a novel mathematics competition problem | ??? | ??? | ??? | ??? |
| C) Real-time coding autocomplete (< 50ms latency) | ??? | ??? | ??? | ??? |

<details>
<summary>Show Answer</summary>

| Scenario | Training-time scaling | Test-time compute | Better choice | Reason |
|----------|----------------------|-------------------|---------------|--------|
| A) 1M customer support queries/day | Larger model handles more query types; fixed latency | Higher quality on difficult queries, but adds latency and cost per query | **Training-time scaling** | Customer support has high volume and predictable query types. Each query should be handled quickly. Training-time scaling (larger or better-trained model) gives consistent quality per query at fixed marginal cost. Test-time compute would multiply cost by 10-50x per query — prohibitive at 1M/day. |
| B) Novel math competition problem | Better general mathematical knowledge | Generate multiple solution paths, verify each, backtrack on errors | **Test-time compute scaling** | Novel competition problems often require exploring multiple approaches before finding the right one. A human mathematician doesn't solve IMO problems in 2 seconds — they think for hours. Test-time compute mimics this: spend 10-100x more tokens on hard problems. Volume is low (1 problem), so high per-problem cost is acceptable. |
| C) Real-time code autocomplete (<50ms) | Model must know common code patterns at inference time | More thinking → more latency → violates 50ms requirement | **Training-time scaling** | Hard latency constraint of 50ms makes test-time compute scaling infeasible (it requires many additional forward passes). Instead: use a well-trained small model (< 1B params) with high accuracy on common code patterns from training. Techniques like speculative decoding can improve speed, but the thinking budget must be near-zero. |

**Key insight**: Test-time compute is most valuable for problems that are **rare, difficult, and allow spending more time** on individual instances. Training-time scaling is better for **high-volume, latency-sensitive, or cost-constrained** applications.

</details>

### Exercise 2: Synthetic Data Model Collapse Risk
The `SyntheticDataGenerator` trains on teacher-generated data. Explain the "model collapse" phenomenon and describe two concrete strategies to detect and prevent it.

```python
# Potential collapse scenario:
# Round 1: Train on human data → Model v1
# Round 2: Generate synthetic data with v1 → Train → Model v2
# Round 3: Generate synthetic data with v2 → Train → Model v3
# ...
# Round N: What happens to the distribution?
```

<details>
<summary>Show Answer</summary>

**Model collapse mechanism**:

At each round, the model generates synthetic training data from its own distribution. Each generation step introduces subtle biases and loses tail-distribution examples (rare but important patterns). When the next model trains on this narrowed distribution, it amplifies the bias further.

```
Round 1 (human data): Distribution covers {common, uncommon, rare} content
Round 2 (v1 synthetic): Drops some "rare" patterns (low probability → rarely sampled)
Round 3 (v2 synthetic): Further shrinks distribution, amplifies v1's biases
Round N: Distribution collapses to high-probability, repetitive outputs
         - Reduced vocabulary diversity
         - Repetitive stylistic patterns
         - Loss of factual edge cases
         - Overconfident on uncertain questions (model v1's confident wrong answers propagate)
```

**Prevention Strategy 1: Human data replay**

```python
def train_with_replay(synthetic_data, human_data, replay_ratio=0.1):
    """Always include a fraction of original human data"""
    # In every training batch: 10% human data + 90% synthetic
    # Human data anchors the distribution to the real world
    # Prevents drift toward narrow synthetic distribution
    human_sample = random.sample(human_data, int(len(synthetic_data) * replay_ratio))
    combined = synthetic_data + human_sample
    return combined
```
This ensures the model always sees "ground truth" human data and cannot fully collapse.

**Prevention Strategy 2: Distribution divergence monitoring**

```python
def check_distribution_health(model_v_current, model_v_previous, test_prompts):
    """Detect collapse by comparing output distributions"""
    # Generate outputs from both model versions
    outputs_current = [model_v_current.generate(p) for p in test_prompts]
    outputs_previous = [model_v_previous.generate(p) for p in test_prompts]

    # Measure: vocabulary diversity (unique tokens / total tokens)
    diversity_current = len(set(tokenize(outputs_current))) / len(tokenize(outputs_current))
    diversity_previous = len(set(tokenize(outputs_previous))) / len(tokenize(outputs_previous))

    # Alert if diversity drops > 10% between generations
    if diversity_current < diversity_previous * 0.9:
        raise CollapseWarning(f"Distribution shrinking: {diversity_previous:.3f} → {diversity_current:.3f}")

    # Also monitor: perplexity on human-written held-out text
    # Increasing perplexity = model drifting from human distribution
```

</details>

### Exercise 3: Multi-Agent Debate vs. Single Model
In a multi-agent debate system, multiple LLMs argue for different positions and then reach consensus. For which types of problems does debate improve answer quality vs. potentially degrading it?

Evaluate these tasks:
- Task A: "Is the following Python code correct? [function with subtle off-by-one error]"
- Task B: "What is the capital of France?"
- Task C: "Should a company prioritize profit or employee well-being?"

<details>
<summary>Show Answer</summary>

**Task A: Code correctness with subtle bug — Debate HELPS**

Different agents independently review the code and may notice the off-by-one error from different angles:
- Agent 1: "The loop condition uses `<` but should use `<=` because..."
- Agent 2: "The initialization at `i=1` misses the first element..."
- Agent 3: "When array length is 0, this crashes because..."

The adversarial structure forces each agent to look for flaws. A single model may confidently say "looks correct" (sycophancy toward the prompt's implicit assumption that it is correct). Multiple agents with independent review are more likely to surface the bug.

**Task B: Capital of France — Debate HURTS (or wastes resources)**

"Paris" is a factual answer with no ambiguity. If agents debate it:
- Agent 1: "Paris"
- Agent 2: "Paris"
- Agent 3: "Paris, but historically Versailles served as capital under Louis XIV..."

The debate adds latency, cost, and potential for agents to introduce spurious "interesting" qualifications that confuse the final answer. For factual, closed-domain questions with deterministic answers, single model with high confidence is better.

**Task C: Profit vs. employee well-being — Debate HELPS (but risks polarization)**

This is a genuinely contested question with multiple valid perspectives. Debate structure can surface:
- Economic efficiency arguments (profit enables reinvestment → long-term employee benefit)
- Social contract arguments (employees are stakeholders, not just resources)
- Empirical evidence (companies with high employee satisfaction outperform long-term)

**Risk**: If agents are assigned fixed "sides" (Agent A = pro-profit, Agent B = pro-employee), they may generate persuasive but one-sided arguments that amplify polarization rather than reaching a nuanced synthesis. Better to have agents generate multiple independent perspectives without fixed assignments.

**General rule**: Debate improves quality for **reasoning problems** (where multiple paths can be cross-checked), **adversarial tasks** (bug finding, security review), and **value-laden questions** (where multiple perspectives add nuance). It adds cost without benefit for **factual lookups**, **simple computations**, and **consensus tasks** where all models agree anyway.

</details>

### Exercise 4: World Models for Physical Reasoning
Sora-style world models learn to simulate physical environments from video data. Explain one fundamental limitation that prevents current world models from being used as reliable physics simulators, and describe what kind of training data would be needed to address it.

<details>
<summary>Show Answer</summary>

**Fundamental limitation: Lack of causal grounding (correlation vs. causation)**

Current video-based world models learn **statistical correlations** in pixel space: "when I see this frame configuration, the next frame typically looks like this." They do not learn the **causal physical laws** (Newton's laws, conservation of energy, fluid dynamics equations) that generate those correlations.

**Concrete example**:
A world model trained on videos of balls rolling down ramps learns to predict: "ball at top of ramp → ball rolling → ball at bottom." But it does this by memorizing the visual trajectory pattern, not by modeling force = mass × acceleration.

**Failure modes**:
- **Counterfactual reasoning**: "If the ball was twice as heavy, how would it roll?" — the model can't answer because it never learned mass-acceleration relationships, only visual patterns.
- **Novel physics configurations**: A ramp with a shape never seen in training videos may generate physically impossible trajectories (ball passing through the ramp surface).
- **Long-horizon extrapolation**: Even if short-term prediction is accurate (learned from videos), extrapolating 10 seconds ahead accumulates errors because small deviations from the learned distribution compound.

**What training data would address this**:

1. **Labeled physics simulations** (e.g., from game engines): Videos paired with ground-truth physical quantities (velocity, force, energy at each frame). The model can learn to predict these quantities, not just pixel values.

2. **Interventional data** (counterfactual pairs): The same physical setup with different object masses, friction coefficients, etc. "Ball A (1kg) rolls to position X in 2s; Ball B (2kg) rolls to position X in 2s" — teaches mass-independence of ramp trajectories (only relevant when air resistance is present), grounding the model in actual causal structure.

3. **Long-horizon physics benchmarks**: Evaluation datasets that test 30+ second extrapolations, forcing models to learn stable physical representations rather than short-term pattern matching.

The key insight: visual world models are excellent for **predictive** tasks (what will happen next?) but insufficient for **causal** tasks (why did it happen, and what if it were different?).

</details>
5. Sora Technical Report (2024)

# 04. 고급 추론 프롬프트(Advanced Reasoning Prompts)

**이전**: [사고의 연쇄](./03_Chain_of_Thought.md) | **다음**: [구조화된 출력과 포매팅](./05_Structured_Output_Prompting.md)

## 학습 목표

- 분기 탐색, 평가, 역추적을 포함한 사고의 나무(Tree of Thoughts) 프롬프팅을 구현한다
- 자기 일관성(Self-consistency)과 사고의 그래프(Graph of Thoughts)를 적용하여 추론 신뢰성을 향상시킨다
- 과제별 프롬프트를 자동으로 생성하는 메타 프롬프트(Meta-prompt)를 설계한다
- 자기 개선(Self-Refine)과 반성(Reflexion) 패턴을 사용하여 반복적 출력 개선을 수행한다
- 주어진 과제에 적절한 고급 추론 기법을 비교하고 선택한다

---

사고의 연쇄(Chain-of-Thought, CoT) 프롬프팅은 단일 선형 추론 경로를 따릅니다. 하지만 많은 실세계 문제는 여러 경로를 탐색하고, 이를 평가·비교하고, 막다른 길에서 역추적하거나, 반복적으로 해를 개선하는 것으로부터 이점을 얻습니다. 이 레슨에서는 선형 체인을 넘어서는 고급 추론 기법 — 모델이 더 폭넓게, 더 비판적으로, 더 유연하게 사고할 수 있게 하는 방법들을 다룹니다.

## 목차

1. [사고의 나무(Tree of Thoughts)](#1-사고의-나무tree-of-thoughts)
2. [자기 일관성 재방문](#2-자기-일관성-재방문)
3. [사고의 그래프(Graph of Thoughts)](#3-사고의-그래프graph-of-thoughts)
4. [사고의 골격(Skeleton-of-Thought)](#4-사고의-골격skeleton-of-thought)
5. [메타 프롬프팅(Meta-Prompting)](#5-메타-프롬프팅meta-prompting)
6. [자기 개선(Self-Refine)](#6-자기-개선self-refine)
7. [반성(Reflexion)](#7-반성reflexion)
8. [한 걸음 물러서기 프롬프팅(Step-Back Prompting)](#8-한-걸음-물러서기-프롬프팅step-back-prompting)
9. [유추적 프롬프팅(Analogical Prompting)](#9-유추적-프롬프팅analogical-prompting)
10. [추론 기법 비교](#10-추론-기법-비교)
11. [연습문제](#연습문제)

---

## 1. 사고의 나무(Tree of Thoughts)

### 1.1 체인에서 나무로

사고의 나무(Tree of Thoughts, ToT)는 Yao et al. (2023)이 소개했으며, CoT를 단일 체인에서 나무 구조로 확장합니다. 각 추론 단계에서 모델은 여러 후보 연속을 생성하고, 각각을 평가하며, 가장 유망한 경로를 선택하여 계속합니다. 경로가 막다른 길로 이어지면 모델은 역추적하여 대안을 탐색할 수 있습니다.

핵심 통찰: 많은 문제에는 처음 시도한 접근법이 항상 최선은 아닌 분기 해 공간이 있습니다. ToT는 모델에게 탐색, 평가, 가지치기 — 인간 추론에서는 자연스럽지만 표준 CoT에는 없는 — 능력을 제공합니다.

```python
import anthropic
from dataclasses import dataclass, field

client = anthropic.Anthropic()

@dataclass
class ThoughtNode:
    """A node in the Tree of Thoughts."""
    thought: str
    evaluation: float  # 0.0 to 1.0 (model's confidence this path is promising)
    children: list["ThoughtNode"] = field(default_factory=list)
    is_terminal: bool = False
    final_answer: str | None = None

def generate_thoughts(
    problem: str,
    current_path: list[str],
    num_thoughts: int = 3
) -> list[str]:
    """Generate multiple candidate next steps."""

    path_text = "\n".join(
        f"Step {i+1}: {step}" for i, step in enumerate(current_path)
    )

    prompt = f"""Problem: {problem}

Current reasoning path:
{path_text if path_text else "(Starting fresh)"}

Generate {num_thoughts} different possible next steps.
Each should represent a distinct approach or reasoning direction.
Number them 1-{num_thoughts}.
Be specific and actionable — each thought should make concrete progress."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.8,  # Higher temp for diverse thoughts
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse numbered thoughts
    text = message.content[0].text
    thoughts = []
    for line in text.split("\n"):
        line = line.strip()
        if line and line[0].isdigit() and "." in line[:3]:
            thoughts.append(line.split(".", 1)[1].strip())

    return thoughts[:num_thoughts]
```

### 1.2 평가와 선택

```python
def evaluate_thought(
    problem: str,
    current_path: list[str],
    candidate_thought: str
) -> float:
    """Evaluate how promising a candidate thought is.

    Returns a score from 0.0 to 1.0.
    """

    path_text = "\n".join(
        f"Step {i+1}: {step}" for i, step in enumerate(current_path)
    )

    prompt = f"""Problem: {problem}

Current reasoning path:
{path_text}

Proposed next step: {candidate_thought}

Evaluate this proposed step on a scale of 0.0 to 1.0:
- 1.0: Clearly correct and makes strong progress toward the solution
- 0.7: Reasonable approach, likely productive
- 0.4: Uncertain, might work but has risks
- 0.1: Likely a dead end or incorrect direction

Consider:
1. Is the logic sound?
2. Does it make progress toward solving the problem?
3. Does it contradict any established facts?
4. Is there a clearly better alternative?

Score (respond with only a number between 0.0 and 1.0):"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        return float(message.content[0].text.strip())
    except ValueError:
        return 0.5  # Default if parsing fails
```

### 1.3 완전한 ToT 알고리즘

```python
import anthropic

client = anthropic.Anthropic()

def tree_of_thoughts(
    problem: str,
    max_depth: int = 4,
    branch_factor: int = 3,
    prune_threshold: float = 0.3,
    beam_width: int = 2
) -> dict:
    """Solve a problem using Tree of Thoughts.

    Args:
        problem: The problem to solve
        max_depth: Maximum reasoning steps
        branch_factor: Number of candidate thoughts per step
        prune_threshold: Minimum evaluation score to continue a path
        beam_width: Number of top paths to keep at each level

    Returns:
        Dict with the best solution path and answer
    """

    # Each beam entry: (path, cumulative_score)
    beams = [([], 0.0)]

    for depth in range(max_depth):
        all_candidates = []

        for path, cum_score in beams:
            # Generate candidate next steps
            thoughts = generate_thoughts(problem, path, branch_factor)

            for thought in thoughts:
                # Evaluate each candidate
                score = evaluate_thought(problem, path, thought)

                if score >= prune_threshold:
                    new_path = path + [thought]
                    new_cum_score = cum_score + score
                    all_candidates.append((new_path, new_cum_score))

        if not all_candidates:
            break  # All paths pruned

        # Keep top-k paths (beam search)
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_width]

    # Extract the best path
    best_path, best_score = beams[0]

    # Generate final answer from the best path
    path_text = "\n".join(
        f"Step {i+1}: {step}" for i, step in enumerate(best_path)
    )

    final_msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"""Problem: {problem}

Reasoning path:
{path_text}

Based on this reasoning, what is the final answer?
State it clearly and concisely."""
            }
        ]
    )

    return {
        "path": best_path,
        "score": best_score,
        "answer": final_msg.content[0].text,
        "depth": len(best_path)
    }

# Example usage
result = tree_of_thoughts(
    problem="Write a function that determines if a number can be expressed "
            "as the sum of exactly three perfect squares. "
            "What is the most efficient algorithm?",
    max_depth=3,
    branch_factor=3,
    beam_width=2
)

print(f"Best path ({len(result['path'])} steps):")
for i, step in enumerate(result["path"]):
    print(f"  {i+1}. {step}")
print(f"\nAnswer: {result['answer']}")
```

### 1.4 창의적 문제를 위한 ToT

ToT는 여러 유효한 접근법이 존재하는 창의적이거나 개방형 문제에 특히 강력합니다:

```python
import anthropic

client = anthropic.Anthropic()

# Creative problem: designing a system architecture
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """I need to design a notification system for a social media app.

Using Tree of Thoughts, explore 3 different architectural approaches.
For each approach:
1. Describe the architecture in 2-3 sentences
2. List 2 pros and 2 cons
3. Rate feasibility for a team of 3 engineers (1-10)

Then select the best approach and explain why.

Approach 1 (Push-based):
Approach 2 (Pull-based):
Approach 3 (Hybrid):

Evaluation and selection:"""
        }
    ]
)
```

---

## 2. 자기 일관성 재방문

### 2.1 다수결 투표를 넘어서

레슨 03에서 여러 CoT 경로에 대한 다수결 투표로 자기 일관성을 소개했습니다. 여기서는 더 정교한 집계 전략으로 이를 확장합니다.

```python
import anthropic
import re
from collections import Counter
from typing import Any

client = anthropic.Anthropic()

def weighted_self_consistency(
    question: str,
    num_paths: int = 7,
    temperature: float = 0.7
) -> dict:
    """Self-consistency with confidence-weighted voting.

    Instead of equal votes, weight each path by the model's
    self-assessed confidence in its reasoning.
    """

    paths = []

    for _ in range(num_paths):
        # Generate reasoning with confidence assessment
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": f"""{question}

Think through this step by step.
After reaching your answer, rate your confidence from 0 to 100.

Format your response as:
REASONING: [your step-by-step reasoning]
ANSWER: [your answer]
CONFIDENCE: [0-100]"""
                }
            ]
        )

        text = msg.content[0].text

        # Parse response
        answer_match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", text)
        confidence_match = re.search(r"CONFIDENCE:\s*(\d+)", text)

        answer = answer_match.group(1).strip() if answer_match else "unknown"
        confidence = int(confidence_match.group(1)) if confidence_match else 50

        paths.append({
            "answer": answer,
            "confidence": confidence,
            "reasoning": text
        })

    # Weighted voting
    weighted_votes: dict[str, float] = {}
    for path in paths:
        answer = path["answer"]
        weight = path["confidence"] / 100.0
        weighted_votes[answer] = weighted_votes.get(answer, 0) + weight

    # Select answer with highest weighted vote
    best_answer = max(weighted_votes, key=weighted_votes.get)
    total_weight = sum(weighted_votes.values())

    return {
        "answer": best_answer,
        "weighted_confidence": weighted_votes[best_answer] / total_weight,
        "vote_distribution": weighted_votes,
        "num_paths": num_paths,
        "paths": paths
    }
```

### 2.2 범용 자기 일관성(Universal Self-Consistency)

정확한 문자열 매칭이 작동하지 않는 개방형 생성 과제의 경우, 모델 자체를 사용하여 답변이 동등한지 판별합니다:

```python
import anthropic

client = anthropic.Anthropic()

def are_answers_equivalent(answer_a: str, answer_b: str, question: str) -> bool:
    """Use the model to determine if two answers are semantically equivalent."""

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"""Given this question: "{question}"

Are these two answers saying the same thing?
Answer A: "{answer_a}"
Answer B: "{answer_b}"

Reply with only "yes" or "no"."""
            }
        ]
    )

    return msg.content[0].text.strip().lower() == "yes"


def universal_self_consistency(
    question: str,
    num_paths: int = 5,
    temperature: float = 0.7
) -> dict:
    """Self-consistency for open-ended answers using semantic matching."""

    # Generate multiple answers
    answers = []
    for _ in range(num_paths):
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": f"{question}\n\nThink step by step, then provide your answer."
                }
            ]
        )
        answers.append(msg.content[0].text)

    # Cluster semantically equivalent answers
    clusters: list[list[int]] = []
    for i, ans_a in enumerate(answers):
        placed = False
        for cluster in clusters:
            representative_idx = cluster[0]
            if are_answers_equivalent(answers[representative_idx], ans_a, question):
                cluster.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])

    # Find the largest cluster
    largest_cluster = max(clusters, key=len)
    representative_answer = answers[largest_cluster[0]]

    return {
        "answer": representative_answer,
        "agreement": len(largest_cluster) / num_paths,
        "num_clusters": len(clusters),
        "cluster_sizes": [len(c) for c in clusters]
    }
```

---

## 3. 사고의 그래프(Graph of Thoughts)

### 3.1 나무에서 그래프로

사고의 그래프(Graph of Thoughts, GoT)는 Besta et al. (2023)이 소개했으며, 사고가 병합, 결합, 순환을 형성할 수 있도록 ToT를 확장합니다. 나무가 각 노드에 정확히 하나의 부모를 강제하는 반면, 그래프는 사고가 여러 이전 사고 위에 동시에 구축되는 것을 허용합니다.

이는 독립적으로 풀 수 있는 하위 구성 요소가 있고 이후에 결합되는 문제에 특히 유용합니다.

```python
import anthropic
from dataclasses import dataclass, field
from typing import Any

client = anthropic.Anthropic()

@dataclass
class GoTNode:
    """A node in the Graph of Thoughts."""
    id: str
    thought: str
    score: float = 0.0
    parent_ids: list[str] = field(default_factory=list)

class GraphOfThoughts:
    """Graph of Thoughts implementation for complex reasoning."""

    def __init__(self, problem: str):
        self.problem = problem
        self.nodes: dict[str, GoTNode] = {}
        self.node_counter = 0

    def _next_id(self) -> str:
        self.node_counter += 1
        return f"thought_{self.node_counter}"

    def generate(self, parent_ids: list[str] | None = None, n: int = 3) -> list[str]:
        """Generate new thought nodes, optionally building on parent thoughts."""

        context = f"Problem: {self.problem}\n\n"

        if parent_ids:
            context += "Building on these previous thoughts:\n"
            for pid in parent_ids:
                if pid in self.nodes:
                    context += f"- [{pid}]: {self.nodes[pid].thought}\n"
            context += "\n"

        prompt = f"""{context}Generate {n} distinct next thoughts that make progress
toward solving this problem.
{'Combine and extend the previous thoughts above.' if parent_ids else 'Start with initial approaches.'}
Number each thought 1-{n}."""

        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            temperature=0.8,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse and create nodes
        new_ids = []
        for line in msg.content[0].text.split("\n"):
            line = line.strip()
            if line and line[0].isdigit() and "." in line[:3]:
                thought = line.split(".", 1)[1].strip()
                node_id = self._next_id()
                self.nodes[node_id] = GoTNode(
                    id=node_id,
                    thought=thought,
                    parent_ids=parent_ids or []
                )
                new_ids.append(node_id)

        return new_ids

    def aggregate(self, node_ids: list[str]) -> str:
        """Merge multiple thoughts into a combined insight.

        This is the key operation that distinguishes GoT from ToT.
        """

        thoughts_text = "\n".join(
            f"- [{nid}]: {self.nodes[nid].thought}"
            for nid in node_ids if nid in self.nodes
        )

        prompt = f"""Problem: {self.problem}

These separate lines of reasoning have been developed:
{thoughts_text}

Synthesize these thoughts into a single, coherent combined insight
that takes the best elements from each. Provide one paragraph."""

        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )

        node_id = self._next_id()
        self.nodes[node_id] = GoTNode(
            id=node_id,
            thought=msg.content[0].text,
            parent_ids=node_ids
        )

        return node_id

    def refine(self, node_id: str) -> str:
        """Improve an existing thought through self-critique."""

        node = self.nodes[node_id]

        prompt = f"""Problem: {self.problem}

Current thought: {node.thought}

Critique this thought:
1. What is correct and valuable?
2. What is missing or could be wrong?
3. How can it be improved?

Provide an improved version of the thought."""

        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )

        new_id = self._next_id()
        self.nodes[new_id] = GoTNode(
            id=new_id,
            thought=msg.content[0].text,
            parent_ids=[node_id]
        )

        return new_id

    def solve(self) -> str:
        """Run the GoT algorithm and return the final answer."""

        # Phase 1: Generate initial thoughts
        initial_ids = self.generate(n=3)

        # Phase 2: Develop each thought independently
        developed_ids = []
        for init_id in initial_ids:
            extended_ids = self.generate(parent_ids=[init_id], n=2)
            developed_ids.extend(extended_ids)

        # Phase 3: Aggregate — combine insights from different branches
        merged_id = self.aggregate(developed_ids)

        # Phase 4: Refine the merged thought
        refined_id = self.refine(merged_id)

        return self.nodes[refined_id].thought

# Usage
# got = GraphOfThoughts(
#     "Design a rate-limiting algorithm for a distributed API gateway "
#     "that handles 10,000 requests per second across 5 nodes."
# )
# solution = got.solve()
# print(solution)
```

---

## 4. 사고의 골격(Skeleton-of-Thought)

### 4.1 병렬 생성

사고의 골격(Skeleton-of-Thought, SoT)은 Ning et al. (2023)이 소개했으며, 먼저 개요(골격)를 만들고 각 섹션을 병렬로 확장하여 LLM 생성 속도를 높입니다. 여러 섹션이 동시에 생성되므로 지연 시간이 줄어듭니다.

```python
import anthropic
import asyncio

client = anthropic.Anthropic()

def generate_skeleton(question: str) -> list[str]:
    """Generate a skeleton outline for the answer."""

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"""Question: {question}

Provide a skeleton outline of your answer. List 3-6 key points
as short bullet points (one line each). Do not explain anything yet —
just list the main points you will cover.

Skeleton:"""
            }
        ]
    )

    # Parse skeleton points
    points = []
    for line in msg.content[0].text.split("\n"):
        line = line.strip()
        if line and (line.startswith("-") or line.startswith("•")):
            points.append(line.lstrip("-•").strip())
        elif line and line[0].isdigit() and "." in line[:3]:
            points.append(line.split(".", 1)[1].strip())

    return points


def expand_point(question: str, skeleton: list[str], point_idx: int) -> str:
    """Expand a single skeleton point into a full paragraph."""

    skeleton_text = "\n".join(f"- {p}" for p in skeleton)

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"""Question: {question}

Full outline:
{skeleton_text}

Now expand point {point_idx + 1}: "{skeleton[point_idx]}"
into a detailed paragraph (3-5 sentences).
Do not repeat information from other points.
Be specific and include concrete details."""
            }
        ]
    )

    return msg.content[0].text


def skeleton_of_thought(question: str) -> str:
    """Full SoT pipeline: skeleton then parallel expansion."""

    # Step 1: Generate skeleton
    skeleton = generate_skeleton(question)
    print(f"Skeleton ({len(skeleton)} points):")
    for i, point in enumerate(skeleton):
        print(f"  {i+1}. {point}")

    # Step 2: Expand each point (sequentially here;
    # in production, use asyncio or threading for parallel execution)
    expanded = []
    for i in range(len(skeleton)):
        paragraph = expand_point(question, skeleton, i)
        expanded.append(paragraph)

    # Step 3: Assemble
    full_answer = "\n\n".join(
        f"**{skeleton[i]}**\n{expanded[i]}"
        for i in range(len(skeleton))
    )

    return full_answer

# Usage
# answer = skeleton_of_thought("What are the main challenges in deploying LLMs in production?")
# print(answer)
```

### 4.2 비동기 병렬 확장

프로덕션에서는 골격 포인트를 병렬로 확장합니다:

```python
import anthropic
import asyncio

async def expand_point_async(
    async_client: anthropic.AsyncAnthropic,
    question: str,
    skeleton: list[str],
    point_idx: int
) -> tuple[int, str]:
    """Expand a single point asynchronously."""

    skeleton_text = "\n".join(f"- {p}" for p in skeleton)

    msg = await async_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"""Question: {question}

Full outline:
{skeleton_text}

Expand point {point_idx + 1}: "{skeleton[point_idx]}"
into a detailed paragraph (3-5 sentences)."""
            }
        ]
    )

    return point_idx, msg.content[0].text


async def skeleton_of_thought_parallel(question: str) -> str:
    """SoT with parallel expansion for reduced latency."""

    sync_client = anthropic.Anthropic()
    async_client = anthropic.AsyncAnthropic()

    # Step 1: Generate skeleton (sequential)
    skeleton_msg = sync_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"Question: {question}\n\nList 4-6 key points as a skeleton outline:"
            }
        ]
    )

    skeleton = [
        line.lstrip("-•0123456789.").strip()
        for line in skeleton_msg.content[0].text.split("\n")
        if line.strip() and (line.strip()[0] in "-•" or line.strip()[0].isdigit())
    ]

    # Step 2: Expand all points in parallel
    tasks = [
        expand_point_async(async_client, question, skeleton, i)
        for i in range(len(skeleton))
    ]

    results = await asyncio.gather(*tasks)
    results.sort(key=lambda x: x[0])  # Sort by index

    # Step 3: Assemble
    full_answer = "\n\n".join(
        f"**{skeleton[idx]}**\n{text}" for idx, text in results
    )

    return full_answer

# Usage:
# answer = asyncio.run(skeleton_of_thought_parallel("Explain microservices architecture"))
```

---

## 5. 메타 프롬프팅(Meta-Prompting)

### 5.1 프롬프트를 생성하는 프롬프트

메타 프롬프팅은 LLM을 사용하여 특정 과제에 최적화된 프롬프트를 생성합니다. 수동으로 프롬프트를 작성하는 대신, 과제를 설명하고 모델이 프롬프트를 설계하도록 합니다.

```python
import anthropic

client = anthropic.Anthropic()

def generate_prompt(
    task_description: str,
    target_model: str = "claude-sonnet-4-20250514",
    optimization_goal: str = "accuracy"
) -> str:
    """Use an LLM to generate an optimized prompt for a task."""

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        temperature=0.3,
        messages=[
            {
                "role": "user",
                "content": f"""You are a prompt engineering expert. Design an optimized
prompt for the following task.

Task: {task_description}
Target model: {target_model}
Optimization goal: {optimization_goal}

Design a complete prompt with these elements:
1. System prompt (if needed)
2. User message template (with {{placeholders}} for variables)
3. Output format specification
4. 2-3 few-shot examples (if beneficial)
5. Edge case handling instructions

Also explain why you made each design decision (in a separate section
marked "Design Notes").

Output the prompt first, then the design notes."""
            }
        ]
    )

    return msg.content[0].text

# Example: Generate a prompt for sentiment analysis
generated = generate_prompt(
    task_description="Classify product reviews into sentiment categories "
                     "(positive, negative, neutral, mixed) with confidence scores. "
                     "Must handle sarcasm, multi-language reviews, and reviews "
                     "that discuss multiple products.",
    optimization_goal="high precision on sarcasm detection"
)

print(generated)
```

### 5.2 반복적 프롬프트 최적화

메타 프롬프팅은 테스트 결과에 기반하여 프롬프트를 반복적으로 개선할 수도 있습니다:

```python
import anthropic

client = anthropic.Anthropic()

def optimize_prompt_iteratively(
    initial_prompt: str,
    test_cases: list[dict],
    max_iterations: int = 3
) -> dict:
    """Iteratively optimize a prompt based on test case performance.

    Args:
        initial_prompt: The starting prompt template
        test_cases: List of {"input": ..., "expected": ...} dicts
        max_iterations: Maximum optimization rounds

    Returns:
        Dict with optimized prompt and performance history
    """

    current_prompt = initial_prompt
    history = []

    for iteration in range(max_iterations):
        # Test current prompt
        results = []
        for tc in test_cases:
            formatted_prompt = current_prompt.replace("{{INPUT}}", tc["input"])

            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=128,
                temperature=0.0,
                messages=[{"role": "user", "content": formatted_prompt}]
            )

            actual = msg.content[0].text.strip()
            correct = actual.lower() == tc["expected"].lower()
            results.append({
                "input": tc["input"],
                "expected": tc["expected"],
                "actual": actual,
                "correct": correct
            })

        accuracy = sum(r["correct"] for r in results) / len(results)
        history.append({"iteration": iteration, "accuracy": accuracy})

        if accuracy >= 1.0:
            break  # Perfect score, stop optimizing

        # Get failures for analysis
        failures = [r for r in results if not r["correct"]]

        # Ask the model to improve the prompt
        improvement_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": f"""The following prompt achieved {accuracy:.0%} accuracy.

Current prompt:
```
{current_prompt}
```

These test cases FAILED:
{chr(10).join(
    f"- Input: {f['input']}, Expected: {f['expected']}, Got: {f['actual']}"
    for f in failures
)}

Analyze why these cases failed and rewrite the prompt to handle them
correctly while maintaining performance on the passing cases.

Output ONLY the improved prompt, nothing else."""
                }
            ]
        )

        current_prompt = improvement_msg.content[0].text.strip()
        # Remove code block markers if present
        if current_prompt.startswith("```"):
            current_prompt = current_prompt.split("\n", 1)[1]
            current_prompt = current_prompt.rsplit("```", 1)[0]

    return {
        "optimized_prompt": current_prompt,
        "final_accuracy": history[-1]["accuracy"],
        "history": history
    }
```

### 5.3 과제 적응형 메타 프롬프팅

```python
import anthropic

client = anthropic.Anthropic()

def task_adaptive_meta_prompt(
    task: str,
    examples: list[dict] | None = None,
    constraints: list[str] | None = None
) -> dict:
    """Generate a task-specific prompt with automatic technique selection.

    The meta-prompt analyzes the task and decides which prompting technique
    (zero-shot, few-shot, CoT, etc.) would work best.
    """

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"""Analyze this task and design the optimal prompting strategy.

Task: {task}
{"Examples available: " + str(len(examples)) if examples else "No examples available"}
{"Constraints: " + ", ".join(constraints) if constraints else "No specific constraints"}

Step 1: Classify the task type (classification, generation, reasoning,
extraction, transformation, analysis).

Step 2: Determine the best prompting technique:
- Zero-shot: for well-understood, simple tasks
- Few-shot: for custom formats or taxonomies
- CoT: for multi-step reasoning
- Self-consistency: for high-stakes reasoning
- ToT: for creative/open-ended reasoning
- Least-to-most: for complex decomposable problems

Step 3: Generate the complete prompt using the selected technique.

Format your response as:
TASK_TYPE: [type]
TECHNIQUE: [technique]
RATIONALE: [1-2 sentences]
PROMPT:
[the complete prompt]"""
            }
        ]
    )

    return {
        "analysis": msg.content[0].text,
    }
```

---

## 6. 자기 개선(Self-Refine)

### 6.1 반복적 자기 개선

자기 개선(Self-Refine, Madaan et al., 2023)은 반복적 개선 루프를 통해 모델이 자신의 출력을 비판하고 개선하도록 합니다:

1. **생성**: 초기 출력을 만듭니다
2. **비판**: 출력을 평가하고 약점을 식별합니다
3. **개선**: 비판에 기반하여 출력을 개선합니다
4. 만족할 때까지 2-3단계를 반복합니다

```python
import anthropic

client = anthropic.Anthropic()

def self_refine(
    task: str,
    max_iterations: int = 3,
    satisfaction_threshold: float = 0.9
) -> dict:
    """Iteratively self-refine an output.

    Args:
        task: The task to complete
        max_iterations: Maximum refinement rounds
        satisfaction_threshold: Stop if quality score exceeds this

    Returns:
        Dict with final output, iteration count, and refinement history
    """

    history = []

    # Step 1: Initial generation
    gen_msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.3,
        messages=[
            {"role": "user", "content": task}
        ]
    )
    current_output = gen_msg.content[0].text

    for iteration in range(max_iterations):
        # Step 2: Critique
        critique_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": f"""Task: {task}

Current output:
\"\"\"
{current_output}
\"\"\"

Critique this output:
1. What is good about it? (2-3 points)
2. What are the weaknesses? (2-3 points)
3. What specific improvements would you make? (2-3 points)
4. Rate the overall quality from 0.0 to 1.0.

Format the rating as: QUALITY: [score]"""
                }
            ]
        )

        critique = critique_msg.content[0].text

        # Extract quality score
        import re
        quality_match = re.search(r"QUALITY:\s*([\d.]+)", critique)
        quality = float(quality_match.group(1)) if quality_match else 0.5

        history.append({
            "iteration": iteration,
            "output": current_output,
            "critique": critique,
            "quality": quality
        })

        if quality >= satisfaction_threshold:
            break

        # Step 3: Refine
        refine_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": f"""Task: {task}

Previous output:
\"\"\"
{current_output}
\"\"\"

Critique of the previous output:
{critique}

Now produce an improved version that addresses all the weaknesses
identified in the critique while preserving the strengths.
Output ONLY the improved version, no commentary."""
                }
            ]
        )

        current_output = refine_msg.content[0].text

    return {
        "final_output": current_output,
        "iterations": len(history),
        "final_quality": history[-1]["quality"],
        "history": history
    }

# Example usage
result = self_refine(
    task="""Write a Python function that implements a thread-safe
LRU cache with a maximum size of N entries. Include:
- Type hints
- Docstring
- Thread safety using threading locks
- O(1) get and put operations""",
    max_iterations=3
)

print(f"Refined over {result['iterations']} iterations")
print(f"Final quality: {result['final_quality']}")
print(f"\n{result['final_output']}")
```

### 6.2 도메인별 비판

더 나은 개선을 위해 도메인별 비판 기준을 제공합니다:

```python
import anthropic

client = anthropic.Anthropic()

def self_refine_with_rubric(
    task: str,
    rubric: dict[str, str],
    max_iterations: int = 3
) -> dict:
    """Self-refine with a specific evaluation rubric.

    Args:
        task: The task to complete
        rubric: Dict of {criterion_name: criterion_description}
        max_iterations: Maximum refinement rounds
    """

    rubric_text = "\n".join(
        f"- {name}: {desc}" for name, desc in rubric.items()
    )

    # Generate initial output
    gen_msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.3,
        messages=[{"role": "user", "content": task}]
    )
    current = gen_msg.content[0].text

    for _ in range(max_iterations):
        # Critique against rubric
        critique_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": f"""Task: {task}

Output to evaluate:
{current}

Evaluate against each criterion (score 1-5 for each):
{rubric_text}

For each criterion, give the score and one sentence of feedback.
Then calculate the average score.
Format: AVERAGE: [score]"""
                }
            ]
        )

        # Refine based on critique
        refine_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": f"""Task: {task}

Current version:
{current}

Feedback:
{critique_msg.content[0].text}

Improve the output to score 5/5 on every criterion.
Output ONLY the improved version."""
                }
            ]
        )

        current = refine_msg.content[0].text

    return {"final_output": current}

# Usage
result = self_refine_with_rubric(
    task="Write an API endpoint documentation for a user registration endpoint.",
    rubric={
        "completeness": "All HTTP details (method, path, headers, body, responses) documented",
        "accuracy": "Response codes and error cases are correct",
        "examples": "Includes curl command and response examples",
        "clarity": "Clear, concise language a junior developer can follow",
    }
)
```

---

## 7. 반성(Reflexion)

### 7.1 언어적 강화학습

반성(Reflexion, Shinn et al., 2023)은 과거 시도와 실패의 메모리를 유지하여 자기 개선(Self-Refine)을 넘어갑니다. 수치적 보상 대신 언어적 피드백을 사용하는 강화학습에서 영감을 받았습니다.

사이클:
1. **행동**: 과제 해결을 시도합니다
2. **평가**: 시도가 성공했는지 실패했는지 판별합니다
3. **반성**: 무엇이 잘못되었고 다음에 무엇을 다르게 해야 하는지에 대한 언어적 설명을 생성합니다
4. **재시도**: 메모리에 저장된 반성으로 다시 시도합니다

```python
import anthropic

client = anthropic.Anthropic()

def reflexion_solve(
    task: str,
    evaluator: callable,
    max_attempts: int = 3
) -> dict:
    """Solve a task using the Reflexion pattern.

    Args:
        task: The task description
        evaluator: A function that takes (task, output) and returns
                   {"success": bool, "feedback": str}
        max_attempts: Maximum retry attempts

    Returns:
        Dict with final output, success status, and reflection history
    """

    reflections: list[str] = []

    for attempt in range(max_attempts):
        # Build context with past reflections
        reflection_context = ""
        if reflections:
            reflection_context = "\n\nLessons from previous attempts:\n"
            for i, ref in enumerate(reflections):
                reflection_context += f"\nAttempt {i+1} reflection:\n{ref}\n"

        # Act: Generate solution
        act_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": f"""{task}
{reflection_context}
{"Use the lessons above to avoid repeating mistakes." if reflections else ""}
Provide your solution:"""
                }
            ]
        )

        output = act_msg.content[0].text

        # Evaluate
        eval_result = evaluator(task, output)

        if eval_result["success"]:
            return {
                "output": output,
                "success": True,
                "attempts": attempt + 1,
                "reflections": reflections
            }

        # Reflect: Analyze the failure
        reflect_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": f"""Task: {task}

My attempt:
{output}

Feedback: {eval_result['feedback']}

Reflect on this failure:
1. What specific mistake did I make?
2. Why did I make this mistake?
3. What should I do differently next time?

Be specific and actionable in your reflection."""
                }
            ]
        )

        reflections.append(reflect_msg.content[0].text)

    return {
        "output": output,
        "success": False,
        "attempts": max_attempts,
        "reflections": reflections
    }

# Example evaluator for code generation
def code_evaluator(task: str, output: str) -> dict:
    """Evaluate generated code by attempting to run test cases."""

    # Extract code from the output
    code = output
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]

    try:
        # Execute the code in a safe namespace
        namespace = {}
        exec(code, namespace)

        # Run test cases (example)
        func = namespace.get("solution") or namespace.get("solve")
        if func is None:
            return {"success": False, "feedback": "No function named 'solution' or 'solve' found"}

        # Test cases for the specific problem
        tests = [
            (([2, 7, 11, 15], 9), [0, 1]),
            (([3, 2, 4], 6), [1, 2]),
        ]

        for args, expected in tests:
            result = func(*args)
            if result != expected:
                return {
                    "success": False,
                    "feedback": f"Test failed: input={args}, expected={expected}, got={result}"
                }

        return {"success": True, "feedback": "All tests passed"}

    except Exception as e:
        return {"success": False, "feedback": f"Error executing code: {str(e)}"}

# Usage:
# result = reflexion_solve(
#     task="Write a Python function called 'solution' that takes a list of "
#          "integers and a target sum, and returns the indices of the two "
#          "numbers that add up to the target. Assume exactly one solution exists.",
#     evaluator=code_evaluator
# )
```

### 7.2 외부 도구와 결합한 반성

반성은 구체적 피드백을 제공하는 외부 도구(코드 실행, 웹 검색, 데이터베이스 쿼리)와 결합할 때 특히 강력합니다:

```python
import anthropic
import subprocess
import tempfile

client = anthropic.Anthropic()

def run_code_safely(code: str, timeout: int = 10) -> dict:
    """Execute Python code in a subprocess with a timeout."""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()

        try:
            result = subprocess.run(
                ["python3", f.name],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Code execution timed out after {timeout} seconds",
            }


def reflexion_code_generation(
    task: str,
    test_code: str,
    max_attempts: int = 4
) -> dict:
    """Generate code with Reflexion, using actual code execution for feedback.

    Args:
        task: Description of the function to write
        test_code: Python code that tests the generated function
        max_attempts: Maximum attempts
    """

    reflections = []

    for attempt in range(max_attempts):
        reflection_text = ""
        if reflections:
            reflection_text = "\n\nPrevious attempt reflections:\n" + "\n".join(
                f"Attempt {i+1}: {r}" for i, r in enumerate(reflections)
            )

        # Generate code
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": f"""{task}
{reflection_text}

Output ONLY the Python code, no markdown formatting."""
                }
            ]
        )

        generated_code = msg.content[0].text.strip()
        if generated_code.startswith("```"):
            generated_code = generated_code.split("\n", 1)[1].rsplit("```", 1)[0]

        # Combine generated code with test code
        full_code = generated_code + "\n\n" + test_code

        # Execute
        exec_result = run_code_safely(full_code)

        if exec_result["success"]:
            return {
                "code": generated_code,
                "success": True,
                "attempts": attempt + 1,
                "reflections": reflections
            }

        # Reflect on the failure
        error_info = exec_result["stderr"] or exec_result["stdout"]
        reflect_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": f"""My code:
```python
{generated_code}
```

Error when running tests:
{error_info}

Identify the specific bug and how to fix it. Be concise (2-3 sentences)."""
                }
            ]
        )

        reflections.append(reflect_msg.content[0].text.strip())

    return {
        "code": generated_code,
        "success": False,
        "attempts": max_attempts,
        "reflections": reflections
    }
```

---

## 8. 한 걸음 물러서기 프롬프팅(Step-Back Prompting)

### 8.1 세부 사항 전에 추상화

한 걸음 물러서기 프롬프팅(Step-Back Prompting, Zheng et al., 2023)은 모델에게 구체적 질문에 답하기 전에 먼저 상위 수준의 원칙이나 개념을 고려하도록 요청합니다. "한 걸음 물러남"으로써 모델은 더 관련 있는 지식에 접근하고 표면적 세부 사항에 빠지는 것을 피합니다.

```python
import anthropic

client = anthropic.Anthropic()

def step_back_prompt(question: str) -> dict:
    """Apply step-back prompting: abstract first, then answer.

    Stage 1: Generate a step-back question (higher-level abstraction)
    Stage 2: Answer the step-back question
    Stage 3: Use the abstract answer to inform the specific answer
    """

    # Stage 1: Generate step-back question
    stepback_msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=128,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"""Given this specific question, what is a more general
or abstract question I should answer first to build the right foundation?

Specific question: {question}

Step-back question:"""
            }
        ]
    )
    stepback_question = stepback_msg.content[0].text.strip()

    # Stage 2: Answer the step-back question
    abstract_msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": stepback_question
            }
        ]
    )
    abstract_answer = abstract_msg.content[0].text

    # Stage 3: Answer the original question using the abstract context
    final_msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"""Background knowledge:
{abstract_answer}

Using the background knowledge above, answer this specific question:
{question}"""
            }
        ]
    )

    return {
        "stepback_question": stepback_question,
        "abstract_answer": abstract_answer,
        "final_answer": final_msg.content[0].text
    }

# Examples where step-back helps:
# Specific: "Why does Python's GIL make multithreading slower for CPU-bound tasks?"
# Step-back: "How do operating systems handle concurrent execution and what role
#             do interpreter locks play in dynamic languages?"

# Specific: "Why did the Roman Empire fall?"
# Step-back: "What factors generally cause the decline of large empires throughout history?"
```

### 8.2 기술적 문제 해결을 위한 한 걸음 물러서기

```python
import anthropic

client = anthropic.Anthropic()

# Step-back is powerful for debugging
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """I'm getting a "connection refused" error when my Python app
tries to connect to PostgreSQL in my Docker container.

Before diagnosing this specific error, let me step back and
consider the general principles:

Step-back question: What are all the possible reasons a TCP
connection might be refused between two processes?

After answering the step-back question, apply those principles
to my specific Docker + PostgreSQL scenario."""
        }
    ]
)

# The step-back approach considers:
# General TCP connection refusal causes:
# 1. Target service not running
# 2. Wrong port
# 3. Firewall/security group blocking
# 4. Wrong hostname/IP
# 5. Service not bound to the right interface
#
# Applied to Docker + PostgreSQL:
# - PostgreSQL not started inside container
# - Using localhost instead of container name
# - Port mapping misconfigured (-p 5432:5432)
# - PostgreSQL bound to 127.0.0.1 instead of 0.0.0.0
# - Docker network not properly configured
```

---

## 9. 유추적 프롬프팅(Analogical Prompting)

### 9.1 자기 생성 유추

유추적 프롬프팅(Analogical Prompting, Yasunaga et al., 2023)은 모델에게 실제 문제를 풀기 전에 훈련 데이터에서 관련 예시나 유추를 생성하도록 요청합니다. 수동으로 퓨샷 예시를 제공하는 대신, 모델이 자체적으로 관련 예시를 회상하고 구성합니다.

```python
import anthropic

client = anthropic.Anthropic()

def analogical_prompt(problem: str) -> dict:
    """Apply analogical prompting: recall similar problems, then solve.

    Stage 1: Generate analogous problems and their solutions
    Stage 2: Apply the analogical reasoning to the target problem
    """

    # Stage 1: Recall analogies
    analogy_msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.3,
        messages=[
            {
                "role": "user",
                "content": f"""Before solving this problem, recall 2-3 similar
problems you know about and how they were solved.

Problem: {problem}

For each analogous problem:
1. State the problem
2. Explain the key insight or solution approach
3. Note what makes it similar to the target problem"""
            }
        ]
    )
    analogies = analogy_msg.content[0].text

    # Stage 2: Solve using the analogies
    solution_msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"""Here are analogous problems and their solutions:

{analogies}

Now, using the insights from these analogies, solve the original problem:
{problem}

Explicitly state which analogous solution approach you are adapting and why."""
            }
        ]
    )

    return {
        "analogies": analogies,
        "solution": solution_msg.content[0].text
    }

# Example
result = analogical_prompt(
    "Design a system to detect fraudulent transactions in real-time "
    "from a stream of credit card transactions. The system should "
    "minimize false positives while catching at least 95% of fraud."
)

# The model might recall:
# 1. Email spam filtering (similar binary classification with imbalanced classes)
# 2. Network intrusion detection (similar real-time stream processing)
# 3. Medical diagnosis screening (similar precision-recall tradeoff)
```

### 9.2 구조화된 유추적 추론

```python
import anthropic

client = anthropic.Anthropic()

# Structured analogical prompting with explicit mapping
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """Problem: How should we design the caching layer for our
microservices architecture?

Use analogical reasoning with this structure:

1. RECALL: Think of a well-known system that solved a similar problem.

2. MAP: Create an explicit mapping between the analogy and our problem:
   | Analogy Component | Our System Component |
   |---|---|
   | ... | ... |

3. TRANSFER: Apply the solution from the analogy to our system.

4. ADAPT: Identify where the analogy breaks down and adjust.

5. SOLVE: Provide the final design recommendation."""
        }
    ]
)
```

---

## 10. 추론 기법 비교

### 10.1 의사결정 매트릭스

| 기법 | 최적 용도 | 지연 시간 | 비용 | 정확도 향상 | 복잡도 |
|------|----------|----------|------|-----------|--------|
| 제로샷 CoT | 일반 추론 | 낮음 | 낮음 | 보통 | 매우 낮음 |
| 수동 CoT | 도메인별 추론 | 낮음 | 낮음 | 높음 | 낮음 |
| 자기 일관성 | 고위험 의사결정 | 높음 | 높음 | 높음 | 낮음 |
| 사고의 나무 | 탐색 문제 | 매우 높음 | 매우 높음 | 매우 높음 | 높음 |
| 사고의 그래프 | 다면적 문제 | 매우 높음 | 매우 높음 | 매우 높음 | 매우 높음 |
| 사고의 골격 | 장문 생성 | 보통 | 보통 | 낮음 | 보통 |
| 메타 프롬프팅 | 프롬프트 최적화 | 보통 | 보통 | 다양 | 보통 |
| 자기 개선 | 품질 개선 | 높음 | 높음 | 높음 | 보통 |
| 반성 | 실패 학습 | 높음 | 높음 | 매우 높음 | 높음 |
| 한 걸음 물러서기 | 지식 집약적 QA | 보통 | 보통 | 보통 | 낮음 |
| 유추적 | 새로운 문제 | 보통 | 보통 | 보통 | 낮음 |

### 10.2 선택 가이드

```python
def select_reasoning_technique(
    task_type: str,
    latency_budget_ms: int,
    accuracy_requirement: float,
    cost_sensitivity: str,  # "low", "medium", "high"
    problem_complexity: str  # "simple", "moderate", "complex"
) -> str:
    """Select the best reasoning technique based on constraints.

    Returns the recommended technique name.
    """

    # Rule-based selection (in practice, learn from experience)

    # Simple tasks: avoid over-engineering
    if problem_complexity == "simple":
        return "zero_shot_cot" if task_type == "reasoning" else "direct"

    # Tight latency: avoid multi-call techniques
    if latency_budget_ms < 3000:
        if accuracy_requirement > 0.95:
            return "manual_cot"  # Best single-call accuracy
        return "zero_shot_cot"

    # High accuracy needed, cost not critical
    if accuracy_requirement > 0.95 and cost_sensitivity != "high":
        if task_type in ["math", "logic", "coding"]:
            return "self_consistency"  # Multiple paths + voting
        if task_type == "creative":
            return "tree_of_thoughts"  # Exploration + evaluation
        if task_type == "analysis":
            return "step_back"  # Abstract then specific

    # Complex multi-part problems
    if problem_complexity == "complex":
        if task_type == "coding":
            return "reflexion"  # Learn from execution failures
        if task_type == "writing":
            return "self_refine"  # Iterative improvement
        return "least_to_most"  # Decomposition

    # Moderate complexity, moderate budget
    if accuracy_requirement > 0.8:
        return "manual_cot"  # Good balance of accuracy and cost

    # Default
    return "zero_shot_cot"

# Examples:
print(select_reasoning_technique("math", 10000, 0.99, "low", "complex"))
# -> "self_consistency"

print(select_reasoning_technique("coding", 30000, 0.95, "medium", "complex"))
# -> "reflexion"

print(select_reasoning_technique("reasoning", 2000, 0.8, "high", "simple"))
# -> "zero_shot_cot"
```

### 10.3 기법 결합

가장 강력한 접근법은 여러 기법을 결합합니다:

```python
import anthropic

client = anthropic.Anthropic()

def combined_reasoning_pipeline(
    problem: str,
    difficulty: str = "hard"
) -> dict:
    """Combine multiple reasoning techniques for maximum effectiveness.

    Pipeline:
    1. Step-back: Get abstract principles
    2. Analogical: Recall similar problems
    3. Least-to-most: Decompose into sub-problems
    4. CoT + Self-consistency: Solve each sub-problem with voting
    5. Self-refine: Polish the final answer
    """

    results = {}

    # Step 1: Step-back for abstract principles
    stepback = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"What general principles apply to this type of problem?\n\n{problem}"
            }
        ]
    )
    results["principles"] = stepback.content[0].text

    # Step 2: Analogical recall
    analogies = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        temperature=0.3,
        messages=[
            {
                "role": "user",
                "content": f"Recall 2 analogous problems and their solutions:\n\n{problem}"
            }
        ]
    )
    results["analogies"] = analogies.content[0].text

    # Step 3: Decomposition
    decompose = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"""Using these principles:
{results['principles']}

And these analogies:
{results['analogies']}

Decompose this problem into 3-5 sub-problems (simplest to hardest):
{problem}"""
            }
        ]
    )
    results["decomposition"] = decompose.content[0].text

    # Step 4: Solve with CoT
    solution = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"""Problem: {problem}

Relevant principles: {results['principles']}
Analogous solutions: {results['analogies']}
Sub-problems: {results['decomposition']}

Now solve the problem step by step, addressing each sub-problem in order."""
            }
        ]
    )
    results["solution"] = solution.content[0].text

    # Step 5: Self-refine
    refined = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"""Problem: {problem}

Draft solution:
{results['solution']}

Critique and improve this solution. Check for:
1. Logical errors
2. Missing edge cases
3. Unclear explanations
4. Better alternatives

Provide the improved final solution."""
            }
        ]
    )
    results["refined_solution"] = refined.content[0].text

    return results
```

---

## 연습문제

### 연습문제 1: 사고의 나무 구현

이 문제에 대한 단순화된 사고의 나무 솔버를 구현하세요: "모양을 구부리거나 책갈피가 아닌, 진정으로 참신한 클립 활용법 3가지를 찾으세요." 4개의 초기 아이디어를 생성하고, 각각을 참신성(1-10)으로 평가하고, 6점 미만인 아이디어를 가지치기하고, 생존자를 정교화하세요.

<details><summary>정답 보기</summary>

```python
import anthropic

client = anthropic.Anthropic()

def paperclip_tot() -> dict:
    """Simplified ToT for creative paperclip uses."""

    # Stage 1: Generate diverse initial ideas
    gen_msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        temperature=1.0,
        messages=[
            {
                "role": "user",
                "content": """Generate 4 creative uses for a standard metal paperclip.

Rules:
- NOT bending into shapes or sculptures
- NOT using as a bookmark
- NOT picking locks
- Must be genuinely novel and practical

Number each idea 1-4. Keep each to one sentence."""
            }
        ]
    )
    ideas_text = gen_msg.content[0].text

    # Parse ideas
    ideas = []
    for line in ideas_text.split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            ideas.append(line.split(".", 1)[1].strip() if "." in line[:3] else line)

    # Stage 2: Evaluate each idea
    evaluated = []
    for idea in ideas:
        eval_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=64,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": f"""Rate this paperclip use on NOVELTY (1-10).
10 = never heard before, genuinely creative
5 = somewhat common but interesting
1 = very common, obvious

Idea: "{idea}"

Respond with only the number."""
                }
            ]
        )
        try:
            score = int(eval_msg.content[0].text.strip())
        except ValueError:
            score = 5
        evaluated.append({"idea": idea, "novelty_score": score})

    # Stage 3: Prune (keep score >= 6)
    survivors = [e for e in evaluated if e["novelty_score"] >= 6]
    pruned = [e for e in evaluated if e["novelty_score"] < 6]

    # Stage 4: Elaborate on survivors
    elaborated = []
    for survivor in survivors[:3]:  # Keep top 3
        elab_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": f"""Elaborate on this creative paperclip use:
"{survivor['idea']}"

Provide:
1. How exactly to do it (step by step)
2. What makes this genuinely useful
3. Any materials needed besides the paperclip"""
                }
            ]
        )
        elaborated.append({
            **survivor,
            "elaboration": elab_msg.content[0].text
        })

    return {
        "all_ideas": evaluated,
        "pruned": pruned,
        "final_ideas": elaborated,
        "stats": {
            "generated": len(ideas),
            "survived_pruning": len(survivors),
            "final_count": len(elaborated)
        }
    }

result = paperclip_tot()

print(f"Generated {result['stats']['generated']} ideas")
print(f"Pruned {len(result['pruned'])} (below novelty threshold)")
print(f"Final ideas: {result['stats']['final_count']}")

for idea in result["final_ideas"]:
    print(f"\n--- Idea (novelty: {idea['novelty_score']}/10) ---")
    print(f"Concept: {idea['idea']}")
    print(f"Details: {idea['elaboration'][:200]}...")
```

시연된 핵심 ToT 원칙:
1. **분기**: 여러 후보를 생성 (4개 아이디어)
2. **평가**: 관련 기준으로 각각을 채점 (참신성)
3. **가지치기**: 저품질 분기를 제거 (점수 < 6)
4. **정교화**: 생존한 분기를 추가로 발전

</details>

### 연습문제 2: 코드를 위한 자기 개선

Luhn 알고리즘을 사용하여 신용카드 번호를 검증하는 Python 함수를 생성, 비판, 개선하는 자기 개선 루프를 구현하세요. 4개의 기준이 있는 구체적인 루브릭을 정의하세요. 2번의 개선 반복을 수행하고 품질 향상을 보여주세요.

<details><summary>정답 보기</summary>

```python
import anthropic
import re

client = anthropic.Anthropic()

RUBRIC = {
    "correctness": "Correctly implements the Luhn algorithm (double every second digit from right, sum digits, check divisible by 10)",
    "robustness": "Handles edge cases: spaces in number, dashes, non-numeric input, empty string, too short/long numbers",
    "code_quality": "Type hints, docstring, meaningful variable names, PEP 8 compliant",
    "testing": "Includes at least 3 test cases covering valid, invalid, and edge case inputs",
}

def score_rubric(code: str, rubric: dict[str, str]) -> dict:
    """Score code against a rubric using the LLM."""

    rubric_text = "\n".join(f"- {k}: {v}" for k, v in rubric.items())

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"""Rate this code on each criterion (1-5 scale):

```python
{code}
```

Criteria:
{rubric_text}

For each criterion, provide:
CRITERION_NAME: SCORE/5 - brief feedback

Then on the last line: TOTAL: X/20"""
            }
        ]
    )

    response = msg.content[0].text
    total_match = re.search(r"TOTAL:\s*(\d+)/20", response)
    total = int(total_match.group(1)) if total_match else 10

    return {"score": total, "max_score": 20, "feedback": response}


def self_refine_luhn() -> dict:
    """Self-refine a Luhn algorithm implementation."""

    task = """Write a Python function `validate_card_number(card_number: str) -> bool`
that validates a credit card number using the Luhn algorithm.
Include type hints, docstring, input sanitization, and test cases."""

    # Initial generation
    gen_msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.3,
        messages=[{"role": "user", "content": task}]
    )
    current_code = gen_msg.content[0].text
    history = []

    for iteration in range(2):
        # Score against rubric
        eval_result = score_rubric(current_code, RUBRIC)
        history.append({
            "iteration": iteration,
            "score": eval_result["score"],
            "feedback": eval_result["feedback"]
        })

        print(f"Iteration {iteration}: Score {eval_result['score']}/{eval_result['max_score']}")

        # Refine
        refine_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": f"""{task}

Current version:
{current_code}

Evaluation feedback:
{eval_result['feedback']}

Improve the code to score 5/5 on EVERY criterion.
Output ONLY the improved Python code."""
                }
            ]
        )
        current_code = refine_msg.content[0].text

    # Final scoring
    final_eval = score_rubric(current_code, RUBRIC)
    history.append({
        "iteration": 2,
        "score": final_eval["score"],
        "feedback": final_eval["feedback"]
    })

    print(f"Final: Score {final_eval['score']}/{final_eval['max_score']}")

    return {
        "final_code": current_code,
        "history": history,
        "improvement": history[-1]["score"] - history[0]["score"]
    }

# result = self_refine_luhn()
# print(f"Improvement: +{result['improvement']} points")
# print(f"\nFinal code:\n{result['final_code']}")

# Expected output pattern:
# Iteration 0: Score 13/20
# Iteration 1: Score 17/20
# Final: Score 19/20
# Improvement: +6 points
```

</details>

### 연습문제 3: 한 걸음 물러서기 프롬프팅

이 질문에 한 걸음 물러서기 프롬프팅을 적용하세요: "왜 얼음물에 소금을 추가하면 더 차가워지나요?" 전체 3단계 프로세스(한 걸음 물러서기 질문 생성, 추상 답변, 구체적 답변)를 구현하고 직접 답변과 출력 품질을 비교하세요.

<details><summary>정답 보기</summary>

```python
import anthropic

client = anthropic.Anthropic()

question = "Why does adding salt to ice water make it colder?"

# Approach 1: Direct answer
direct_msg = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=512,
    temperature=0.0,
    messages=[{"role": "user", "content": question}]
)
direct_answer = direct_msg.content[0].text

# Approach 2: Step-back prompting

# Stage 1: Generate step-back question
stepback_msg = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=128,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": f"""What is a more fundamental scientific question I should
answer first to understand this specific question?

Specific question: {question}

Step-back question (one sentence):"""
        }
    ]
)
stepback_q = stepback_msg.content[0].text.strip()
print(f"Step-back question: {stepback_q}")
# Expected: "How do solutes affect the freezing point of solvents,
# and what is the thermodynamics of the dissolution process?"

# Stage 2: Answer the abstract question
abstract_msg = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=512,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": f"""Answer this question thoroughly, covering the
underlying physics and chemistry:

{stepback_q}"""
        }
    ]
)
abstract_answer = abstract_msg.content[0].text

# Stage 3: Apply to specific question
specific_msg = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=512,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": f"""Using this background knowledge:
{abstract_answer}

Now explain specifically: {question}

Be precise about the mechanism and include the relevant equation
(freezing point depression formula)."""
        }
    ]
)
stepback_answer = specific_msg.content[0].text

print("\n=== DIRECT ANSWER ===")
print(direct_answer[:300])
print("\n=== STEP-BACK ANSWER ===")
print(stepback_answer[:300])

# The step-back answer typically:
# 1. Mentions colligative properties explicitly
# 2. Includes the formula: delta_T = K_f * m * i
# 3. Explains the endothermic dissolution process
# 4. Distinguishes between freezing point depression and the
#    endothermic heat absorption (both contribute)
# 5. Is more scientifically precise
#
# The direct answer typically:
# 1. May conflate freezing point depression with cooling
# 2. Often misses the endothermic dissolution contribution
# 3. Less likely to include the quantitative formula
```

</details>

### 연습문제 4: 메타 프롬프팅

다음 과제를 위한 최적화된 프롬프트를 생성하는 메타 프롬프트를 작성하세요: "레스토랑 메뉴 이미지(텍스트로 설명)에서 구조화된 데이터를 추출." 메타 프롬프트는 형식 명세, 예시, 엣지 케이스 처리를 포함한 완전한 프롬프트를 생성해야 합니다. 그런 다음 2개의 테스트 케이스로 생성된 프롬프트를 평가하세요.

<details><summary>정답 보기</summary>

```python
import anthropic

client = anthropic.Anthropic()

# Stage 1: Meta-prompt generates the task prompt
meta_msg = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    temperature=0.3,
    messages=[
        {
            "role": "user",
            "content": """You are a prompt engineering expert. Create an optimized
prompt for extracting structured data from restaurant menu text.

Requirements:
- Input: Text description of a menu (may be OCR output with errors)
- Output: JSON with menu items
- Must handle: prices in various formats ($12, $12.99, 12.99, MKT, "ask server")
- Must handle: items with/without descriptions
- Must handle: section headers (Appetizers, Entrees, etc.)
- Must handle: dietary markers (V, VG, GF, spicy indicators)

Design the prompt with:
1. Clear task description
2. Exact JSON output schema
3. 2 few-shot examples (one simple, one complex)
4. Rules for edge cases

Output the prompt between <PROMPT> and </PROMPT> tags."""
        }
    ]
)

generated_prompt_text = meta_msg.content[0].text

# Extract the prompt between tags
import re
prompt_match = re.search(r"<PROMPT>(.*?)</PROMPT>", generated_prompt_text, re.DOTALL)
generated_prompt = prompt_match.group(1).strip() if prompt_match else generated_prompt_text

print("=== GENERATED PROMPT ===")
print(generated_prompt[:500] + "...")

# Stage 2: Test the generated prompt

test_cases = [
    {
        "input": """STARTERS
Bruschetta (V) - 8.99
Crispy calamari with marinara sauce - $12
Soup of the day - ask server

MAINS
Grilled salmon with lemon butter, served with seasonal vegetables - $24.99
8oz Ribeye steak - MKT
Mushroom risotto (V, GF) - $18""",
        "expected_items": 6,
        "expected_sections": 2
    },
    {
        "input": """Special - Lobster Roll 29
Fish & Chips w/ tartar sauce 16.50
Kids chicken tenders 8
Dessert: Chocolate cake 7  Tiramisu 9""",
        "expected_items": 4,  # or 5 depending on how desserts are parsed
        "expected_sections": 0  # No clear section headers
    }
]

for i, tc in enumerate(test_cases):
    test_prompt = generated_prompt.replace("{{MENU_TEXT}}", tc["input"])
    # If the prompt doesn't use {{MENU_TEXT}}, append the menu text
    if tc["input"] not in test_prompt:
        test_prompt += f"\n\nMenu text to extract:\n{tc['input']}"

    result = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.0,
        messages=[{"role": "user", "content": test_prompt}]
    )

    print(f"\n=== TEST CASE {i+1} ===")
    print(f"Input: {tc['input'][:100]}...")
    print(f"Output: {result.content[0].text[:300]}...")

    # Validate output is valid JSON
    import json
    try:
        parsed = json.loads(result.content[0].text)
        items = parsed.get("items", parsed.get("menu_items", []))
        print(f"Parsed {len(items)} items (expected ~{tc['expected_items']})")
        print("Valid JSON output")
    except json.JSONDecodeError:
        print("WARNING: Output is not valid JSON — prompt needs refinement")
```

</details>

### 연습문제 5: 기법 선택

다음 5개 시나리오 각각에 대해 최선의 고급 추론 기법을 선택하고 해당 기법을 시작하는 첫 번째 프롬프트 호출을 작성하세요. 한 문장으로 선택을 정당화하세요.

1. 정렬 알고리즘에 대한 포괄적 테스트 모음 생성
2. 장기 실행 Python 서비스의 메모리 누수 디버깅
3. 재생 에너지에 대한 설득력 있는 에세이 작성
4. 복잡한 시스템 설계 면접 질문 풀기
5. 영어에서 스페인어로 법률 계약서를 도메인 정확도로 번역

<details><summary>정답 보기</summary>

**1. 정렬 알고리즘에 대한 테스트 모음 생성**
**기법**: 사고의 골격(Skeleton-of-Thought, SoT)
**정당화**: 테스트 카테고리(엣지 케이스, 성능, 안정성 등)는 독립적이며 테스트 카테고리를 개요한 후 병렬로 생성할 수 있습니다.

```python
import anthropic
client = anthropic.Anthropic()

msg = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """I need to generate a comprehensive test suite for a sorting algorithm.

List 5-7 test categories as a skeleton outline (one line each).
Do not write tests yet — just list the categories.

Example categories: edge cases, performance, stability, ...

Skeleton:"""
        }
    ]
)
# Then expand each category in parallel
```

**2. 메모리 누수 디버깅**
**기법**: 한 걸음 물러서기 프롬프팅(Step-Back Prompting)
**정당화**: Python의 메모리 누수의 일반적 원인을 먼저 고려하면 세부 사항에 들어가기 전에 체계적인 진단 프레임워크를 제공합니다.

```python
msg = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """Before debugging this specific memory leak, let me step back.

Step-back question: What are all the common causes of memory leaks
in long-running Python services, and what diagnostic tools exist
for each cause?

Answer the step-back question comprehensively."""
        }
    ]
)
```

**3. 설득력 있는 에세이 작성**
**기법**: 자기 개선(Self-Refine)
**정당화**: 설득력 있는 글쓰기는 반복적 개선으로부터 이점을 얻습니다 — 비판 단계에서 논증 강도, 논리적 흐름, 반론 처리, 수사적 효과를 평가할 수 있습니다.

```python
msg = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    temperature=0.5,
    messages=[
        {
            "role": "user",
            "content": """Write a persuasive essay (500 words) arguing that
governments should increase investment in renewable energy.

Include:
- A compelling opening hook
- 3 main arguments with evidence
- Acknowledgment and rebuttal of one counterargument
- A call to action"""
        }
    ]
)
# Then critique on: argument strength, evidence quality, rhetorical devices, flow
# Then refine based on critique
```

**4. 시스템 설계 면접 질문**
**기법**: 사고의 나무(Tree of Thoughts, ToT)
**정당화**: 시스템 설계는 여러 유효한 아키텍처가 있습니다. ToT는 다른 접근법(모놀리스, 마이크로서비스, 서버리스)을 탐색하고, 요구 사항에 대해 각각을 평가하며, 최적의 것을 선택합니다.

```python
msg = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=512,
    temperature=0.7,
    messages=[
        {
            "role": "user",
            "content": """Design a URL shortening service like bit.ly that handles
10,000 URLs per second.

Generate 3 fundamentally different architectural approaches:

Approach 1: [name and 3-sentence description]
Approach 2: [name and 3-sentence description]
Approach 3: [name and 3-sentence description]

For each, rate: scalability (1-10), simplicity (1-10), cost (1-10)."""
        }
    ]
)
# Then develop the highest-rated approach further
```

**5. 법률 계약서 번역**
**기법**: 반성(Reflexion)
**정당화**: 법률 번역은 특정 법률 용어 데이터베이스에 대해 검증할 수 있는 도메인 정확도가 필요합니다. 반성은 알려진 올바른 용어에 대해 번역을 테스트하고 오류로부터 학습할 수 있습니다.

```python
msg = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """Translate this contract clause from English to Spanish.
Use formal legal Spanish (Spain, not Latin America).

Clause: "The Licensor hereby grants to the Licensee a non-exclusive,
non-transferable, revocable license to use the Software for the
purpose of internal business operations, subject to the terms and
conditions set forth herein."

Ensure these legal terms are translated using standard legal equivalents:
- Licensor = El Licenciante
- Licensee = El Licenciatario
- non-exclusive = no exclusiva
- revocable = revocable

Provide the translation:"""
        }
    ]
)
# Then verify against legal terminology, reflect on errors, retry
```

</details>

---

**이전**: [사고의 연쇄](./03_Chain_of_Thought.md) | **다음**: [구조화된 출력과 포매팅](./05_Structured_Output_Prompting.md)

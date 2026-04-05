# 03. 사고의 연쇄 프롬프팅(Chain-of-Thought Prompting)

**이전**: [제로샷과 퓨샷](./02_Zero_Shot_and_Few_Shot.md) | **다음**: [고급 추론 프롬프트](./04_Advanced_Reasoning_Prompts.md)

## 학습 목표

- 사고의 연쇄(Chain-of-Thought) 프롬프팅이 대규모 언어 모델에서 추론을 향상시키는 이유를 설명한다
- 다양한 추론 과제에 대해 제로샷(Zero-shot)과 수동(Manual) CoT 프롬프팅을 구현한다
- 다수결 투표를 통해 답변 신뢰성을 향상시키기 위해 CoT와 자기 일관성(Self-consistency)을 적용한다
- 최소에서 최대(Least-to-Most) 프롬프팅을 사용하여 복잡한 문제를 풀 수 있는 하위 문제로 분해한다
- CoT가 성능을 향상시키는 경우와 저하시키는 경우를 평가하고 적절한 기법을 선택한다

---

사고의 연쇄(Chain-of-Thought, CoT) 프롬프팅은 프롬프트 엔지니어링에서 가장 영향력 있는 발견 중 하나입니다. 모델에게 직접 답을 생성하도록 요청하는 대신, CoT는 결론에 도달하기 전에 중간 추론 단계를 생성하도록 모델을 유도합니다. 이 겉보기에 단순한 변화 — "풀이 과정을 보여줘" — 는 논리, 산술, 다단계 추론, 상식적 추론이 필요한 과제에서 성능을 극적으로 향상시킵니다. 이 레슨에서는 CoT와 그 확장의 이론, 변형, 실용적 응용을 다룹니다.

## 목차

1. [사고의 연쇄 기초](#1-사고의-연쇄-기초)
2. [CoT가 작동하는 이유](#2-cot가-작동하는-이유)
3. [제로샷 CoT](#3-제로샷-cot)
4. [시연을 포함한 수동 CoT](#4-시연을-포함한-수동-cot)
5. [자동 CoT(Auto-CoT)](#5-자동-cotauto-cot)
6. [CoT가 도움이 되는 경우 vs 해가 되는 경우](#6-cot가-도움이-되는-경우-vs-해가-되는-경우)
7. [CoT와 자기 일관성](#7-cot와-자기-일관성)
8. [최소에서 최대 프롬프팅](#8-최소에서-최대-프롬프팅)
9. [사고의 프로그램(Program-of-Thought)](#9-사고의-프로그램program-of-thought)
10. [CoT를 활용한 수학적 추론](#10-cot를-활용한-수학적-추론)
11. [연습문제](#연습문제)

---

## 1. 사고의 연쇄 기초

### 1.1 핵심 아이디어

표준 프롬프팅은 모델에게 질문에서 답으로 직접 가도록 요청합니다:

```
Q: If a store has 3 shelves with 8 books each, and 5 books are removed, how many remain?
A: 19
```

사고의 연쇄 프롬프팅은 모델에게 추론을 보여달라고 요청합니다:

```
Q: If a store has 3 shelves with 8 books each, and 5 books are removed, how many remain?
A: The store starts with 3 shelves × 8 books = 24 books total.
   After removing 5 books: 24 - 5 = 19 books remain.
   The answer is 19.
```

둘 다 같은 답을 만들지만, CoT 버전은 각 추론 단계가 검증 가능한 중간 결과 위에 구축되기 때문에 더 어려운 문제에서 더 신뢰할 수 있습니다.

### 1.2 기본 CoT 구현

```python
import anthropic

client = anthropic.Anthropic()

# Standard prompting (direct answer)
standard_response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """A train leaves Station A at 9:00 AM traveling at 60 mph.
Another train leaves Station B (which is 210 miles away) at 10:00 AM
traveling toward Station A at 80 mph.
At what time do they meet?

Answer:"""
        }
    ]
)

# Chain-of-Thought prompting (reasoning steps)
cot_response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=512,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """A train leaves Station A at 9:00 AM traveling at 60 mph.
Another train leaves Station B (which is 210 miles away) at 10:00 AM
traveling toward Station A at 80 mph.
At what time do they meet?

Let's solve this step by step:"""
        }
    ]
)

# The CoT version will explicitly work through:
# Step 1: By 10:00 AM, Train A has traveled 60 miles (1 hour × 60 mph)
# Step 2: Remaining distance: 210 - 60 = 150 miles
# Step 3: Combined speed: 60 + 80 = 140 mph
# Step 4: Time to meet: 150 / 140 ≈ 1.07 hours ≈ 1 hour 4.3 minutes
# Step 5: Meeting time: 10:00 AM + 1h 4.3min ≈ 11:04 AM
```

### 1.3 사고의 연쇄의 해부학

잘 형성된 사고의 연쇄는 다음과 같은 속성을 가집니다:

1. **분해**: 문제가 더 작은 하위 문제로 분해됩니다
2. **순차적 의존성**: 각 단계가 이전 단계의 결과를 사용합니다
3. **명시적 계산**: 수학적 연산이 단계별로 수행됩니다
4. **중간 결론**: 각 단계가 명확한 결과로 끝납니다
5. **최종 종합**: 단계들이 최종 답으로 결합됩니다

```python
import anthropic

client = anthropic.Anthropic()

# Structuring CoT with explicit step markers
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """A company has 150 employees. 40% work in engineering,
30% in sales, and the rest in operations. Engineering is getting a 10%
headcount increase, sales is losing 5 people, and operations stays the same.
How many total employees will there be?

Solve step by step:

Step 1: Calculate current department sizes
Step 2: Apply changes to each department
Step 3: Sum to get the new total"""
        }
    ]
)
```

---

## 2. CoT가 작동하는 이유

### 2.1 계산적 논증

LLM은 본질적으로 일정 깊이 회로입니다 — 고정된 수의 레이어를 통해 각 토큰을 처리합니다. CoT 없이는 모델이 단일 순방향 패스에서 전체 문제를 풀어야 하므로, 풀 수 있는 문제의 계산 복잡도가 제한됩니다.

CoT는 모델에게 중간 결과를 기록하도록 허용하여 효과적으로 더 많은 "계산 시간"을 제공합니다. 생성된 각 토큰은 모델에 입력으로 되돌려 추가 처리 단계를 제공합니다. 본질적으로, CoT는 모델을 고정 깊이 회로에서 가변 깊이 회로로 변환합니다.

```python
# Without CoT: One forward pass must solve everything
# Input: "What is 37 × 24?" -> Model must compute in ~100 layers
# This is like asking someone to solve 37 × 24 in their head

# With CoT: Multiple forward passes, each building on the last
# "37 × 24"
# -> "37 × 20 = 740" (one forward pass, simple multiplication)
# -> "37 × 4 = 148" (another forward pass, simple multiplication)
# -> "740 + 148 = 888" (another forward pass, addition)
# This is like giving someone paper to write intermediate steps
```

### 2.2 창발적 추론(Emergent Reasoning)

CoT는 **창발적 능력(Emergent Capability)**입니다 — 특정 크기 임계값(대략 100B+ 매개변수, 더 나은 훈련으로 이 임계값이 낮아졌지만) 이상의 모델에서만 잘 작동합니다. 소규모 모델에서 CoT를 시도하면 그럴듯하게 들리지만 부정확한 추론 체인을 생성하는 경우가 많습니다.

```python
import anthropic

client = anthropic.Anthropic()

# CoT excels at tasks requiring multi-step reasoning
# Here's an example that requires logical deduction

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """Three friends (Alice, Bob, Carol) each own a different pet
(cat, dog, fish) and live on different floors (1, 2, 3).

Clues:
1. Alice does not live on floor 1.
2. The cat owner lives on a higher floor than Bob.
3. Carol does not own the fish.
4. The dog owner lives on floor 1.

Who owns which pet and lives on which floor?

Think through this step by step, using each clue to eliminate possibilities."""
        }
    ]
)

# Without CoT, the model might guess randomly
# With CoT, it systematically applies constraints:
# From clue 4: Dog owner is on floor 1
# From clue 1: Alice is not on floor 1, so Alice doesn't own the dog
# From clue 2: Cat owner > Bob's floor, so Bob doesn't own the cat
# etc.
```

### 2.3 추론의 충실도(Faithfulness of Reasoning)

중요한 주의사항: 모델이 생성하는 추론 체인이 반드시 모델이 내부적으로 수행하는 실제 계산인 것은 아닙니다. 연구에 따르면 모델은 때때로 잘못된 이유로 올바른 답에 도달하거나, 내부 처리를 실제로 반영하지 않는 설득력 있는 추론 체인을 생성할 수 있습니다.

이것은 다음을 의미합니다:
- CoT는 정확도를 향상시키지만, 설명이 완전히 신뢰할 수 있는 것은 아닙니다
- 외부 수단을 통한 최종 답변 검증이 여전히 중요합니다
- 추론의 품질은 모델과 과제 유형에 따라 다릅니다

---

## 3. 제로샷 CoT

### 3.1 마법의 문구

CoT의 가장 단순한 형태는 Kojima et al. (2022)이 발견한 제로샷 CoT입니다. 프롬프트에 "Let's think step by step"을 추가하는 것만으로 예시 없이 추론 정확도를 크게 향상시킵니다.

```python
import anthropic

client = anthropic.Anthropic()

def zero_shot_cot(question: str) -> str:
    """Apply zero-shot CoT by appending the trigger phrase."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"{question}\n\nLet's think step by step."
            }
        ]
    )

    return message.content[0].text

# Examples where zero-shot CoT helps
questions = [
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "If you have 3 apples and take away 2, how many do you have?",
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {zero_shot_cot(q)}")
    print("---")
```

### 3.2 트리거 문구의 변형

다른 트리거 문구가 다른 과제에 더 잘 작동할 수 있습니다:

```python
import anthropic

client = anthropic.Anthropic()

# General reasoning
general = "Let's think step by step."

# Mathematical problems
math = "Let's solve this mathematically, showing each calculation."

# Logical deduction
logic = "Let's work through this logically, considering each constraint."

# Code/algorithm problems
code = "Let's trace through this algorithm step by step."

# Analytical/comparison tasks
analysis = "Let's analyze this systematically, considering each factor."

# Decision-making
decision = "Let's weigh the pros and cons step by step."

# Example: Using a domain-specific trigger
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """Is the following argument logically valid?

Premise 1: All mammals are warm-blooded.
Premise 2: All whales are mammals.
Premise 3: Some marine animals are whales.
Conclusion: Some marine animals are warm-blooded.

Let's evaluate this using formal logic, checking each inference step."""
        }
    ]
)
```

### 3.3 2단계 제로샷 CoT

최대 정확도를 위해 2단계 접근법을 사용합니다: 먼저 추론을 생성하고, 그 다음 답을 추출합니다.

```python
import anthropic

client = anthropic.Anthropic()

def two_stage_cot(question: str) -> dict:
    """Two-stage CoT: generate reasoning, then extract the answer.

    Stage 1: Generate the full reasoning chain
    Stage 2: Extract just the final answer from the reasoning
    """

    # Stage 1: Reasoning
    reasoning_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"{question}\n\nLet's think step by step."
            }
        ]
    )

    reasoning = reasoning_response.content[0].text

    # Stage 2: Answer extraction
    answer_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=64,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"""{question}

{reasoning}

Based on the reasoning above, what is the final answer?
Respond with ONLY the answer, no explanation."""
            }
        ]
    )

    return {
        "reasoning": reasoning,
        "answer": answer_response.content[0].text.strip()
    }

# Example
result = two_stage_cot(
    "A farmer has 15 sheep. All but 8 die. How many are left?"
)
print(f"Reasoning: {result['reasoning']}")
print(f"Answer: {result['answer']}")
# Answer: 8 (the common trick question — "all but 8" means 8 survive)
```

---

## 4. 시연을 포함한 수동 CoT

### 4.1 추론 예시 제공

수동 CoT는 퓨샷(Few-shot) 프롬프팅과 사고의 연쇄를 결합합니다. 입력과 출력뿐만 아니라 완전한 추론 과정을 포함하는 예시를 제공합니다.

```python
import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """Solve the following word problems by thinking step by step.

Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls.
He bought 2 cans × 3 balls per can = 6 new balls.
Total: 5 + 6 = 11 tennis balls.
The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and
bought 6 more, how many apples do they have?
A: The cafeteria started with 23 apples.
They used 20 apples for lunch: 23 - 20 = 3 apples remaining.
They bought 6 more: 3 + 6 = 9 apples.
The answer is 9.

Q: There are 15 trees in the grove. Grove workers will plant trees
today. After they are done, there will be 21 trees. How many trees
did the grove workers plant today?
A:"""
        }
    ]
)

print(message.content[0].text)
```

### 4.2 효과적인 시연 작성

CoT 시연의 품질이 출력 품질에 직접 영향을 미칩니다. 효과적인 시연을 작성하기 위한 원칙은 다음과 같습니다:

```python
import anthropic

client = anthropic.Anthropic()

# Principle 1: Make reasoning granular (one operation per step)
# BAD: "5 × 3 + 2 × 4 = 15 + 8 = 23"
# GOOD: Show each multiplication separately, then add

# Principle 2: Use natural language, not just equations
# BAD: "15 - 8 = 7, 7 × 2 = 14"
# GOOD: "There are 15 items. Removing 8 leaves 15 - 8 = 7.
#        Doubling gives 7 × 2 = 14."

# Principle 3: State intermediate conclusions explicitly
# BAD: "Calculating further..."
# GOOD: "So we now know that the car has traveled 120 miles."

# Principle 4: Match the complexity level of your demonstrations
# to the target problem (don't use trivial demos for hard problems)

# Good demonstration with all principles applied
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """Solve multi-step percentage problems by reasoning step by step.

Q: A shirt originally costs $80. It's on sale for 25% off. You also have
a coupon for 10% off the sale price. What is the final price?
A: Let me work through this step by step.
Step 1: Calculate the sale discount.
   25% of $80 = 0.25 × 80 = $20 discount
Step 2: Calculate the sale price.
   Sale price = $80 - $20 = $60
Step 3: Apply the coupon to the sale price (not the original price).
   10% of $60 = 0.10 × 60 = $6 coupon discount
Step 4: Calculate the final price.
   Final price = $60 - $6 = $54
The answer is $54.

Q: A restaurant bill is $85. You want to tip 20% on the pre-tax amount.
Tax is 8.5%. What is the total you pay (bill + tax + tip)?
A:"""
        }
    ]
)
```

### 4.3 다중 도메인 CoT 템플릿

다른 추론 도메인은 다른 CoT 구조를 필요로 합니다:

```python
import anthropic

client = anthropic.Anthropic()

# Domain: Causal Reasoning
causal_cot = """Q: A company's website went down. They recently deployed a new version
and also had a power outage. What is the most likely cause?
A: Let me reason about the causal chain.
Step 1: Identify the event timeline.
   - New version was deployed (recent change)
   - Power outage occurred (external event)
   - Website went down (the observed problem)
Step 2: Evaluate each potential cause.
   - New deployment: Could introduce bugs, misconfigurations, or resource issues.
     This is a common cause of outages and is directly related to the website.
   - Power outage: Could affect servers, but modern hosting uses redundant power.
     If the site is cloud-hosted, a local power outage wouldn't affect it.
Step 3: Apply Occam's razor.
   - The deployment is the most direct and likely cause because it's a change
     to the system itself, and most outages correlate with recent changes.
The most likely cause is the new deployment."""

# Domain: Ethical Reasoning
ethical_cot = """Q: Should a self-driving car prioritize passengers or pedestrians in
an unavoidable accident?
A: Let me analyze this ethical dilemma systematically.
Step 1: Identify the ethical frameworks.
   - Utilitarian: Minimize total harm (save the larger group)
   - Deontological: Follow rules (do not actively cause harm)
   - Contractarian: What rules would rational people agree to?
Step 2: Apply each framework.
   - Utilitarian: Depends on numbers — save whichever group is larger.
   - Deontological: There's a moral difference between action and inaction.
     Swerving to hit a pedestrian is an action; maintaining course is inaction.
   - Contractarian: People would want rules that minimize their overall risk.
Step 3: Consider practical implications.
   - If cars don't protect passengers, fewer people will buy them, reducing
     the overall safety benefit of autonomous vehicles.
Step 4: Synthesize.
   There is no universal answer. However, most ethicists and regulators
   lean toward minimizing total casualties while avoiding active targeting."""
```

---

## 5. 자동 CoT(Auto-CoT)

### 5.1 개념

Auto-CoT (Zhang et al., 2022)는 CoT 시연의 생성을 자동화합니다. 추론 체인을 수동으로 작성하는 대신, 다양한 질문 집합에 대해 모델이 직접 생성하도록 하고, 이 생성된 체인을 퓨샷 예시로 사용합니다.

```python
import anthropic
from typing import Any

client = anthropic.Anthropic()

def generate_auto_cot_examples(
    questions: list[str],
    num_clusters: int = 5
) -> list[dict[str, str]]:
    """Generate CoT demonstrations automatically.

    Steps:
    1. Cluster questions by similarity
    2. Select one representative question per cluster
    3. Generate a CoT answer for each representative
    4. Use these as few-shot demonstrations

    Args:
        questions: Pool of unlabeled questions
        num_clusters: Number of diverse examples to generate

    Returns:
        List of {"question": ..., "reasoning": ...} dicts
    """

    # Step 1: Select diverse questions
    # (In practice, use clustering on embeddings;
    # here we simulate with simple heuristics)
    selected_questions = questions[:num_clusters]

    # Step 2: Generate CoT reasoning for each
    demonstrations = []
    for q in selected_questions:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": f"{q}\n\nLet's think step by step."
                }
            ]
        )

        demonstrations.append({
            "question": q,
            "reasoning": message.content[0].text
        })

    return demonstrations

def use_auto_cot(
    query: str,
    demonstrations: list[dict[str, str]]
) -> str:
    """Apply Auto-CoT demonstrations to a new query."""

    demo_text = "\n\n".join(
        f"Q: {d['question']}\nA: {d['reasoning']}"
        for d in demonstrations
    )

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"""{demo_text}

Q: {query}
A: Let's think step by step."""
            }
        ]
    )

    return message.content[0].text
```

### 5.2 Auto-CoT에서의 다양성

Auto-CoT의 핵심 통찰은 다양한 시연이 무작위 시연보다 더 효과적이라는 것입니다. 클러스터링은 시연이 다른 추론 패턴을 커버하도록 보장합니다.

```python
import numpy as np
from typing import Any

def cluster_questions(
    questions: list[str],
    embeddings: list[list[float]],
    k: int = 5
) -> list[int]:
    """Simple k-means-style clustering to select diverse questions.

    Returns indices of the most central question in each cluster.
    """
    # Convert to numpy
    embedding_matrix = np.array(embeddings)

    # Simple k-means (in production, use sklearn.cluster.KMeans)
    n = len(questions)
    # Initialize centroids with random selection
    rng = np.random.default_rng(42)
    centroid_indices = rng.choice(n, size=k, replace=False)
    centroids = embedding_matrix[centroid_indices]

    for _ in range(20):  # iterations
        # Assign clusters
        distances = np.linalg.norm(
            embedding_matrix[:, None, :] - centroids[None, :, :],
            axis=2
        )
        assignments = np.argmin(distances, axis=1)

        # Update centroids
        for i in range(k):
            mask = assignments == i
            if mask.any():
                centroids[i] = embedding_matrix[mask].mean(axis=0)

    # Select the question closest to each centroid
    selected_indices = []
    for i in range(k):
        mask = assignments == i
        cluster_indices = np.where(mask)[0]
        if len(cluster_indices) > 0:
            cluster_embeddings = embedding_matrix[cluster_indices]
            dists = np.linalg.norm(cluster_embeddings - centroids[i], axis=1)
            best_local_idx = np.argmin(dists)
            selected_indices.append(cluster_indices[best_local_idx])

    return selected_indices
```

---

## 6. CoT가 도움이 되는 경우 vs 해가 되는 경우

### 6.1 CoT가 도움이 되는 과제

CoT는 다음에서 일관되게 성능을 향상시킵니다:

| 과제 유형 | 예시 | CoT가 도움이 되는 이유 |
|-----------|------|----------------------|
| 산술 | 다단계 계산 | 계산을 외부화 |
| 논리 퍼즐 | 제약 충족 | 체계적 소거 |
| 문장제 문제 | 스토리 기반 수학 | 이해와 계산을 분리 |
| 상식 추론 | "Y이면 X가 일어날까?" | 암묵적 지식을 명시적으로 |
| 코드 추론 | "이 코드의 출력은?" | 실행을 시뮬레이션 |
| 다중 단계 QA | 사실 연결이 필요한 질문 | 추론을 명시적으로 연결 |

```python
import anthropic

client = anthropic.Anthropic()

# Multi-hop question answering — CoT is essential
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=512,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """Based on these facts:
- The CEO of TechCorp is Jane Smith
- TechCorp acquired DataFlow in 2023
- DataFlow's main product is StreamDB
- StreamDB uses PostgreSQL as its backend

Question: What database does the company led by Jane Smith's acquired
subsidiary use?

Let's trace through the chain of facts step by step."""
        }
    ]
)
# CoT traces: Jane Smith -> CEO of TechCorp -> acquired DataFlow ->
# DataFlow makes StreamDB -> StreamDB uses PostgreSQL
# Answer: PostgreSQL
```

### 6.2 CoT가 해가 되는 과제

CoT는 실제로 다음에서 성능을 저하시킬 수 있습니다:

| 과제 유형 | 예시 | CoT가 해가 되는 이유 |
|-----------|------|---------------------|
| 단순 검색 | "프랑스의 수도는?" | 사소한 답에 과도한 사고 |
| 패턴 매칭 | 단순 분류 | 추론이 노이즈를 추가 |
| 창의적 글쓰기 | 시, 이야기 | 분석적 사고가 창의성을 제약 |
| 빠른 사실 답변 | "X는 몇 년에 태어났나?" | 불필요한 장황함 |
| 낮은 복잡도 과제 | 예/아니오 질문 | 추론이 오류를 도입할 수 있음 |

```python
import anthropic

client = anthropic.Anthropic()

# Example where CoT is counterproductive
# This is a simple factual retrieval — CoT adds noise

# BAD: Forcing CoT on a simple task
bad_response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=512,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """What is the chemical symbol for gold?

Let's think step by step."""
        }
    ]
)
# The model will produce several unnecessary paragraphs before saying "Au"

# GOOD: Direct answer for a simple task
good_response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=16,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": "What is the chemical symbol for gold?"
        }
    ]
)
# The model directly answers "Au"
```

### 6.3 모델 크기 의존성

CoT의 효과는 모델 크기에 따라 달라집니다:

```python
# Rule of thumb for CoT effectiveness:
# - Very small models (<7B params): CoT often produces incorrect reasoning
#   that leads to wrong answers. Direct prompting may be better.
#
# - Medium models (7B-70B params): CoT helps on some tasks but can
#   produce plausible-sounding incorrect reasoning.
#
# - Large models (70B+ params): CoT consistently helps on reasoning tasks.
#   The model produces genuine, useful reasoning chains.
#
# - Frontier models (Claude, GPT-4): CoT is highly effective and the
#   model can even recognize when CoT is unnecessary.

# Practical advice:
# 1. Always test with and without CoT on your specific task
# 2. If using a smaller model, verify the reasoning chain manually
# 3. For production, use two-stage CoT (reasoning + answer extraction)
#    to prevent reasoning errors from corrupting the final output
```

---

## 7. CoT와 자기 일관성

### 7.1 자기 일관성 접근법

자기 일관성(Self-consistency, Wang et al., 2022)은 동일한 답에 대해 여러 유효한 추론 경로가 있을 수 있음을 인식합니다. 단일 그리디 CoT 체인에 의존하는 대신:

1. 여러 추론 체인을 생성합니다 (온도 > 0)
2. 각 체인에서 답을 추출합니다
3. 다수결 투표를 최종 답으로 채택합니다

```python
import anthropic
from collections import Counter

client = anthropic.Anthropic()

def self_consistent_cot(
    question: str,
    num_paths: int = 5,
    temperature: float = 0.7
) -> dict:
    """Generate multiple CoT reasoning paths and take a majority vote.

    Args:
        question: The question to answer
        num_paths: Number of independent reasoning chains to generate
        temperature: Higher = more diverse reasoning paths

    Returns:
        Dict with 'answer', 'confidence', and 'all_answers'
    """

    answers = []

    for _ in range(num_paths):
        # Stage 1: Generate reasoning
        reasoning_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": f"{question}\n\nLet's think step by step."
                }
            ]
        )
        reasoning = reasoning_msg.content[0].text

        # Stage 2: Extract answer (deterministic)
        answer_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=32,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": f"""{question}

{reasoning}

What is the final numerical answer? Reply with only the number."""
                }
            ]
        )
        answer = answer_msg.content[0].text.strip()
        answers.append(answer)

    # Majority vote
    vote_counts = Counter(answers)
    majority_answer = vote_counts.most_common(1)[0][0]
    confidence = vote_counts[majority_answer] / num_paths

    return {
        "answer": majority_answer,
        "confidence": confidence,
        "all_answers": answers,
        "vote_distribution": dict(vote_counts)
    }

# Example
result = self_consistent_cot(
    "If a ball is thrown upward at 20 m/s, and gravity is 10 m/s², "
    "what is the maximum height reached?",
    num_paths=5
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.0%}")
print(f"All paths: {result['all_answers']}")
print(f"Votes: {result['vote_distribution']}")
```

### 7.2 신뢰도 추정

자기 일관성은 자연스러운 신뢰도 추정을 제공합니다: 답에 동의하는 추론 경로의 비율입니다.

```python
def self_consistent_with_threshold(
    question: str,
    confidence_threshold: float = 0.6,
    initial_paths: int = 5,
    max_paths: int = 15
) -> dict:
    """Adaptive self-consistency: generate more paths if confidence is low."""

    all_answers = []
    num_paths = initial_paths

    while num_paths <= max_paths:
        # Generate additional paths (only the delta)
        new_answers_needed = num_paths - len(all_answers)
        for _ in range(new_answers_needed):
            # (Generate reasoning and extract answer as in previous example)
            # Placeholder for answer extraction
            answer = "placeholder"
            all_answers.append(answer)

        vote_counts = Counter(all_answers)
        majority_answer = vote_counts.most_common(1)[0][0]
        confidence = vote_counts[majority_answer] / len(all_answers)

        if confidence >= confidence_threshold:
            return {
                "answer": majority_answer,
                "confidence": confidence,
                "paths_used": len(all_answers),
                "status": "confident"
            }

        # Need more paths
        num_paths += 5

    # Max paths reached without achieving confidence threshold
    vote_counts = Counter(all_answers)
    majority_answer = vote_counts.most_common(1)[0][0]
    return {
        "answer": majority_answer,
        "confidence": vote_counts[majority_answer] / len(all_answers),
        "paths_used": len(all_answers),
        "status": "low_confidence"
    }
```

### 7.3 자기 일관성을 사용할 때

자기 일관성은 다음과 같은 경우에 가장 가치가 있습니다:
- 과제에 단일 정답이 있는 경우 (개방형 생성이 아님)
- 다른 추론 접근법이 다른 결과를 만들 수 있는 경우
- 높은 정확도가 지연 시간이나 비용보다 더 중요한 경우
- 답변 공간이 이산적인 경우 (숫자, 레이블, 예/아니오)

```python
# Cost analysis: self-consistency uses N times the API calls
#
# Standard CoT: 1 call × $0.003 per call = $0.003
# Self-consistency (N=5): 10 calls × $0.003 = $0.030
# (5 reasoning + 5 extraction = 10 calls)
#
# Use self-consistency when the cost of being wrong exceeds
# the cost of additional API calls. For example:
# - Medical triage (high stakes) -> N=10+ is justified
# - Customer email classification (low stakes) -> N=1 is fine
# - Financial calculations (medium stakes) -> N=3-5 is reasonable
```

---

## 8. 최소에서 최대 프롬프팅

### 8.1 분해 전략

최소에서 최대(Least-to-Most) 프롬프팅 (Zhou et al., 2022)은 복잡한 문제를 명시적으로 더 단순한 하위 문제로 분해하고, 각 하위 문제를 순서대로 풀며, 이전 해를 사용하여 이후 문제를 푸는 방식으로 해결합니다.

표준 CoT와 다른 점은 분해 단계가 명시적이며 모델에 먼저 하위 문제를 식별하도록 요청한다는 것입니다.

```python
import anthropic

client = anthropic.Anthropic()

# Standard CoT might get confused by a complex multi-part problem
# Least-to-most explicitly decomposes first

# Stage 1: Decomposition
decomposition = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=512,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """I want to solve this problem:
"A car rental company charges $40 per day plus $0.25 per mile.
If a customer rents a car for 3 days and drives 200 miles,
and there's a 15% surcharge for drivers under 25, and the customer
is 23 years old, what is the total cost including 8% sales tax?"

To solve this, what sub-problems do I need to solve first?
List them from simplest to most complex."""
        }
    ]
)

# Expected decomposition:
# 1. Calculate the base daily charge (3 days × $40)
# 2. Calculate the mileage charge (200 miles × $0.25)
# 3. Calculate the subtotal (daily + mileage)
# 4. Apply the under-25 surcharge (subtotal × 1.15)
# 5. Apply sales tax (surcharge total × 1.08)
```

### 8.2 전체 최소에서 최대 파이프라인

```python
import anthropic

client = anthropic.Anthropic()

def least_to_most_solve(problem: str) -> dict:
    """Solve a complex problem using least-to-most decomposition.

    Stage 1: Decompose the problem into ordered sub-problems
    Stage 2: Solve each sub-problem sequentially, building on prior solutions
    Stage 3: Synthesize the final answer
    """

    # Stage 1: Decompose
    decompose_msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"""Problem: {problem}

Break this problem into sub-problems, ordered from simplest to most complex.
Each sub-problem should be solvable using only information from the problem
statement and solutions to previous sub-problems.

List sub-problems as a numbered list."""
            }
        ]
    )
    sub_problems_text = decompose_msg.content[0].text

    # Stage 2: Solve each sub-problem
    solutions = []
    accumulated_context = f"Problem: {problem}\n\n"

    # Parse sub-problems (simple line-based parsing)
    lines = sub_problems_text.strip().split("\n")
    sub_problems = [
        line.strip() for line in lines
        if line.strip() and line.strip()[0].isdigit()
    ]

    for i, sub_problem in enumerate(sub_problems):
        prior_solutions = "\n".join(
            f"Sub-problem {j+1}: {sol['problem']}\nSolution: {sol['answer']}"
            for j, sol in enumerate(solutions)
        )

        context = accumulated_context
        if prior_solutions:
            context += f"Previously solved:\n{prior_solutions}\n\n"

        solve_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": f"""{context}Now solve this sub-problem:
{sub_problem}

Show your calculation and give the answer."""
                }
            ]
        )

        solutions.append({
            "problem": sub_problem,
            "answer": solve_msg.content[0].text
        })

    # Stage 3: Final synthesis
    all_solutions = "\n\n".join(
        f"Sub-problem {i+1}: {sol['problem']}\nSolution: {sol['answer']}"
        for i, sol in enumerate(solutions)
    )

    final_msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=128,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"""Problem: {problem}

{all_solutions}

Based on all the sub-problem solutions above, what is the final answer?
State it clearly in one sentence."""
            }
        ]
    )

    return {
        "sub_problems": sub_problems,
        "solutions": solutions,
        "final_answer": final_msg.content[0].text
    }
```

### 8.3 구성적 일반화를 위한 최소에서 최대

최소에서 최대는 구성적 일반화(Compositional Generalization)에 특히 효과적입니다 — 익숙한 하위 문제의 새로운 조합을 해결하는 것. 예를 들어, 모델이 "리스트 정렬"과 "짝수 필터링"을 독립적으로 본 적이 있다면, 최소에서 최대는 "이 리스트에서 짝수를 정렬"하는 것을 결합하는 데 도움이 됩니다.

```python
import anthropic

client = anthropic.Anthropic()

# Compositional task: Combine multiple known operations
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """Task: Given a list of student records, find the average GPA
of students in the Computer Science department who have taken more
than 4 courses, sorted by name.

Data:
- Alice: CS department, 5 courses, GPA 3.8
- Bob: Math department, 6 courses, GPA 3.5
- Carol: CS department, 3 courses, GPA 3.9
- Dave: CS department, 7 courses, GPA 3.2
- Eve: CS department, 5 courses, GPA 3.6
- Frank: Math department, 4 courses, GPA 3.7

Let me decompose this into sub-problems from simplest to most complex:

Sub-problem 1: Filter students in the CS department.
Sub-problem 2: From those, filter students with more than 4 courses.
Sub-problem 3: Sort the filtered students by name.
Sub-problem 4: Calculate the average GPA of the sorted list.

Now solve each sub-problem in order:"""
        }
    ]
)
```

---

## 9. 사고의 프로그램(Program-of-Thought)

### 9.1 코드 실행을 포함한 CoT

사고의 프로그램(Program-of-Thought, PoT)은 자연어 추론을 코드로 대체합니다. "5 곱하기 3은 15"라고 쓰는 대신 모델이 Python에서 `5 * 3`을 작성합니다. 생성된 코드는 정확한 답을 얻기 위해 실행됩니다.

```python
import anthropic

client = anthropic.Anthropic()

def program_of_thought(question: str) -> dict:
    """Use Program-of-Thought: generate code instead of reasoning.

    Stage 1: Model generates Python code to solve the problem
    Stage 2: Execute the code to get the exact answer
    """

    # Stage 1: Generate code
    code_msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"""Solve this problem by writing Python code.
Output ONLY the Python code, no explanations.
The code should print the final answer.

Problem: {question}"""
            }
        ]
    )

    code = code_msg.content[0].text

    # Extract code from markdown code block if present
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    code = code.strip()

    # Stage 2: Execute the code (with safety measures)
    # WARNING: In production, use a sandboxed execution environment
    try:
        # Capture stdout
        import io
        import contextlib

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(code, {"__builtins__": __builtins__})

        result = output.getvalue().strip()
    except Exception as e:
        result = f"Error: {e}"

    return {
        "code": code,
        "result": result
    }

# Example
result = program_of_thought(
    "A store offers a 20% discount on purchases over $100. "
    "If someone buys 3 items at $45, $30, and $55, "
    "what is the total after discount and 8.5% sales tax?"
)

print(f"Generated code:\n{result['code']}")
print(f"\nResult: {result['result']}")
```

### 9.2 PoT vs CoT 비교

```python
import anthropic

client = anthropic.Anthropic()

# The same problem solved with CoT vs PoT

question = """A rectangle has a perimeter of 48 cm.
Its length is 3 times its width.
What is the area of the rectangle?"""

# CoT approach: Natural language reasoning
cot_response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=512,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": f"{question}\n\nLet's solve this step by step."
        }
    ]
)
# Model writes: "Let width = w, length = 3w. Perimeter = 2(w + 3w) = 48..."

# PoT approach: Code-based reasoning
pot_response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=512,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": f"""{question}

Solve this by writing Python code. Print only the final answer."""
        }
    ]
)
# Model writes:
# perimeter = 48
# # 2 * (length + width) = 48, length = 3 * width
# # 2 * (3w + w) = 48 -> 8w = 48 -> w = 6
# width = perimeter / 8
# length = 3 * width
# area = length * width
# print(f"The area is {area} square cm")

# Advantages of PoT:
# 1. Exact arithmetic (no rounding errors)
# 2. Can handle complex calculations (statistics, simulations)
# 3. Code is verifiable and testable
# 4. Can use libraries (numpy, sympy, etc.)
#
# Advantages of CoT:
# 1. No code execution infrastructure needed
# 2. More interpretable to non-technical users
# 3. Works for non-mathematical reasoning (ethics, common sense)
# 4. No security risks from code execution
```

---

## 10. CoT를 활용한 수학적 추론

### 10.1 구조화된 수학적 CoT

수학 문제의 경우, 개념적 설정과 계산을 분리하도록 CoT를 구조화합니다:

```python
import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """Solve this problem with a structured mathematical approach.

Problem: A cylindrical tank with radius 3m and height 10m is being filled
with water at a rate of 2 cubic meters per minute. How long will it take
to fill the tank to 80% capacity?

Structure your solution as:
1. IDENTIFY: What are the known quantities and what do we need to find?
2. FORMULATE: What formula(s) apply?
3. CALCULATE: Show each computation step
4. VERIFY: Does the answer make sense?
5. ANSWER: State the final answer with units"""
        }
    ]
)

# Expected reasoning:
# 1. IDENTIFY: r=3m, h=10m, fill rate=2m³/min, target=80%
# 2. FORMULATE: V = πr²h, target_volume = 0.8 × V, time = target_volume / rate
# 3. CALCULATE:
#    V = π × 3² × 10 = 90π ≈ 282.74 m³
#    target = 0.8 × 282.74 = 226.19 m³
#    time = 226.19 / 2 = 113.10 minutes ≈ 1 hour 53 minutes
# 4. VERIFY: Full tank takes ~141 minutes, 80% should be ~113 min. ✓
# 5. ANSWER: Approximately 113.1 minutes (1 hour 53 minutes)
```

### 10.2 문장제 문제를 위한 CoT

문장제 문제는 자연어를 수학적 표현으로 번역하는 것을 요구합니다. CoT는 이 번역을 명시적으로 만듭니다:

```python
import anthropic

client = anthropic.Anthropic()

# Teaching the model a structured approach to word problems
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """Solve this word problem by first translating it into equations.

Problem: A school is organizing a field trip. Each bus holds 45 students.
There are 3 grades going on the trip. Grade 5 has 120 students,
Grade 6 has 135 students, and Grade 7 has 150 students. If each bus
must have at least one teacher, and there is 1 teacher for every
15 students, how many buses are needed in total (for both students
and teachers)?

Step 1: Extract quantities from the problem
Step 2: Set up equations
Step 3: Solve
Step 4: Handle real-world constraints (can't have partial buses/teachers)"""
        }
    ]
)
```

### 10.3 수학에서의 일반적인 CoT 오류

LLM은 수학적 CoT에서 예측 가능한 오류를 만듭니다. 이를 알면 더 나은 프롬프트를 설계하는 데 도움이 됩니다:

```python
import anthropic

client = anthropic.Anthropic()

# Error type 1: Arithmetic mistakes
# Mitigation: Ask the model to verify each calculation
verify_prompt = """Solve: What is 17% of 834?

Show your work and verify each step:
Step 1: Convert percentage to decimal
Step 2: Multiply
Step 3: Verify by checking if the answer is approximately
        17/100 of 834 (should be about 1/6 of 834 ≈ 139)"""

# Error type 2: Unit errors
# Mitigation: Require explicit unit tracking
unit_prompt = """A car travels 150 km in 2 hours and 30 minutes.
What is its average speed in meters per second?

Track units explicitly at each step:
- Distance: convert km to m
- Time: convert hours and minutes to seconds
- Speed: divide to get m/s
Show the units at every step."""

# Error type 3: Off-by-one errors
# Mitigation: Use concrete examples to verify
fence_post_prompt = """How many fence posts are needed for a 100-meter
straight fence with posts every 10 meters?

Before answering, consider a small example:
A 20-meter fence with posts every 10 meters needs posts at:
0m, 10m, 20m = 3 posts (not 2!)

Now solve the original problem."""
```

---

## 연습문제

### 연습문제 1: 제로샷 CoT 적용

이 문제를 푸는 제로샷 CoT 프롬프트를 작성하세요: "회문(Palindrome)은 앞에서 읽으나 뒤에서 읽으나 같은 문자열입니다. 모든 숫자가 홀수인 4자리 회문은 몇 개 있나요?" 2단계 접근법(추론 + 추출)을 적용하세요.

<details><summary>정답 보기</summary>

```python
import anthropic

client = anthropic.Anthropic()

# Stage 1: Reasoning
reasoning_msg = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """How many 4-digit palindromes are there where all digits are odd?

A 4-digit palindrome has the form ABBA where each letter represents a digit.

Let's think step by step."""
        }
    ]
)

reasoning = reasoning_msg.content[0].text

# Expected reasoning:
# Step 1: A 4-digit palindrome has the form ABBA.
#   - 1st digit = 4th digit (both A)
#   - 2nd digit = 3rd digit (both B)
# Step 2: All digits must be odd. Odd digits: {1, 3, 5, 7, 9} = 5 choices.
# Step 3: A can be any odd digit. Since it's the first digit of a
#   4-digit number, A cannot be 0. But since A must be odd, A is
#   already non-zero. So A has 5 choices.
# Step 4: B can be any odd digit. B has 5 choices.
# Step 5: Total palindromes = choices for A × choices for B = 5 × 5 = 25

# Stage 2: Answer extraction
answer_msg = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=32,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": f"""How many 4-digit palindromes are there where all digits are odd?

{reasoning}

What is the final numerical answer? Reply with only the number."""
        }
    ]
)

print(f"Answer: {answer_msg.content[0].text.strip()}")
# Answer: 25
```

</details>

### 연습문제 2: 수동 CoT 설계

다음 과제를 위한 2개의 시연이 포함된 수동 CoT 프롬프트를 설계하세요: 주어진 연도가 윤년인지 판별하기. 시연에서 윤년 규칙이 순서대로 적용되는 것을 명시적으로 보여야 합니다.

<details><summary>정답 보기</summary>

```python
import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """Determine if a year is a leap year by applying the rules in order.

Q: Is 1900 a leap year?
A: Let me apply the leap year rules in order.
Rule 1: Is 1900 divisible by 4? 1900 / 4 = 475. Yes, it is divisible by 4.
Rule 2: Is 1900 divisible by 100? 1900 / 100 = 19. Yes, it is divisible by 100.
Rule 3: Is 1900 divisible by 400? 1900 / 400 = 4.75. No, it is NOT divisible by 400.
Conclusion: Since 1900 is divisible by 100 but NOT by 400, it is NOT a leap year.
The answer is: Not a leap year.

Q: Is 2000 a leap year?
A: Let me apply the leap year rules in order.
Rule 1: Is 2000 divisible by 4? 2000 / 4 = 500. Yes, it is divisible by 4.
Rule 2: Is 2000 divisible by 100? 2000 / 100 = 20. Yes, it is divisible by 100.
Rule 3: Is 2000 divisible by 400? 2000 / 400 = 5. Yes, it IS divisible by 400.
Conclusion: Since 2000 is divisible by 400, it IS a leap year.
The answer is: Leap year.

Q: Is 2100 a leap year?
A:"""
        }
    ]
)

print(message.content[0].text)

# The demonstrations are carefully chosen:
# 1900: Divisible by 4 and 100, but NOT 400 (NOT a leap year)
# 2000: Divisible by 4, 100, AND 400 (IS a leap year)
# These two examples cover the tricky cases. A simpler year like
# 2024 (divisible by 4 but not 100) would exit at Rule 1.
#
# 2100 (the test case): Similar to 1900, should be NOT a leap year.
```

</details>

### 연습문제 3: 자기 일관성 구현

다음 문제에 대한 자기 일관성 솔버를 구현하세요: "6면 주사위와 8면 주사위가 있습니다. 둘 다 굴렸을 때 합이 10보다 클 확률은?" 5개의 추론 경로를 생성하고 다수결 투표를 하세요. 신뢰도 지표를 포함하세요.

<details><summary>정답 보기</summary>

```python
import anthropic
from collections import Counter
import re

client = anthropic.Anthropic()

def extract_fraction_or_decimal(text: str) -> str:
    """Extract a probability value from text."""
    # Look for fractions like 6/48 or simplified like 1/8
    fraction_match = re.search(r'(\d+)/(\d+)', text)
    if fraction_match:
        num, den = int(fraction_match.group(1)), int(fraction_match.group(2))
        return f"{num}/{den}"

    # Look for decimals
    decimal_match = re.search(r'0\.\d+', text)
    if decimal_match:
        return decimal_match.group()

    # Look for percentages
    pct_match = re.search(r'(\d+\.?\d*)%', text)
    if pct_match:
        return f"{float(pct_match.group(1))/100:.4f}"

    return text.strip()

def solve_with_self_consistency(
    question: str,
    num_paths: int = 5,
    temperature: float = 0.7
) -> dict:
    """Solve using self-consistency with multiple reasoning paths."""

    paths = []

    for i in range(num_paths):
        # Generate reasoning
        reasoning_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": f"{question}\n\nLet's solve this step by step."
                }
            ]
        )
        reasoning = reasoning_msg.content[0].text

        # Extract answer
        answer_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=64,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": f"""{question}

{reasoning}

Express the probability as a simplified fraction (e.g., 1/8).
Reply with ONLY the fraction."""
                }
            ]
        )

        answer = answer_msg.content[0].text.strip()

        paths.append({
            "path_id": i + 1,
            "reasoning_summary": reasoning[:200] + "...",
            "answer": answer
        })

    # Normalize and count
    normalized_answers = []
    for p in paths:
        normalized = extract_fraction_or_decimal(p["answer"])
        normalized_answers.append(normalized)

    vote_counts = Counter(normalized_answers)
    majority_answer = vote_counts.most_common(1)[0][0]
    confidence = vote_counts[majority_answer] / num_paths

    return {
        "question": question,
        "majority_answer": majority_answer,
        "confidence": f"{confidence:.0%}",
        "num_paths": num_paths,
        "vote_distribution": dict(vote_counts),
        "paths": paths,
        "status": "high_confidence" if confidence >= 0.6 else "low_confidence"
    }

# Solve the problem
result = solve_with_self_consistency(
    "I have a 6-sided die and an 8-sided die. If I roll both, "
    "what is the probability that the sum is greater than 10?",
    num_paths=5
)

print(f"Answer: {result['majority_answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Vote distribution: {result['vote_distribution']}")

# Correct answer: The possible outcomes with sum > 10 are:
# (3,8), (4,7), (4,8), (5,6), (5,7), (5,8), (6,5), (6,6), (6,7), (6,8)
# That's 10 outcomes out of 6 × 8 = 48 total
# Probability = 10/48 = 5/24
```

</details>

### 연습문제 4: 최소에서 최대 문제

최소에서 최대 프롬프팅을 사용하여 이 문제를 분해하고 풀어보세요: "한 회사가 247명의 모든 직원을 3일간의 교육 컨퍼런스에 보내려 합니다. 호텔은 1인실 $120/박, 2인실 1인당 $90/박을 부과합니다. 직원의 30%가 1인실을 선호한다면 최소 총 호텔 비용은 얼마인가요?"

완전한 분해와 풀이 체인을 작성하세요.

<details><summary>정답 보기</summary>

```python
import anthropic

client = anthropic.Anthropic()

problem = """A company wants to send all 247 employees to a 3-day training conference.
Hotels charge $120/night for single rooms and $90/night per person for double rooms.
If 30% of employees prefer single rooms, what is the minimum total hotel cost?"""

# Stage 1: Decomposition
decompose = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=512,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": f"""Problem: {problem}

Break this into sub-problems from simplest to most complex.
Each sub-problem should produce a number I need for later sub-problems."""
        }
    ]
)

print("=== DECOMPOSITION ===")
print(decompose.content[0].text)

# Stage 2: Solve sequentially
sub_problems = [
    "How many employees prefer single rooms? (30% of 247)",
    "How many employees need double rooms? (remainder)",
    "How many double rooms are needed? (handle odd numbers)",
    "What is the total cost for single rooms over 3 nights?",
    "What is the total cost for double rooms over 3 nights?",
    "What is the total minimum hotel cost?"
]

solutions = []
context = f"Problem: {problem}\n\n"

for i, sub in enumerate(sub_problems):
    prior = ""
    if solutions:
        prior = "Previously solved:\n" + "\n".join(
            f"  {j+1}. {s['problem']} => {s['answer']}"
            for j, s in enumerate(solutions)
        ) + "\n\n"

    solve_msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"""{context}{prior}Now solve sub-problem {i+1}: {sub}

Show your calculation and give a single numerical answer."""
            }
        ]
    )

    solutions.append({
        "problem": sub,
        "answer": solve_msg.content[0].text.strip()
    })

    print(f"\n=== SUB-PROBLEM {i+1} ===")
    print(f"Q: {sub}")
    print(f"A: {solutions[-1]['answer']}")

# Expected solutions:
# 1. Single room employees: 30% × 247 = 74.1 -> round to 75 (can't split a person)
#    (or 74, depending on interpretation — "prefer" means 74 people prefer it)
# 2. Double room employees: 247 - 74 = 173
# 3. Double rooms needed: 173 / 2 = 86.5 -> 87 rooms (one person alone pays double rate)
# 4. Single room cost: 74 rooms × $120/night × 3 nights = $26,640
# 5. Double room cost: 173 people × $90/night × 3 nights = $46,710
# 6. Total: $26,640 + $46,710 = $73,350
```

</details>

### 연습문제 5: CoT vs 직접 비교

5개의 구체적인 문제에서 CoT와 직접 프롬프팅을 비교하는 실험을 설계하세요. 각 문제에 대해 예상 정답을 제공하고 CoT가 도움이 될지 해가 될지 설명하세요. CoT가 성능을 저하시킬 것으로 예상되는 문제를 최소 하나 포함하세요.

<details><summary>정답 보기</summary>

```python
import anthropic

client = anthropic.Anthropic()

test_problems = [
    {
        "question": "What is the next number in the sequence: 2, 6, 18, 54, ?",
        "correct_answer": "162",
        "cot_prediction": "helps",
        "reasoning": "Pattern recognition (×3) benefits from explicit step-by-step analysis"
    },
    {
        "question": "A snail climbs 3 feet up a wall during the day but slides back 2 feet at night. How many days to reach the top of a 10-foot wall?",
        "correct_answer": "8",
        "cot_prediction": "helps",
        "reasoning": "Tricky problem: the naive answer is 10 days (1ft/day × 10ft), but CoT reveals the snail reaches the top on day 8 during the daytime climb, before it can slide back"
    },
    {
        "question": "What is the capital of Australia?",
        "correct_answer": "Canberra",
        "cot_prediction": "hurts",
        "reasoning": "Simple factual recall. CoT adds unnecessary verbosity and the reasoning might introduce doubt (e.g., considering Sydney or Melbourne)"
    },
    {
        "question": "If you rearrange the letters 'CIFAIPC' you get the name of a: (a) city, (b) animal, (c) ocean, (d) river",
        "correct_answer": "c",
        "cot_prediction": "helps",
        "reasoning": "Anagram solving benefits from CoT because the model can try different letter arrangements systematically (PACIFIC = ocean)"
    },
    {
        "question": "In a room of 23 people, what is the approximate probability that at least two share a birthday? (Round to nearest 10%)",
        "correct_answer": "50%",
        "cot_prediction": "helps",
        "reasoning": "The birthday problem is counter-intuitive. CoT helps the model work through the complementary probability calculation rather than guessing"
    },
]

def run_comparison(problems: list[dict]) -> None:
    """Run CoT vs Direct comparison on test problems."""

    print(f"{'Problem':>4} | {'Direct':>10} | {'CoT':>10} | {'Correct':>10} | {'Expected':>10}")
    print("-" * 60)

    for i, prob in enumerate(problems):
        # Direct prompting
        direct_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=32,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": f"{prob['question']}\n\nAnswer with ONLY the answer, nothing else."
                }
            ]
        )
        direct_answer = direct_msg.content[0].text.strip()

        # CoT prompting
        cot_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": f"{prob['question']}\n\nLet's think step by step."
                }
            ]
        )
        cot_reasoning = cot_msg.content[0].text

        # Extract CoT answer
        cot_extract = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=32,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": f"{prob['question']}\n\n{cot_reasoning}\n\nWhat is the final answer? Reply with ONLY the answer."
                }
            ]
        )
        cot_answer = cot_extract.content[0].text.strip()

        print(
            f"{i+1:>4} | {direct_answer:>10} | {cot_answer:>10} | "
            f"{prob['correct_answer']:>10} | CoT {prob['cot_prediction']:>5}"
        )

# run_comparison(test_problems)

# Expected results table:
# Prob |     Direct |        CoT |    Correct |   Expected
# ---------------------------------------------------------
#    1 |        162 |        162 |        162 | CoT helps  (both correct, CoT adds confidence)
#    2 |         10 |          8 |          8 | CoT helps  (direct gets tricked, CoT catches edge case)
#    3 |   Canberra |   Canberra |   Canberra | CoT hurts  (both correct, but CoT wastes tokens)
#    4 |          c |          c |          c | CoT helps  (CoT can systematically try arrangements)
#    5 |        10% |        50% |        50% | CoT helps  (intuition is wrong, calculation is needed)
#
# Key takeaways:
# - CoT adds the most value on problems 2 and 5 (counter-intuitive answers)
# - CoT is wasteful on problem 3 (simple factual recall)
# - Problems requiring calculation or systematic search benefit most from CoT
```

</details>

---

**이전**: [제로샷과 퓨샷](./02_Zero_Shot_and_Few_Shot.md) | **다음**: [고급 추론 프롬프트](./04_Advanced_Reasoning_Prompts.md)

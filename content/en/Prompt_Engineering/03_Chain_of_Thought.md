# 03. Chain-of-Thought Prompting

**Previous**: [Zero-Shot and Few-Shot](./02_Zero_Shot_and_Few_Shot.md) | **Next**: [Advanced Reasoning Prompts](./04_Advanced_Reasoning_Prompts.md)

## Learning Objectives

- Explain why Chain-of-Thought prompting improves reasoning in large language models
- Implement both zero-shot and manual CoT prompting for diverse reasoning tasks
- Apply self-consistency with CoT to improve answer reliability through majority voting
- Use least-to-most prompting to decompose complex problems into solvable sub-problems
- Evaluate when CoT helps versus hurts performance and select the appropriate technique

---

Chain-of-Thought (CoT) prompting is one of the most impactful discoveries in prompt engineering. Instead of asking a model to produce an answer directly, CoT prompts the model to generate intermediate reasoning steps before arriving at a conclusion. This seemingly simple change — "show your work" — dramatically improves performance on tasks requiring logic, arithmetic, multi-step reasoning, and common-sense inference. This lesson covers the theory, variants, and practical applications of CoT and its extensions.

## Table of Contents

1. [Chain-of-Thought Fundamentals](#1-chain-of-thought-fundamentals)
2. [Why CoT Works](#2-why-cot-works)
3. [Zero-Shot CoT](#3-zero-shot-cot)
4. [Manual CoT with Demonstrations](#4-manual-cot-with-demonstrations)
5. [Auto-CoT](#5-auto-cot)
6. [When CoT Helps vs Hurts](#6-when-cot-helps-vs-hurts)
7. [Self-Consistency with CoT](#7-self-consistency-with-cot)
8. [Least-to-Most Prompting](#8-least-to-most-prompting)
9. [Program-of-Thought](#9-program-of-thought)
10. [Mathematical Reasoning with CoT](#10-mathematical-reasoning-with-cot)
11. [Exercises](#exercises)

---

## 1. Chain-of-Thought Fundamentals

### 1.1 The Core Idea

Standard prompting asks a model to go directly from question to answer:

```
Q: If a store has 3 shelves with 8 books each, and 5 books are removed, how many remain?
A: 19
```

Chain-of-Thought prompting asks the model to show its reasoning:

```
Q: If a store has 3 shelves with 8 books each, and 5 books are removed, how many remain?
A: The store starts with 3 shelves × 8 books = 24 books total.
   After removing 5 books: 24 - 5 = 19 books remain.
   The answer is 19.
```

Both produce the same answer, but the CoT version is more reliable on harder problems because each reasoning step builds on verifiable intermediate results.

### 1.2 Basic CoT Implementation

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

### 1.3 The Anatomy of a Chain of Thought

A well-formed chain of thought has these properties:

1. **Decomposition**: The problem is broken into smaller sub-problems
2. **Sequential dependencies**: Each step uses results from previous steps
3. **Explicit computation**: Mathematical operations are performed step by step
4. **Intermediate conclusions**: Each step ends with a clear result
5. **Final synthesis**: Steps are combined into a final answer

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

## 2. Why CoT Works

### 2.1 The Computational Argument

LLMs are essentially constant-depth circuits — they process each token through a fixed number of layers. Without CoT, the model must solve the entire problem in a single forward pass, which limits the computational complexity of problems it can solve.

CoT effectively gives the model more "compute time" by allowing it to write out intermediate results. Each generated token feeds back into the model as input, providing additional processing steps. In essence, CoT transforms the model from a fixed-depth circuit into a variable-depth one.

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

### 2.2 Emergent Reasoning

CoT is an **emergent capability** — it only works well in models above a certain size threshold (roughly 100B+ parameters, though this threshold has decreased with better training). Smaller models attempting CoT often produce plausible-sounding but incorrect reasoning chains.

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

### 2.3 Faithfulness of Reasoning

An important caveat: the reasoning chain a model produces is not necessarily the actual computation the model performs internally. Research has shown that models can sometimes arrive at the right answer for the wrong reasons, or produce convincing reasoning chains that do not actually reflect their internal processing.

This means:
- CoT improves accuracy, but the explanations may not be fully trustworthy
- Verification of the final answer through external means is still important
- The quality of reasoning varies by model and task type

---

## 3. Zero-Shot CoT

### 3.1 The Magic Phrase

The simplest form of CoT is zero-shot CoT, discovered by Kojima et al. (2022). Simply appending "Let's think step by step" to a prompt significantly improves reasoning accuracy without any examples.

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

### 3.2 Variations of the Trigger Phrase

Different trigger phrases can work better for different tasks:

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

### 3.3 Two-Stage Zero-Shot CoT

For maximum accuracy, use a two-stage approach: first generate the reasoning, then extract the answer.

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

## 4. Manual CoT with Demonstrations

### 4.1 Providing Reasoning Examples

Manual CoT combines few-shot prompting with Chain-of-Thought. You provide examples that include not just the input and output, but the complete reasoning process.

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

### 4.2 Crafting Effective Demonstrations

The quality of your CoT demonstrations directly affects output quality. Here are principles for crafting effective demonstrations:

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

### 4.3 Multi-Domain CoT Templates

Different reasoning domains require different CoT structures:

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

## 5. Auto-CoT

### 5.1 Concept

Auto-CoT (Zhang et al., 2022) automates the creation of CoT demonstrations. Instead of manually writing reasoning chains, you let the model generate them for a diverse set of questions, then use these generated chains as few-shot examples.

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

### 5.2 Diversity in Auto-CoT

The key insight of Auto-CoT is that diverse demonstrations are more effective than random ones. Clustering ensures the demonstrations cover different reasoning patterns.

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

## 6. When CoT Helps vs Hurts

### 6.1 Tasks Where CoT Helps

CoT consistently improves performance on:

| Task Type | Example | Why CoT Helps |
|-----------|---------|---------------|
| Arithmetic | Multi-step calculations | Externalizes computation |
| Logic puzzles | Constraint satisfaction | Systematic elimination |
| Word problems | Story-based math | Separates understanding from computation |
| Common-sense reasoning | "Would X happen if Y?" | Makes implicit knowledge explicit |
| Code reasoning | "What does this code output?" | Simulates execution |
| Multi-hop QA | Questions requiring chaining facts | Explicitly chains inferences |

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

### 6.2 Tasks Where CoT Hurts

CoT can actually decrease performance on:

| Task Type | Example | Why CoT Hurts |
|-----------|---------|---------------|
| Simple retrieval | "What is the capital of France?" | Overthinking a trivial answer |
| Pattern matching | Simple classification | Reasoning adds noise |
| Creative writing | Poetry, stories | Analytical thinking constrains creativity |
| Quick factual answers | "What year was X born?" | Unnecessary verbosity |
| Low-complexity tasks | Yes/no questions | The reasoning can introduce errors |

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

### 6.3 Model Size Dependency

CoT's effectiveness depends on model size:

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

## 7. Self-Consistency with CoT

### 7.1 The Self-Consistency Approach

Self-consistency (Wang et al., 2022) recognizes that there are often multiple valid reasoning paths to the same answer. Instead of relying on a single greedy CoT chain, it:

1. Generates multiple reasoning chains (with temperature > 0)
2. Extracts the answer from each chain
3. Takes the majority vote as the final answer

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

### 7.2 Confidence Estimation

Self-consistency provides a natural confidence estimate: the proportion of reasoning paths that agree on the answer.

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

### 7.3 When to Use Self-Consistency

Self-consistency is most valuable when:
- The task has a single correct answer (not open-ended generation)
- Different reasoning approaches might yield different results
- High accuracy is more important than latency or cost
- The answer space is discrete (numbers, labels, yes/no)

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

## 8. Least-to-Most Prompting

### 8.1 The Decomposition Strategy

Least-to-most prompting (Zhou et al., 2022) solves complex problems by explicitly decomposing them into simpler sub-problems, solving each sub-problem in order, and using earlier solutions to solve later ones.

It differs from standard CoT in that the decomposition step is explicit and the model is asked to identify the sub-problems first.

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

### 8.2 Full Least-to-Most Pipeline

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

### 8.3 Least-to-Most for Compositional Generalization

Least-to-most is particularly effective for compositional generalization — solving novel combinations of familiar sub-problems. For example, if a model has seen "sort a list" and "filter even numbers" independently, least-to-most helps it combine them for "sort the even numbers from this list."

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

## 9. Program-of-Thought

### 9.1 CoT with Code Execution

Program-of-Thought (PoT) replaces natural language reasoning with code. Instead of writing "5 times 3 is 15," the model writes `5 * 3` in Python. The generated code is then executed to get the exact answer.

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

### 9.2 PoT vs CoT Comparison

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

## 10. Mathematical Reasoning with CoT

### 10.1 Structured Mathematical CoT

For mathematical problems, structure the CoT to separate the conceptual setup from the computation:

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

### 10.2 CoT for Word Problems

Word problems require translating natural language into mathematical representations. CoT makes this translation explicit:

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

### 10.3 Common CoT Errors in Math

LLMs make predictable errors in mathematical CoT. Knowing these helps you design better prompts:

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

## Exercises

### Exercise 1: Zero-Shot CoT Application

Write a zero-shot CoT prompt that solves this problem: "A palindrome is a string that reads the same forwards and backwards. How many 4-digit palindromes are there where the digits are all odd?" Apply the two-stage approach (reasoning + extraction).

<details><summary>Show Answer</summary>

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

### Exercise 2: Manual CoT Design

Design a manual CoT prompt with 2 demonstrations for the following task: determining whether a given year is a leap year. Your demonstrations should explicitly show the leap year rules being applied in order.

<details><summary>Show Answer</summary>

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

### Exercise 3: Self-Consistency Implementation

Implement a self-consistency solver for the following problem: "I have a 6-sided die and a 8-sided die. If I roll both, what is the probability that the sum is greater than 10?" Generate 5 reasoning paths and take a majority vote. Include a confidence metric.

<details><summary>Show Answer</summary>

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

### Exercise 4: Least-to-Most Problem

Decompose and solve this problem using least-to-most prompting: "A company wants to send all 247 employees to a 3-day training conference. Hotels charge $120/night for single rooms and $90/night per person for double rooms. If 30% of employees prefer single rooms, what is the minimum total hotel cost?"

Write the complete decomposition and solution chain.

<details><summary>Show Answer</summary>

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

### Exercise 5: CoT vs Direct Comparison

Design an experiment that compares CoT and direct prompting on 5 specific problems. For each problem, provide the expected correct answer and explain whether CoT should help or hurt. Include at least one problem where CoT is expected to hurt performance.

<details><summary>Show Answer</summary>

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

**Previous**: [Zero-Shot and Few-Shot](./02_Zero_Shot_and_Few_Shot.md) | **Next**: [Advanced Reasoning Prompts](./04_Advanced_Reasoning_Prompts.md)

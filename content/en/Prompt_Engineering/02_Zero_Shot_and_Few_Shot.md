# 02. Zero-Shot and Few-Shot Prompting

**Previous**: [Prompt Fundamentals](./01_Prompt_Fundamentals.md) | **Next**: [Chain of Thought](./03_Chain_of_Thought.md)

## Learning Objectives

- Distinguish between zero-shot and few-shot prompting and select the appropriate technique for a given task
- Design effective few-shot examples using diversity, similarity, and representativeness criteria
- Implement dynamic few-shot selection using semantic search and embedding similarity
- Manage token budgets when constructing few-shot prompts with limited context windows
- Evaluate the performance tradeoffs of zero-shot vs few-shot across different task categories

---

Zero-shot and few-shot prompting are the two most fundamental prompting strategies. Zero-shot relies entirely on the model's pre-trained knowledge and instruction-following ability, while few-shot provides concrete input-output examples that demonstrate the desired behavior. Choosing between them — and executing few-shot effectively — is one of the highest-leverage skills in prompt engineering. This lesson covers the theory, implementation, and optimization of both approaches.

## Table of Contents

1. [Zero-Shot Prompting](#1-zero-shot-prompting)
2. [Few-Shot Prompting Fundamentals](#2-few-shot-prompting-fundamentals)
3. [Example Selection Strategies](#3-example-selection-strategies)
4. [Example Ordering and Bias Effects](#4-example-ordering-and-bias-effects)
5. [Dynamic Few-Shot with Semantic Search](#5-dynamic-few-shot-with-semantic-search)
6. [K-Shot Selection Optimization](#6-k-shot-selection-optimization)
7. [Label Balance in Examples](#7-label-balance-in-examples)
8. [Token Budget Management](#8-token-budget-management)
9. [Zero-Shot vs Few-Shot: Task Comparison](#9-zero-shot-vs-few-shot-task-comparison)
10. [Exercises](#exercises)

---

## 1. Zero-Shot Prompting

Zero-shot prompting means asking a model to perform a task with no examples — only the task description. The model relies entirely on patterns learned during pre-training and instruction tuning.

### 1.1 When Zero-Shot Works Well

Zero-shot prompting succeeds when:
- The task is well-defined and commonly encountered in training data
- The model has strong instruction-following capabilities
- The output format is simple and unambiguous
- The task aligns with natural language understanding (classification, translation, summarization)

```python
import anthropic

client = anthropic.Anthropic()

# Zero-shot classification — works well because sentiment is a common task
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=64,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """Classify the sentiment of this review as
positive, negative, or neutral.

Review: "The battery life is incredible and the screen is sharp,
but the camera could be better."

Sentiment:"""
        }
    ]
)

print(message.content[0].text)
# Output: "mixed" or "positive" — the model understands sentiment natively
```

### 1.2 When Zero-Shot Struggles

Zero-shot prompting struggles when:
- The task requires a specific output format the model has not seen before
- Domain-specific conventions differ from general knowledge
- The task is ambiguous and different interpretations are equally valid
- Custom label taxonomies are used (e.g., labeling emails as "P1" through "P5")

```python
import anthropic

client = anthropic.Anthropic()

# Zero-shot with a custom taxonomy — the model does not know your labels
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=128,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """Classify this support ticket into one of these categories:
- ACCT_BILLING: Account and billing issues
- TECH_CONN: Technical connectivity problems
- TECH_PERF: Technical performance issues
- FEAT_REQ: Feature requests
- GEN_INQ: General inquiries

Ticket: "My dashboard loads but takes over 30 seconds. All other
websites work fine. This started after your last update."

Category:"""
        }
    ]
)

print(message.content[0].text)
# Works reasonably well because the labels are descriptive
# Would struggle if labels were opaque like "CAT_A", "CAT_B"
```

### 1.3 Improving Zero-Shot Performance

When you must use zero-shot (e.g., token constraints, high variety of inputs), these techniques improve reliability:

```python
import anthropic

client = anthropic.Anthropic()

# Technique 1: Explicit output format specification
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """Extract entities from this text and return valid JSON.

Text: "Dr. Sarah Chen at MIT published a paper on quantum computing
in Nature on March 15, 2025."

Return ONLY a JSON object with these exact keys:
{
    "persons": [list of person names],
    "organizations": [list of organization names],
    "publications": [list of publication names],
    "dates": [list of dates in ISO 8601 format]
}"""
        }
    ]
)

# Technique 2: Role priming for domain expertise
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    temperature=0.0,
    system="You are an expert medical coder with 15 years of experience "
           "assigning ICD-10 codes. You always provide the most specific "
           "code available and explain your reasoning.",
    messages=[
        {
            "role": "user",
            "content": "Assign the ICD-10 code for: 'Patient presents with "
                       "acute bronchitis with bronchospasm'"
        }
    ]
)

# Technique 3: Step-by-step structure without examples
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=512,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """Determine if this argument is logically valid.

Argument: "All birds can fly. Penguins are birds.
Therefore, penguins can fly."

Follow these steps:
1. Identify the premises and conclusion
2. Check if the premises are factually accurate
3. Check if the conclusion follows logically from the premises
4. State whether the argument is valid, sound, both, or neither
5. Explain in one sentence"""
        }
    ]
)
```

---

## 2. Few-Shot Prompting Fundamentals

Few-shot prompting provides concrete input-output examples before the actual query. These examples serve as implicit instructions, demonstrating the pattern the model should follow.

### 2.1 The Basic Pattern

```python
import anthropic

client = anthropic.Anthropic()

# Basic few-shot: 3 examples before the query
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=128,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """Convert the product description to a structured format.

Description: "Nike Air Max 90, men's size 10, white/black colorway, $129.99"
Output: {"brand": "Nike", "model": "Air Max 90", "gender": "men", "size": "10", "colors": ["white", "black"], "price": 129.99}

Description: "Adidas Ultraboost 22, women's size 7.5, cloud white, $189.99"
Output: {"brand": "Adidas", "model": "Ultraboost 22", "gender": "women", "size": "7.5", "colors": ["cloud white"], "price": 189.99}

Description: "New Balance 574, unisex size 9, grey/navy, on sale for $79.99"
Output: {"brand": "New Balance", "model": "574", "gender": "unisex", "size": "9", "colors": ["grey", "navy"], "price": 79.99}

Description: "Puma RS-X, men's size 11, black/red/white, $109.99"
Output:"""
        }
    ]
)

print(message.content[0].text)
```

### 2.2 Why Few-Shot Works

Few-shot examples work through a mechanism called **in-context learning (ICL)**. The model does not update its weights — instead, it recognizes the pattern in the examples and applies it to the new input. This works because:

1. **Pattern recognition**: Transformers are powerful pattern matchers. Given consistent input-output pairs, they infer the transformation rule.

2. **Format demonstration**: Examples show the exact output format more precisely than any description could.

3. **Edge case coverage**: Examples can demonstrate how to handle ambiguous or tricky cases.

4. **Label grounding**: Examples ground abstract labels in concrete instances, reducing misinterpretation.

```python
import anthropic

client = anthropic.Anthropic()

# Without examples, the model might classify differently
# than your labeling guidelines specify

# Your label "urgent" means specific things in your domain
# Examples ground this definition

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=64,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """Classify each support ticket as "urgent" or "normal".

Ticket: "Cannot log in to my account, error 500 on every attempt"
Label: urgent

Ticket: "How do I change my profile picture?"
Label: normal

Ticket: "Payment failed for 3 consecutive transactions"
Label: urgent

Ticket: "Would love to see a dark mode option"
Label: normal

Ticket: "All data in my dashboard shows $0.00 since this morning"
Label: urgent

Ticket: "My API calls are returning 503 errors for the past hour"
Label:"""
        }
    ]
)

print(message.content[0].text)
# Output: "urgent" (production-impacting, similar to previous urgent examples)
```

### 2.3 Few-Shot with the Messages API

Modern APIs use a conversation structure. You can provide few-shot examples as alternating user/assistant messages:

```python
import anthropic

client = anthropic.Anthropic()

# Few-shot using the conversation format
# Each example is a user message + assistant response pair
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    temperature=0.0,
    messages=[
        # Example 1
        {
            "role": "user",
            "content": "Translate to formal Japanese: 'Where is the train station?'"
        },
        {
            "role": "assistant",
            "content": "駅はどちらにございますでしょうか。(Eki wa dochira ni gozaimasu deshō ka.)"
        },
        # Example 2
        {
            "role": "user",
            "content": "Translate to formal Japanese: 'I would like to make a reservation.'"
        },
        {
            "role": "assistant",
            "content": "予約をお願いしたいのですが。(Yoyaku o onegai shitai no desu ga.)"
        },
        # Example 3
        {
            "role": "user",
            "content": "Translate to formal Japanese: 'Thank you for your help.'"
        },
        {
            "role": "assistant",
            "content": "ご助力いただきまして、ありがとうございます。(Go-joryoku itadakimashite, arigatō gozaimasu.)"
        },
        # Actual query
        {
            "role": "user",
            "content": "Translate to formal Japanese: 'Could you please explain this document?'"
        }
    ]
)

print(message.content[0].text)
```

### 2.4 One-Shot vs Few-Shot

Sometimes a single example is sufficient — this is **one-shot** prompting. Use one-shot when:
- The task pattern is simple and consistent
- Token budget is limited
- The format is the main thing being demonstrated

```python
import anthropic

client = anthropic.Anthropic()

# One-shot is often enough for format demonstration
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=512,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """Generate a changelog entry from a git commit message.

Commit: "fix(auth): resolve token refresh race condition causing 401 errors #1234"
Changelog: - **Fixed**: Token refresh race condition that caused intermittent 401 errors during concurrent API calls ([#1234](https://github.com/org/repo/issues/1234))

Commit: "feat(dashboard): add real-time notification bell with WebSocket support #892"
Changelog:"""
        }
    ]
)

print(message.content[0].text)
```

---

## 3. Example Selection Strategies

The quality of few-shot examples often matters more than the quantity. Poor examples can actively hurt performance compared to zero-shot.

### 3.1 Diversity Strategy

Select examples that cover different aspects of the task space. This helps the model generalize rather than overfit to a narrow pattern.

```python
# Task: Classify customer complaints
# BAD: All examples are about billing (low diversity)
bad_examples = [
    ("I was charged twice for my subscription", "billing"),
    ("My invoice shows the wrong amount", "billing"),
    ("The price went up without notice", "billing"),
]

# GOOD: Examples cover different categories (high diversity)
good_examples = [
    ("I was charged twice for my subscription", "billing"),
    ("The app crashes every time I open the settings page", "technical"),
    ("Your delivery arrived 3 days late and the box was damaged", "shipping"),
    ("I'd love to see integration with Slack", "feature_request"),
    ("How do I export my data to CSV?", "how_to"),
]
```

```python
import anthropic

client = anthropic.Anthropic()

def classify_with_diverse_examples(text: str) -> str:
    """Classify a customer complaint using diverse few-shot examples."""

    examples = [
        ("I was charged twice for my subscription", "billing"),
        ("The app crashes when I open settings", "technical"),
        ("Delivery arrived 3 days late with damaged box", "shipping"),
        ("Would love Slack integration", "feature_request"),
        ("How do I export my data?", "how_to"),
    ]

    examples_text = "\n\n".join(
        f"Complaint: \"{text}\"\nCategory: {label}"
        for text, label in examples
    )

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=32,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"""Classify the customer complaint into one of these
categories: billing, technical, shipping, feature_request, how_to.

{examples_text}

Complaint: "{text}"
Category:"""
            }
        ]
    )

    return message.content[0].text.strip()
```

### 3.2 Similarity Strategy

Select examples that are semantically similar to the query. The intuition is that similar inputs require similar processing, so relevant examples provide the most useful signal.

```python
import anthropic
import numpy as np
from typing import Any

client = anthropic.Anthropic()

def get_embedding(text: str) -> list[float]:
    """Get text embedding using a hypothetical embedding API.

    In practice, use an embedding model like:
    - Anthropic's Voyage embeddings
    - OpenAI's text-embedding-3-small
    - Sentence-transformers (local)
    """
    # Placeholder: in production, call an embedding API
    # response = embedding_client.embed(text)
    # return response.embedding
    pass

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr, b_arr = np.array(a), np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

def select_similar_examples(
    query: str,
    example_pool: list[dict[str, Any]],
    k: int = 3
) -> list[dict[str, Any]]:
    """Select the k most similar examples to the query.

    Args:
        query: The input text to classify
        example_pool: List of dicts with 'text', 'label', 'embedding' keys
        k: Number of examples to select

    Returns:
        The k most similar examples, sorted by similarity (descending)
    """
    query_embedding = get_embedding(query)

    scored_examples = []
    for example in example_pool:
        sim = cosine_similarity(query_embedding, example["embedding"])
        scored_examples.append((sim, example))

    scored_examples.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored_examples[:k]]
```

### 3.3 Representativeness Strategy

Select examples that represent the typical distribution of your data. If 60% of your inputs are category A, ensure your examples reflect that proportion.

```python
import random

def select_representative_examples(
    example_pool: list[dict],
    k: int = 6,
    category_distribution: dict[str, float] | None = None
) -> list[dict]:
    """Select examples that match the expected input distribution.

    Args:
        example_pool: All available examples
        k: Total number of examples to select
        category_distribution: Expected proportion per category
            e.g., {"billing": 0.4, "technical": 0.3, "shipping": 0.2, "other": 0.1}
    """
    if category_distribution is None:
        # Infer distribution from the pool
        categories = [ex["label"] for ex in example_pool]
        total = len(categories)
        category_distribution = {}
        for cat in set(categories):
            category_distribution[cat] = categories.count(cat) / total

    selected = []
    for category, proportion in category_distribution.items():
        n_examples = max(1, round(k * proportion))
        pool_for_category = [
            ex for ex in example_pool if ex["label"] == category
        ]
        chosen = random.sample(
            pool_for_category,
            min(n_examples, len(pool_for_category))
        )
        selected.extend(chosen)

    # Trim to exactly k if rounding caused overshoot
    return selected[:k]
```

### 3.4 Boundary Examples Strategy

Include examples that sit near decision boundaries — cases where the correct label might be surprising or ambiguous. These examples provide the most informative signal.

```python
import anthropic

client = anthropic.Anthropic()

# Boundary examples: cases where the classification is not obvious
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=64,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """Classify each text as "spam" or "not_spam".

Text: "You won a $1000 gift card! Click here!"
Label: spam

Text: "Meeting rescheduled to 3pm tomorrow"
Label: not_spam

Text: "Your order #4521 has shipped, track it here: company.com/track/4521"
Label: not_spam

Text: "URGENT: Your account will be suspended. Verify your info NOW"
Label: spam

Text: "Hi! Long time no see. I started a new business and thought of you. Want to grab coffee to discuss an opportunity?"
Label: spam

Text: "Reminder: Your subscription renews tomorrow for $9.99"
Label: not_spam

Text: "Congratulations on your work anniversary! Your team sent you a gift at company.com/gifts/12345"
Label:"""
        }
    ]
)
```

The boundary examples here are critical. Examples 3 and 6 contain links and urgency language but are legitimate. Example 5 is a social engineering attempt disguised as a friendly message. These boundary cases teach the model your specific classification criteria.

---

## 4. Example Ordering and Bias Effects

### 4.1 Recency Bias

Research has shown that LLMs can be influenced by the order of few-shot examples, particularly the last example. This is called **recency bias** — the model over-weights the most recent example.

```python
import anthropic

client = anthropic.Anthropic()

# Recency bias demonstration
# If the last few examples all have the same label,
# the model may be biased toward that label

# BIASED ordering: last 3 examples are all "positive"
biased_prompt = """Classify sentiment as positive, negative, or neutral.

Text: "The service was terrible and the food was cold."
Sentiment: negative

Text: "Nothing special about this place."
Sentiment: neutral

Text: "Absolutely wonderful experience from start to finish!"
Sentiment: positive

Text: "The staff was incredibly friendly."
Sentiment: positive

Text: "Great value for the price."
Sentiment: positive

Text: "The room was acceptable but overpriced for what you get."
Sentiment:"""
# Recency bias may push toward "positive" even though this is "negative/mixed"

# BALANCED ordering: alternate labels to reduce bias
balanced_prompt = """Classify sentiment as positive, negative, or neutral.

Text: "Absolutely wonderful experience from start to finish!"
Sentiment: positive

Text: "The service was terrible and the food was cold."
Sentiment: negative

Text: "Nothing special about this place."
Sentiment: neutral

Text: "Great value for the price."
Sentiment: positive

Text: "I would never recommend this to anyone."
Sentiment: negative

Text: "The room was acceptable but overpriced for what you get."
Sentiment:"""
# More balanced — the model considers all labels equally
```

### 4.2 Primacy Bias

The first examples in a few-shot prompt can also anchor the model's interpretation of the task. Make your first example a "gold standard" that perfectly represents the task.

```python
import anthropic

client = anthropic.Anthropic()

# Strategy: Put your clearest, most representative example first
# It anchors the model's understanding of the task pattern

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": """Extract the action items from meeting notes.

Meeting notes: "We need to finalize the Q3 budget by Friday. Sarah will
prepare the marketing deck. Tom should follow up with the vendor about
pricing. Everyone should review the draft proposal before Monday."
Action items:
1. Finalize Q3 budget — Deadline: Friday — Owner: (unspecified)
2. Prepare marketing deck — Deadline: (unspecified) — Owner: Sarah
3. Follow up with vendor about pricing — Deadline: (unspecified) — Owner: Tom
4. Review draft proposal — Deadline: before Monday — Owner: Everyone

Meeting notes: "Quick sync - the deployment is blocked by the DNS issue.
James is investigating. We'll reconvene at 2pm if not resolved.
Also, please update your timesheets."
Action items:"""
        }
    ]
)
```

### 4.3 Mitigating Ordering Effects

```python
import random
from typing import Any

def create_few_shot_prompt(
    examples: list[dict[str, Any]],
    query: str,
    task_description: str,
    shuffle: bool = True,
    ensure_label_alternation: bool = True
) -> str:
    """Create a few-shot prompt with ordering bias mitigation.

    Args:
        examples: List of dicts with 'input' and 'label' keys
        query: The input to classify
        task_description: Description of the task
        shuffle: Whether to randomize order (good for repeated calls)
        ensure_label_alternation: Whether to alternate labels
    """

    if ensure_label_alternation:
        # Group by label, then interleave
        by_label: dict[str, list] = {}
        for ex in examples:
            by_label.setdefault(ex["label"], []).append(ex)

        ordered = []
        while any(by_label.values()):
            for label in list(by_label.keys()):
                if by_label[label]:
                    ordered.append(by_label[label].pop(0))
                else:
                    del by_label[label]
        examples = ordered
    elif shuffle:
        examples = examples.copy()
        random.shuffle(examples)

    # Build prompt
    prompt_parts = [task_description, ""]
    for ex in examples:
        prompt_parts.append(f"Input: {ex['input']}")
        prompt_parts.append(f"Output: {ex['label']}")
        prompt_parts.append("")

    prompt_parts.append(f"Input: {query}")
    prompt_parts.append("Output:")

    return "\n".join(prompt_parts)
```

---

## 5. Dynamic Few-Shot with Semantic Search

Static examples work for simple tasks, but dynamic example selection — choosing examples on-the-fly based on the query — dramatically improves performance on diverse inputs.

### 5.1 Architecture

The dynamic few-shot pipeline:
1. Maintain an **example store** with pre-computed embeddings
2. When a new query arrives, compute its embedding
3. Retrieve the K most similar examples
4. Construct the prompt with those examples
5. Send to the LLM

```python
import anthropic
import numpy as np
from dataclasses import dataclass

client = anthropic.Anthropic()

@dataclass
class Example:
    text: str
    label: str
    embedding: list[float]

class DynamicFewShotClassifier:
    """Classifier that selects examples dynamically based on query similarity."""

    def __init__(
        self,
        example_store: list[Example],
        k: int = 5,
        task_description: str = ""
    ):
        self.example_store = example_store
        self.k = k
        self.task_description = task_description
        # Pre-compute embedding matrix for fast similarity search
        self.embedding_matrix = np.array(
            [ex.embedding for ex in example_store]
        )

    def _find_similar(self, query_embedding: list[float]) -> list[Example]:
        """Find the k most similar examples using cosine similarity."""
        query_vec = np.array(query_embedding)

        # Batch cosine similarity
        norms = np.linalg.norm(self.embedding_matrix, axis=1)
        query_norm = np.linalg.norm(query_vec)
        similarities = self.embedding_matrix @ query_vec / (norms * query_norm)

        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-self.k:][::-1]
        return [self.example_store[i] for i in top_k_indices]

    def _build_prompt(self, examples: list[Example], query: str) -> str:
        """Build the few-shot prompt from selected examples."""
        parts = [self.task_description, ""]

        for ex in examples:
            parts.append(f"Text: \"{ex.text}\"")
            parts.append(f"Label: {ex.label}")
            parts.append("")

        parts.append(f"Text: \"{query}\"")
        parts.append("Label:")

        return "\n".join(parts)

    def classify(self, query: str, query_embedding: list[float]) -> str:
        """Classify a query using dynamically selected few-shot examples."""
        similar_examples = self._find_similar(query_embedding)
        prompt = self._build_prompt(similar_examples, query)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=32,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )

        return message.content[0].text.strip()
```

### 5.2 Building the Example Store

```python
import json
from pathlib import Path

def build_example_store(
    labeled_data_path: str,
    embedding_function: callable
) -> list[Example]:
    """Build an example store with pre-computed embeddings.

    Args:
        labeled_data_path: Path to a JSONL file with 'text' and 'label' fields
        embedding_function: Function that takes text and returns an embedding vector

    Returns:
        List of Example objects with embeddings
    """
    examples = []

    with open(labeled_data_path) as f:
        for line in f:
            data = json.loads(line)
            embedding = embedding_function(data["text"])
            examples.append(Example(
                text=data["text"],
                label=data["label"],
                embedding=embedding
            ))

    return examples

# Usage pattern:
# 1. Prepare labeled data as JSONL
# 2. Build the store once (cache embeddings to disk)
# 3. Load at runtime for fast retrieval
#
# store = build_example_store("labeled_tickets.jsonl", get_embedding)
# classifier = DynamicFewShotClassifier(
#     example_store=store,
#     k=5,
#     task_description="Classify the support ticket into a category."
# )
# result = classifier.classify("My dashboard is loading slowly", query_embedding)
```

### 5.3 Using a Vector Database for Scale

For production systems with thousands of examples, use a vector database instead of in-memory search:

```python
# Example using ChromaDB (lightweight, local)
import chromadb

def create_vector_store(labeled_data: list[dict]) -> chromadb.Collection:
    """Create a ChromaDB collection for few-shot example retrieval."""

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="few_shot_examples",
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        documents=[d["text"] for d in labeled_data],
        metadatas=[{"label": d["label"]} for d in labeled_data],
        ids=[f"ex_{i}" for i in range(len(labeled_data))]
    )

    return collection

def retrieve_examples(
    collection: chromadb.Collection,
    query: str,
    k: int = 5
) -> list[dict]:
    """Retrieve the k most similar examples from the vector store."""

    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas"]
    )

    examples = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        examples.append({"text": doc, "label": meta["label"]})

    return examples
```

---

## 6. K-Shot Selection Optimization

### 6.1 How Many Examples?

The optimal number of examples (K) depends on several factors:

| Factor | Fewer Examples (1-3) | More Examples (5-10+) |
|--------|---------------------|----------------------|
| Task complexity | Simple, well-defined tasks | Complex, nuanced tasks |
| Label set | Binary (yes/no) | Multi-class (10+ categories) |
| Model capability | Large, instruction-tuned models | Smaller models |
| Token budget | Limited context window | Large context window |
| Output format | Simple (label, number) | Complex (JSON, structured) |

```python
import anthropic
from typing import Any

client = anthropic.Anthropic()

def evaluate_k_shots(
    test_set: list[dict[str, Any]],
    example_pool: list[dict[str, Any]],
    k_values: list[int],
    task_prompt_template: str
) -> dict[int, float]:
    """Evaluate classification accuracy for different K values.

    Returns a dict mapping K to accuracy percentage.
    """
    results = {}

    for k in k_values:
        correct = 0
        total = len(test_set)

        for test_item in test_set:
            # Select k examples (could be random, similar, etc.)
            selected_examples = example_pool[:k]

            # Build prompt
            examples_text = "\n".join(
                f"Input: {ex['text']}\nOutput: {ex['label']}"
                for ex in selected_examples
            )

            prompt = f"""{task_prompt_template}

{examples_text}

Input: {test_item['text']}
Output:"""

            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=32,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )

            predicted = message.content[0].text.strip().lower()
            expected = test_item["label"].lower()

            if predicted == expected:
                correct += 1

        results[k] = round(correct / total * 100, 1)

    return results

# Example usage:
# results = evaluate_k_shots(
#     test_set=test_data,
#     example_pool=training_data,
#     k_values=[0, 1, 3, 5, 10],
#     task_prompt_template="Classify the sentiment as positive, negative, or neutral."
# )
# print(results)
# {0: 75.0, 1: 82.0, 3: 88.0, 5: 91.0, 10: 91.5}
# -> Diminishing returns after K=5 for this task
```

### 6.2 Diminishing Returns

Beyond a certain K, additional examples provide marginal benefit but consume valuable tokens. The typical pattern:

- **K=0 to K=1**: Largest jump (demonstrates format)
- **K=1 to K=3**: Significant improvement (demonstrates pattern)
- **K=3 to K=5**: Moderate improvement (covers edge cases)
- **K=5 to K=10**: Small improvement (diminishing returns)
- **K>10**: Often negligible improvement, wastes tokens

```python
# Rule of thumb: Start with K=3, increase if accuracy is insufficient
# For classification: K=3-5 is usually optimal
# For generation: K=2-3 is usually sufficient
# For format specification: K=1 is often enough
```

### 6.3 Negative Examples

Including negative examples — showing what the output should NOT look like — can be highly effective for tasks where the model tends to make a specific type of mistake.

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
            "content": """Generate a professional email subject line for the given context.

Context: Following up on a sales meeting from last week
Good: "Follow-up: Action Items from Our March 10 Meeting"
Bad: "Just Checking In!!!" (too casual, no specifics, excessive punctuation)

Context: Announcing a product launch to customers
Good: "Introducing CloudSync Pro — Available March 20"
Bad: "YOU WON'T BELIEVE OUR NEW PRODUCT!!!" (clickbait, all caps, no information)

Context: Requesting a deadline extension from a client
Good:"""
        }
    ]
)
```

---

## 7. Label Balance in Examples

### 7.1 The Balance Problem

If your few-shot examples are unbalanced (e.g., 4 positive examples and 1 negative), the model will be biased toward the majority label. This is a form of base-rate manipulation.

```python
import anthropic

client = anthropic.Anthropic()

# UNBALANCED: 4 positive, 1 negative examples
# This biases the model toward predicting "positive"
unbalanced = """Classify as positive or negative.

"Great product!" -> positive
"Love it!" -> positive
"Works perfectly" -> positive
"Highly recommend" -> positive
"Broke after one day" -> negative

"It's okay I guess" ->"""

# BALANCED: 3 positive, 3 negative examples
# Fair representation of both classes
balanced = """Classify as positive or negative.

"Great product!" -> positive
"Broke after one day" -> negative
"Love it!" -> positive
"Terrible customer service" -> negative
"Works perfectly" -> positive
"Complete waste of money" -> negative

"It's okay I guess" ->"""
```

### 7.2 Handling Imbalanced Real-World Distributions

In production, your actual data might be imbalanced (e.g., 95% non-spam, 5% spam). Your few-shot examples should balance between reflecting reality and avoiding bias:

```python
def select_balanced_examples(
    example_pool: list[dict],
    k_per_class: int = 2,
    classes: list[str] | None = None
) -> list[dict]:
    """Select an equal number of examples per class.

    For few-shot prompting, balanced examples are usually better
    even if the real distribution is imbalanced. The task description
    can convey base rates instead.
    """
    if classes is None:
        classes = list(set(ex["label"] for ex in example_pool))

    balanced = []
    for cls in classes:
        cls_examples = [ex for ex in example_pool if ex["label"] == cls]
        # Prefer diverse examples within each class
        selected = cls_examples[:k_per_class]
        balanced.extend(selected)

    return balanced

# If base rates matter, state them in the task description:
task_description = """Classify emails as spam or not_spam.
Note: In our system, approximately 5% of emails are spam.
Only classify as spam if you are confident.

Examples:
"""
```

### 7.3 Multi-Class Balancing

For tasks with many classes, you may not have token budget for equal representation. Prioritize:

1. Classes that are easily confused with each other
2. Rare classes that the model might forget exist
3. Classes with non-obvious boundaries

```python
# With 10 classes but budget for only 6 examples:
# Don't try to show all 10 classes — show the 6 most important

priority_classes = [
    # Classes that are often confused
    ("technical_bug", "The app crashes when I upload large files"),
    ("technical_perf", "The app is very slow when uploading files"),
    # Rare class the model might miss
    ("security", "I found an XSS vulnerability on the login page"),
    # Clear examples of common classes
    ("billing", "I was charged twice this month"),
    ("feature_req", "Can you add dark mode?"),
    ("account", "I need to change my email address"),
]
# List remaining classes in the task description so the model knows they exist
```

---

## 8. Token Budget Management

### 8.1 The Token Tradeoff

Every token spent on examples is a token not available for the response or additional context. Managing this budget is critical, especially with shorter context windows.

```python
def estimate_few_shot_budget(
    system_prompt_chars: int,
    task_description_chars: int,
    avg_example_chars: int,
    num_examples: int,
    query_chars: int,
    max_response_tokens: int,
    model_context_window: int = 200000,
    chars_per_token: int = 4
) -> dict:
    """Calculate token budget allocation for a few-shot prompt."""

    system_tokens = system_prompt_chars // chars_per_token
    task_tokens = task_description_chars // chars_per_token
    example_tokens = (avg_example_chars * num_examples) // chars_per_token
    query_tokens = query_chars // chars_per_token

    total_input = system_tokens + task_tokens + example_tokens + query_tokens
    total_used = total_input + max_response_tokens
    remaining = model_context_window - total_used

    return {
        "input_breakdown": {
            "system": system_tokens,
            "task_description": task_tokens,
            "examples": example_tokens,
            "query": query_tokens,
            "total_input": total_input,
        },
        "response_reserved": max_response_tokens,
        "total_used": total_used,
        "remaining": remaining,
        "max_additional_examples": remaining // (avg_example_chars // chars_per_token)
    }

# Example calculation
budget = estimate_few_shot_budget(
    system_prompt_chars=500,
    task_description_chars=200,
    avg_example_chars=300,  # ~75 tokens per example
    num_examples=5,
    query_chars=200,
    max_response_tokens=1024,
    model_context_window=200000
)
print(f"Input tokens: {budget['input_breakdown']['total_input']}")
print(f"Remaining budget: {budget['remaining']} tokens")
```

### 8.2 Compression Strategies

When examples are long, compress them without losing essential information:

```python
import anthropic

client = anthropic.Anthropic()

# UNCOMPRESSED: Full-length examples (expensive)
uncompressed_example = """
Input: "Dear Customer Support, I am writing to express my extreme
dissatisfaction with the product I received on January 15th, 2025.
The item arrived with significant damage to the packaging, and upon
opening it, I discovered that the screen was cracked. I have been
a loyal customer for over 5 years and this is the first time I've
had such a negative experience. I would like a full refund or
replacement shipped immediately. Please respond within 24 hours.
Regards, John Smith"
Output: {"category": "product_damage", "sentiment": "negative",
"urgency": "high", "action": "refund_or_replace"}
"""

# COMPRESSED: Essential patterns preserved, reduced tokens
compressed_example = """
Input: "Product arrived damaged (cracked screen). Requesting refund or replacement. Long-time customer, wants response within 24h."
Output: {"category": "product_damage", "sentiment": "negative", "urgency": "high", "action": "refund_or_replace"}
"""

# The compressed version uses ~40% fewer tokens while preserving
# the classification-relevant information
```

### 8.3 Truncation Strategies

When you cannot fit all desired examples, use intelligent truncation:

```python
def fit_examples_to_budget(
    examples: list[dict],
    query: str,
    max_input_tokens: int,
    task_description: str,
    chars_per_token: int = 4
) -> list[dict]:
    """Select examples that fit within the token budget.

    Strategy: Add examples in priority order until budget is exhausted.
    Examples should be pre-sorted by priority (similarity, diversity, etc.).
    """

    # Fixed costs
    fixed_chars = len(task_description) + len(query) + 100  # overhead
    fixed_tokens = fixed_chars // chars_per_token
    available_tokens = max_input_tokens - fixed_tokens

    selected = []
    used_tokens = 0

    for example in examples:
        example_text = f"Input: {example['text']}\nOutput: {example['label']}\n\n"
        example_tokens = len(example_text) // chars_per_token

        if used_tokens + example_tokens <= available_tokens:
            selected.append(example)
            used_tokens += example_tokens
        else:
            break  # No more budget

    return selected
```

---

## 9. Zero-Shot vs Few-Shot: Task Comparison

### 9.1 Decision Framework

| Task Type | Recommended | Reasoning |
|-----------|-------------|-----------|
| Standard classification (sentiment, topic) | Zero-shot | Well-represented in training data |
| Custom taxonomy classification | Few-shot | Model needs to learn your specific labels |
| Data extraction (standard entities) | Zero-shot | NER is well-trained |
| Data extraction (custom formats) | Few-shot | Format requires demonstration |
| Translation | Zero-shot | Well-trained, examples rarely help |
| Code generation | Zero-shot | Strong instruction following |
| Text summarization | Zero-shot | Well-trained, but few-shot helps with style |
| Style transfer | Few-shot | Style is hard to describe, easy to demonstrate |
| Complex structured output | Few-shot | JSON schemas need concrete examples |
| Reasoning tasks | Zero-shot (with CoT) | Examples of reasoning less helpful than instructions |

### 9.2 Empirical Comparison

```python
import anthropic
from typing import Any

client = anthropic.Anthropic()

def compare_zero_vs_few_shot(
    test_cases: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    task_description: str
) -> dict[str, float]:
    """Compare zero-shot and few-shot accuracy on a test set."""

    zero_shot_correct = 0
    few_shot_correct = 0

    examples_text = "\n".join(
        f"Input: {ex['text']}\nOutput: {ex['label']}"
        for ex in examples
    )

    for test in test_cases:
        # Zero-shot
        zero_prompt = f"{task_description}\n\nInput: {test['text']}\nOutput:"
        zero_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=32,
            temperature=0.0,
            messages=[{"role": "user", "content": zero_prompt}]
        )
        zero_pred = zero_msg.content[0].text.strip().lower()

        # Few-shot
        few_prompt = (
            f"{task_description}\n\n{examples_text}\n\n"
            f"Input: {test['text']}\nOutput:"
        )
        few_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=32,
            temperature=0.0,
            messages=[{"role": "user", "content": few_prompt}]
        )
        few_pred = few_msg.content[0].text.strip().lower()

        expected = test["label"].lower()
        if zero_pred == expected:
            zero_shot_correct += 1
        if few_pred == expected:
            few_shot_correct += 1

    total = len(test_cases)
    return {
        "zero_shot_accuracy": round(zero_shot_correct / total * 100, 1),
        "few_shot_accuracy": round(few_shot_correct / total * 100, 1),
        "improvement": round(
            (few_shot_correct - zero_shot_correct) / total * 100, 1
        )
    }
```

### 9.3 Hybrid Approaches

The best approach is often a hybrid that uses zero-shot for simple cases and dynamically adds examples for harder ones:

```python
import anthropic

client = anthropic.Anthropic()

def adaptive_classify(
    query: str,
    example_store: list[dict],
    task_description: str,
    confidence_threshold: float = 0.8
) -> dict:
    """Try zero-shot first; fall back to few-shot if confidence is low."""

    # Step 1: Zero-shot attempt
    zero_shot_prompt = f"""{task_description}

Input: {query}

Respond with a JSON object: {{"label": "...", "confidence": 0.0-1.0}}"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=64,
        temperature=0.0,
        messages=[{"role": "user", "content": zero_shot_prompt}]
    )

    import json
    try:
        result = json.loads(message.content[0].text)
    except json.JSONDecodeError:
        result = {"label": "unknown", "confidence": 0.0}

    if result.get("confidence", 0) >= confidence_threshold:
        result["method"] = "zero_shot"
        return result

    # Step 2: Few-shot fallback with similar examples
    # (In production, use embedding similarity to select examples)
    examples_text = "\n".join(
        f"Input: {ex['text']}\nOutput: {ex['label']}"
        for ex in example_store[:5]
    )

    few_shot_prompt = f"""{task_description}

{examples_text}

Input: {query}
Output:"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=32,
        temperature=0.0,
        messages=[{"role": "user", "content": few_shot_prompt}]
    )

    return {
        "label": message.content[0].text.strip(),
        "confidence": None,
        "method": "few_shot_fallback"
    }
```

---

## Exercises

### Exercise 1: Zero-Shot Design

Design a zero-shot prompt that classifies a customer email into one of exactly seven categories: `billing`, `technical`, `shipping`, `account`, `feedback`, `partnership`, `other`. The prompt should work reliably without any examples. Include explicit handling for ambiguous cases.

<details><summary>Show Answer</summary>

```python
import anthropic

client = anthropic.Anthropic()

def classify_email_zero_shot(email_text: str) -> str:
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=64,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"""Classify this customer email into exactly ONE category.

Categories and their definitions:
- billing: Charges, invoices, refunds, payment methods, pricing, subscriptions
- technical: Bugs, errors, crashes, performance issues, integration problems
- shipping: Delivery status, tracking, damaged packages, shipping addresses
- account: Login issues, profile changes, password resets, account deletion
- feedback: Compliments, complaints about experience, suggestions (not feature requests)
- partnership: Business proposals, affiliate programs, collaboration inquiries
- other: Anything that does not clearly fit the above categories

Rules:
1. Choose the SINGLE most relevant category
2. If an email covers multiple categories, pick the one that represents the primary intent
3. Use "other" ONLY if none of the first 6 categories apply
4. Return ONLY the category name, nothing else

Email:
<email>
{email_text}
</email>

Category:"""
            }
        ]
    )
    return message.content[0].text.strip()
```

Key techniques:
- Each category has a clear definition (reduces ambiguity)
- Explicit rule for multi-category emails (primary intent)
- "other" is a catch-all with a usage rule (prevents overuse)
- Output format is constrained to a single word
- Email is delimited with XML tags

</details>

### Exercise 2: Few-Shot Example Selection

Given the following pool of 10 labeled examples for a toxicity classifier, select the optimal 5 examples for a few-shot prompt. Explain your selection criteria.

```
1. "I hope you have a great day!" -> not_toxic
2. "You're an absolute idiot and should be banned" -> toxic
3. "The weather is nice" -> not_toxic
4. "Kill yourself loser" -> toxic
5. "I disagree with your point about tax policy" -> not_toxic
6. "Your argument is trash, just like your face" -> toxic
7. "Maybe try reading a book sometime, just saying" -> not_toxic
8. "I think we can improve the process by adding reviews" -> not_toxic
9. "People like you are what's wrong with this world" -> toxic
10. "Shut up nobody asked for your stupid opinion" -> toxic
```

<details><summary>Show Answer</summary>

**Selected examples (with reasoning):**

1. **Example 5**: "I disagree with your point about tax policy" -> not_toxic
   *Boundary case*: Disagreement is not toxicity. This is the most important example because models often incorrectly flag disagreement as toxic.

2. **Example 6**: "Your argument is trash, just like your face" -> toxic
   *Boundary case*: Starts with an opinion about an argument (could be not_toxic) but escalates to a personal attack. Teaches the model where the line is.

3. **Example 7**: "Maybe try reading a book sometime, just saying" -> not_toxic
   *Boundary case*: This is passive-aggressive and condescending but not toxic by most guidelines. Critical for calibrating the model's threshold.

4. **Example 9**: "People like you are what's wrong with this world" -> toxic
   *Boundary case*: No profanity, but dehumanizing. Teaches that toxicity is not just about bad words.

5. **Example 2**: "You're an absolute idiot and should be banned" -> toxic
   *Clear positive example*: Direct insult with clear toxic intent. Anchors the "obviously toxic" end of the spectrum.

**Why these 5?**

- **Label balance**: 2 not_toxic, 3 toxic (slight imbalance toward toxic because the model's default is to classify things as non-toxic — we want it to be sensitive to toxicity)
- **Boundary cases**: 4 of 5 examples are near the decision boundary, which is where the model needs the most guidance
- **Diversity of toxic patterns**: personal insult (#2), personal attack mixed with opinion (#6), dehumanization (#9)
- **Avoided**: Examples 1, 3, 8 (too obvious — no learning signal), Example 4 (extreme, not representative of typical borderline cases)

</details>

### Exercise 3: Dynamic Few-Shot Implementation

Write a function that implements dynamic few-shot example selection using cosine similarity. The function should accept a query string, an example pool (list of dicts with `text`, `label`, and `embedding` fields), and return a formatted few-shot prompt with the top-K most similar examples, ensuring label balance.

<details><summary>Show Answer</summary>

```python
import numpy as np
from collections import Counter

def dynamic_few_shot_prompt(
    query: str,
    query_embedding: list[float],
    example_pool: list[dict],
    task_description: str,
    k: int = 6,
    max_per_label: int | None = None
) -> str:
    """Build a few-shot prompt with dynamically selected, label-balanced examples.

    Args:
        query: The input text to classify
        query_embedding: Pre-computed embedding for the query
        example_pool: List of dicts with 'text', 'label', 'embedding' keys
        task_description: Description of the classification task
        k: Total number of examples to include
        max_per_label: Maximum examples per label (for balance).
                       If None, set to ceil(k / num_labels)

    Returns:
        A formatted few-shot prompt string
    """
    # Step 1: Compute similarities
    query_vec = np.array(query_embedding)
    query_norm = np.linalg.norm(query_vec)

    scored = []
    for ex in example_pool:
        ex_vec = np.array(ex["embedding"])
        similarity = float(
            np.dot(query_vec, ex_vec) / (query_norm * np.linalg.norm(ex_vec))
        )
        scored.append((similarity, ex))

    # Step 2: Sort by similarity (descending)
    scored.sort(key=lambda x: x[0], reverse=True)

    # Step 3: Select with label balance
    all_labels = set(ex["label"] for ex in example_pool)
    if max_per_label is None:
        max_per_label = -(-k // len(all_labels))  # Ceiling division

    label_counts: Counter = Counter()
    selected = []

    for sim, ex in scored:
        label = ex["label"]
        if label_counts[label] < max_per_label and len(selected) < k:
            selected.append(ex)
            label_counts[label] += 1

    # Step 4: If we haven't reached k examples, fill from remaining
    if len(selected) < k:
        selected_texts = {ex["text"] for ex in selected}
        for sim, ex in scored:
            if ex["text"] not in selected_texts and len(selected) < k:
                selected.append(ex)
                selected_texts.add(ex["text"])

    # Step 5: Interleave labels to reduce recency bias
    by_label: dict[str, list] = {}
    for ex in selected:
        by_label.setdefault(ex["label"], []).append(ex)

    interleaved = []
    while any(by_label.values()):
        for label in sorted(by_label.keys()):
            if by_label[label]:
                interleaved.append(by_label[label].pop(0))

    # Step 6: Build prompt
    parts = [task_description, ""]
    for ex in interleaved:
        parts.append(f'Text: "{ex["text"]}"')
        parts.append(f'Label: {ex["label"]}')
        parts.append("")

    parts.append(f'Text: "{query}"')
    parts.append("Label:")

    return "\n".join(parts)
```

Key design decisions:
1. **Similarity-first selection** ensures relevance
2. **Label balancing** prevents majority-class bias
3. **Interleaving** by label reduces recency bias
4. **Fallback filling** ensures we always reach K examples
5. **Configurable max_per_label** allows tuning the balance

</details>

### Exercise 4: Token Budget Optimization

You have a 4096-token context window (small model). Your system prompt uses 200 tokens, you need 512 tokens for the response, and your query averages 50 tokens. Each few-shot example averages 80 tokens. Calculate: (a) the maximum number of examples you can fit, (b) if you compress examples to 50 tokens each, how many more can you fit, and (c) write a function that dynamically determines K based on query length.

<details><summary>Show Answer</summary>

**(a) Maximum examples at 80 tokens each:**
```
Available = 4096 - 200 (system) - 512 (response) - 50 (query) = 3334 tokens
Max examples = 3334 // 80 = 41 examples
```

**(b) Compressed examples at 50 tokens each:**
```
Max examples = 3334 // 50 = 66 examples
Additional examples = 66 - 41 = 25 more examples
```

**(c) Dynamic K function:**

```python
def calculate_dynamic_k(
    context_window: int,
    system_prompt_tokens: int,
    query_tokens: int,
    max_response_tokens: int,
    tokens_per_example: int,
    task_description_tokens: int = 50,
    min_k: int = 1,
    max_k: int = 20,
    safety_margin: float = 0.95  # Use only 95% of available space
) -> int:
    """Dynamically calculate how many few-shot examples to include.

    Args:
        context_window: Total model context window size
        system_prompt_tokens: Tokens used by system prompt
        query_tokens: Tokens used by the current query
        max_response_tokens: Tokens reserved for model response
        tokens_per_example: Average tokens per few-shot example
        task_description_tokens: Tokens for the task description
        min_k: Minimum number of examples
        max_k: Maximum number of examples (even if budget allows more)
        safety_margin: Fraction of available budget to use (prevents overflow)

    Returns:
        Optimal number of examples K
    """
    # Calculate available budget
    fixed_costs = (
        system_prompt_tokens
        + query_tokens
        + max_response_tokens
        + task_description_tokens
    )

    available = context_window - fixed_costs
    safe_available = int(available * safety_margin)

    if safe_available <= 0:
        return min_k  # Barely enough space, use minimum examples

    # Calculate K
    k = safe_available // tokens_per_example

    # Clamp to [min_k, max_k]
    return max(min_k, min(k, max_k))


# Example usage with varying query lengths
for query_len in [50, 200, 500, 1000, 2000]:
    k = calculate_dynamic_k(
        context_window=4096,
        system_prompt_tokens=200,
        query_tokens=query_len,
        max_response_tokens=512,
        tokens_per_example=80,
        task_description_tokens=50,
        min_k=1,
        max_k=15
    )
    print(f"Query length: {query_len} tokens -> K={k} examples")

# Output:
# Query length: 50 tokens -> K=15 examples  (budget allows 41, capped at 15)
# Query length: 200 tokens -> K=15 examples  (budget allows 39, capped at 15)
# Query length: 500 tokens -> K=15 examples  (budget allows 35, capped at 15)
# Query length: 1000 tokens -> K=15 examples (budget allows 29, capped at 15)
# Query length: 2000 tokens -> K=15 examples (budget allows 16, capped at 15)
```

</details>

### Exercise 5: Comparative Analysis

Design an experiment that compares zero-shot, 3-shot, and 5-shot prompting on an email intent classification task with 5 categories. Write the complete evaluation code (mock the API calls) that calculates accuracy, per-class precision, and generates a results summary table.

<details><summary>Show Answer</summary>

```python
import anthropic
from collections import defaultdict

client = anthropic.Anthropic()

CATEGORIES = ["billing", "technical", "shipping", "feedback", "account"]

# Test dataset (in production, use hundreds of examples)
TEST_SET = [
    {"text": "I was charged twice for order #4521", "label": "billing"},
    {"text": "The app crashes when I try to upload photos", "label": "technical"},
    {"text": "Where is my package? It's been 2 weeks", "label": "shipping"},
    {"text": "Your new UI design is fantastic", "label": "feedback"},
    {"text": "I need to change my email address", "label": "account"},
    {"text": "Can I get a refund for the annual subscription?", "label": "billing"},
    {"text": "Error 500 when accessing the dashboard", "label": "technical"},
    {"text": "The box arrived damaged", "label": "shipping"},
    {"text": "I think the search feature could be improved", "label": "feedback"},
    {"text": "How do I enable two-factor authentication?", "label": "account"},
]

# Example pool for few-shot (disjoint from test set)
EXAMPLE_POOL = [
    {"text": "My credit card was declined but I got charged", "label": "billing"},
    {"text": "The website is very slow today", "label": "technical"},
    {"text": "Can I change my delivery address?", "label": "shipping"},
    {"text": "Great customer service experience", "label": "feedback"},
    {"text": "I forgot my password and can't reset it", "label": "account"},
]

TASK_DESC = (
    "Classify the email into one of these categories: "
    + ", ".join(CATEGORIES) + "."
)

def build_prompt(query: str, examples: list[dict], task: str) -> str:
    """Build a classification prompt with optional examples."""
    parts = [task, ""]
    for ex in examples:
        parts.append(f'Email: "{ex["text"]}"')
        parts.append(f"Category: {ex['label']}")
        parts.append("")
    parts.append(f'Email: "{query}"')
    parts.append("Category:")
    return "\n".join(parts)

def run_experiment(
    test_set: list[dict],
    example_pool: list[dict],
    k_values: list[int]
) -> dict:
    """Run the comparative experiment across different K values."""

    results = {}

    for k in k_values:
        examples = example_pool[:k]
        predictions = []

        for test_item in test_set:
            prompt = build_prompt(test_item["text"], examples, TASK_DESC)

            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=16,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )

            predicted = message.content[0].text.strip().lower()
            predictions.append({
                "true": test_item["label"],
                "predicted": predicted,
                "correct": predicted == test_item["label"]
            })

        # Calculate metrics
        total = len(predictions)
        correct = sum(1 for p in predictions if p["correct"])
        accuracy = correct / total

        # Per-class precision
        class_metrics = {}
        for cat in CATEGORIES:
            true_positives = sum(
                1 for p in predictions
                if p["predicted"] == cat and p["true"] == cat
            )
            predicted_positives = sum(
                1 for p in predictions if p["predicted"] == cat
            )
            precision = (
                true_positives / predicted_positives
                if predicted_positives > 0
                else 0.0
            )
            class_metrics[cat] = {
                "precision": round(precision, 3),
                "predicted_count": predicted_positives,
                "true_positives": true_positives
            }

        results[k] = {
            "accuracy": round(accuracy * 100, 1),
            "correct": correct,
            "total": total,
            "class_metrics": class_metrics,
            "predictions": predictions
        }

    return results

def print_results_table(results: dict) -> None:
    """Print a formatted results summary table."""

    # Overall accuracy table
    print("=" * 60)
    print(f"{'K-Shot':>8} | {'Accuracy':>10} | {'Correct':>8} | {'Total':>6}")
    print("-" * 60)
    for k, data in sorted(results.items()):
        label = "zero-shot" if k == 0 else f"{k}-shot"
        print(
            f"{label:>8} | {data['accuracy']:>9.1f}% | "
            f"{data['correct']:>8} | {data['total']:>6}"
        )
    print("=" * 60)

    # Per-class precision table
    print("\nPer-Class Precision:")
    print("-" * 60)
    header = f"{'Category':>12}"
    for k in sorted(results.keys()):
        label = "0-shot" if k == 0 else f"{k}-shot"
        header += f" | {label:>8}"
    print(header)
    print("-" * 60)

    for cat in CATEGORIES:
        row = f"{cat:>12}"
        for k in sorted(results.keys()):
            prec = results[k]["class_metrics"][cat]["precision"]
            row += f" | {prec:>8.3f}"
        print(row)
    print("-" * 60)

# Run the experiment
# results = run_experiment(TEST_SET, EXAMPLE_POOL, k_values=[0, 3, 5])
# print_results_table(results)

# Expected output format:
# ============================================================
#   K-Shot |   Accuracy |  Correct |  Total
# ------------------------------------------------------------
# zero-shot |      80.0% |        8 |     10
#    3-shot |      90.0% |        9 |     10
#    5-shot |      90.0% |        9 |     10
# ============================================================
#
# Per-Class Precision:
# ------------------------------------------------------------
#     Category |   0-shot |   3-shot |   5-shot
# ------------------------------------------------------------
#      billing |    1.000 |    1.000 |    1.000
#    technical |    0.667 |    1.000 |    1.000
#     shipping |    1.000 |    1.000 |    1.000
#     feedback |    0.500 |    0.667 |    1.000
#      account |    1.000 |    1.000 |    0.667
# ------------------------------------------------------------
```

Key insights from this type of experiment:
- Zero-shot baseline shows where the model already performs well
- The jump from 0-shot to 3-shot is usually the largest improvement
- Per-class precision reveals which categories benefit most from examples
- Diminishing returns typically appear after K=5 for classification tasks

</details>

---

**Previous**: [Prompt Fundamentals](./01_Prompt_Fundamentals.md) | **Next**: [Chain of Thought](./03_Chain_of_Thought.md)

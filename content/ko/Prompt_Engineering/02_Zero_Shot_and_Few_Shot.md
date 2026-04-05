# 02. 제로샷과 퓨샷 프롬프팅(Zero-Shot and Few-Shot Prompting)

**이전**: [프롬프트 기초](./01_Prompt_Fundamentals.md) | **다음**: [사고의 연쇄](./03_Chain_of_Thought.md)

## 학습 목표

- 제로샷(Zero-shot)과 퓨샷(Few-shot) 프롬프팅을 구분하고 주어진 과제에 적절한 기법을 선택한다
- 다양성, 유사성, 대표성 기준을 사용하여 효과적인 퓨샷 예시를 설계한다
- 의미 검색과 임베딩 유사도를 사용하여 동적 퓨샷 선택을 구현한다
- 제한된 컨텍스트 윈도우에서 퓨샷 프롬프트를 구성할 때 토큰 예산을 관리한다
- 다양한 과제 유형에 걸쳐 제로샷 vs 퓨샷의 성능 트레이드오프를 평가한다

---

제로샷(Zero-shot)과 퓨샷(Few-shot) 프롬프팅은 가장 기본적인 두 가지 프롬프팅 전략입니다. 제로샷은 전적으로 모델의 사전 훈련된 지식과 명령어 따르기 능력에 의존하는 반면, 퓨샷은 원하는 동작을 시연하는 구체적인 입력-출력 예시를 제공합니다. 이 둘 사이에서 선택하고 — 퓨샷을 효과적으로 실행하는 것은 프롬프트 엔지니어링에서 가장 큰 영향력을 가진 기술 중 하나입니다. 이 레슨에서는 두 접근법의 이론, 구현, 최적화를 다룹니다.

## 목차

1. [제로샷 프롬프팅](#1-제로샷-프롬프팅)
2. [퓨샷 프롬프팅 기초](#2-퓨샷-프롬프팅-기초)
3. [예시 선택 전략](#3-예시-선택-전략)
4. [예시 순서와 편향 효과](#4-예시-순서와-편향-효과)
5. [의미 검색을 활용한 동적 퓨샷](#5-의미-검색을-활용한-동적-퓨샷)
6. [K-Shot 선택 최적화](#6-k-shot-선택-최적화)
7. [예시에서의 레이블 균형](#7-예시에서의-레이블-균형)
8. [토큰 예산 관리](#8-토큰-예산-관리)
9. [제로샷 vs 퓨샷: 과제 비교](#9-제로샷-vs-퓨샷-과제-비교)
10. [연습문제](#연습문제)

---

## 1. 제로샷 프롬프팅

제로샷(Zero-shot) 프롬프팅은 예시 없이 — 과제 설명만으로 모델에 과제를 수행하도록 요청하는 것을 의미합니다. 모델은 전적으로 사전 훈련과 명령어 튜닝 중에 학습한 패턴에 의존합니다.

### 1.1 제로샷이 잘 작동하는 경우

제로샷 프롬프팅은 다음과 같은 경우에 성공합니다:
- 과제가 잘 정의되어 있고 훈련 데이터에서 흔히 접하는 경우
- 모델이 강력한 명령어 따르기 능력을 가진 경우
- 출력 형식이 단순하고 모호하지 않은 경우
- 과제가 자연어 이해(분류, 번역, 요약)와 일치하는 경우

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

### 1.2 제로샷이 어려운 경우

제로샷 프롬프팅은 다음과 같은 경우에 어려움을 겪습니다:
- 과제가 모델이 이전에 보지 못한 특정 출력 형식을 요구하는 경우
- 도메인별 관례가 일반 지식과 다른 경우
- 과제가 모호하여 여러 해석이 동등하게 유효한 경우
- 사용자 정의 레이블 분류 체계가 사용되는 경우 (예: 이메일을 "P1"에서 "P5"로 분류)

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

### 1.3 제로샷 성능 향상

제로샷을 사용해야 하는 경우(예: 토큰 제약, 입력의 높은 다양성), 다음 기법들이 신뢰성을 향상시킵니다:

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

## 2. 퓨샷 프롬프팅 기초

퓨샷(Few-shot) 프롬프팅은 실제 쿼리 전에 구체적인 입력-출력 예시를 제공합니다. 이러한 예시는 모델이 따라야 할 패턴을 시연하는 암묵적 지시로 작용합니다.

### 2.1 기본 패턴

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

### 2.2 퓨샷이 작동하는 이유

퓨샷 예시는 **인컨텍스트 학습(In-Context Learning, ICL)**이라는 메커니즘을 통해 작동합니다. 모델은 가중치를 업데이트하지 않습니다 — 대신 예시에서 패턴을 인식하고 새로운 입력에 적용합니다. 이것이 작동하는 이유:

1. **패턴 인식**: 트랜스포머는 강력한 패턴 매칭 도구입니다. 일관된 입력-출력 쌍이 주어지면 변환 규칙을 추론합니다.

2. **형식 시연**: 예시는 어떤 설명보다 더 정밀하게 정확한 출력 형식을 보여줍니다.

3. **엣지 케이스 커버리지**: 예시는 모호하거나 까다로운 경우를 처리하는 방법을 시연할 수 있습니다.

4. **레이블 그라운딩**: 예시는 추상적 레이블을 구체적 인스턴스에 그라운딩하여 오해를 줄입니다.

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

### 2.3 Messages API를 활용한 퓨샷

최신 API는 대화 구조를 사용합니다. 교차하는 사용자/어시스턴트 메시지로 퓨샷 예시를 제공할 수 있습니다:

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

### 2.4 원샷(One-Shot) vs 퓨샷(Few-Shot)

때때로 단일 예시로 충분합니다 — 이를 **원샷(One-shot)** 프롬프팅이라 합니다. 원샷을 사용하는 경우:
- 과제 패턴이 단순하고 일관적인 경우
- 토큰 예산이 제한적인 경우
- 시연해야 할 주요 대상이 형식인 경우

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

## 3. 예시 선택 전략

퓨샷 예시의 품질은 종종 양보다 더 중요합니다. 부적절한 예시는 제로샷에 비해 성능을 오히려 악화시킬 수 있습니다.

### 3.1 다양성 전략

과제 공간의 다양한 측면을 커버하는 예시를 선택합니다. 이는 모델이 좁은 패턴에 과적합하지 않고 일반화하는 데 도움이 됩니다.

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

### 3.2 유사성 전략

쿼리와 의미적으로 유사한 예시를 선택합니다. 유사한 입력은 유사한 처리를 필요로 하므로, 관련 예시가 가장 유용한 신호를 제공한다는 직관입니다.

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

### 3.3 대표성 전략

데이터의 전형적인 분포를 나타내는 예시를 선택합니다. 입력의 60%가 카테고리 A인 경우, 예시가 해당 비율을 반영하도록 합니다.

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

### 3.4 경계 예시 전략

결정 경계 근처에 위치한 예시 — 올바른 레이블이 놀랍거나 모호할 수 있는 경우를 포함합니다. 이러한 예시가 가장 정보적인 신호를 제공합니다.

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

여기서 경계 예시가 핵심입니다. 예시 3과 6은 링크와 긴급성 언어를 포함하지만 합법적입니다. 예시 5는 친근한 메시지로 위장한 소셜 엔지니어링 시도입니다. 이러한 경계 사례는 모델에 당신의 구체적인 분류 기준을 가르칩니다.

---

## 4. 예시 순서와 편향 효과

### 4.1 최신 편향(Recency Bias)

연구에 따르면 LLM은 퓨샷 예시의 순서, 특히 마지막 예시에 영향을 받을 수 있습니다. 이를 **최신 편향(Recency Bias)**이라 합니다 — 모델이 가장 최근 예시에 과도한 가중치를 부여하는 것입니다.

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

### 4.2 초두 편향(Primacy Bias)

퓨샷 프롬프트의 첫 번째 예시는 또한 모델의 과제 해석을 앵커링할 수 있습니다. 첫 번째 예시를 과제 패턴을 완벽하게 대표하는 "골드 스탠다드"로 만드세요.

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

### 4.3 순서 효과 완화

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

## 5. 의미 검색을 활용한 동적 퓨샷

정적 예시는 단순한 과제에 적합하지만, 동적 예시 선택 — 쿼리에 따라 즉석에서 예시를 선택하는 것은 다양한 입력에 대한 성능을 극적으로 향상시킵니다.

### 5.1 아키텍처

동적 퓨샷 파이프라인:
1. 사전 계산된 임베딩이 있는 **예시 저장소**를 유지합니다
2. 새로운 쿼리가 도착하면 임베딩을 계산합니다
3. 가장 유사한 K개의 예시를 검색합니다
4. 해당 예시로 프롬프트를 구성합니다
5. LLM에 전송합니다

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

### 5.2 예시 저장소 구축

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

### 5.3 대규모를 위한 벡터 데이터베이스 사용

수천 개의 예시가 있는 프로덕션 시스템의 경우, 인메모리 검색 대신 벡터 데이터베이스를 사용합니다:

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

## 6. K-Shot 선택 최적화

### 6.1 예시는 몇 개가 적당할까?

최적의 예시 수(K)는 여러 요인에 따라 달라집니다:

| 요인 | 적은 예시 (1-3) | 많은 예시 (5-10+) |
|------|-----------------|-------------------|
| 과제 복잡도 | 단순하고 잘 정의된 과제 | 복잡하고 미묘한 과제 |
| 레이블 집합 | 이진 (예/아니오) | 다중 클래스 (10개 이상 카테고리) |
| 모델 능력 | 대규모, 명령어 튜닝 모델 | 소규모 모델 |
| 토큰 예산 | 제한된 컨텍스트 윈도우 | 큰 컨텍스트 윈도우 |
| 출력 형식 | 단순 (레이블, 숫자) | 복잡 (JSON, 구조화) |

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

### 6.2 수확 체감(Diminishing Returns)

특정 K를 넘으면 추가 예시는 한계적 이점을 제공하지만 귀중한 토큰을 소비합니다. 일반적인 패턴:

- **K=0에서 K=1**: 가장 큰 도약 (형식을 시연)
- **K=1에서 K=3**: 상당한 개선 (패턴을 시연)
- **K=3에서 K=5**: 적당한 개선 (엣지 케이스를 커버)
- **K=5에서 K=10**: 작은 개선 (수확 체감)
- **K>10**: 종종 무시할 수 있는 개선, 토큰 낭비

```python
# Rule of thumb: Start with K=3, increase if accuracy is insufficient
# For classification: K=3-5 is usually optimal
# For generation: K=2-3 is usually sufficient
# For format specification: K=1 is often enough
```

### 6.3 부정 예시(Negative Examples)

모델이 특정 유형의 실수를 저지르는 경향이 있는 과제에 대해, 출력이 어떻게 되어서는 안 되는지를 보여주는 부정 예시를 포함하는 것이 매우 효과적일 수 있습니다.

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

## 7. 예시에서의 레이블 균형

### 7.1 균형 문제

퓨샷 예시가 불균형하면(예: 긍정 예시 4개, 부정 예시 1개), 모델은 다수 레이블 쪽으로 편향됩니다. 이것은 기저율 조작(Base-rate Manipulation)의 한 형태입니다.

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

### 7.2 불균형 실세계 분포 처리

프로덕션에서 실제 데이터는 불균형할 수 있습니다(예: 95% 비스팸, 5% 스팸). 퓨샷 예시는 현실 반영과 편향 방지 사이에서 균형을 맞춰야 합니다:

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

### 7.3 다중 클래스 균형

많은 클래스가 있는 과제의 경우, 동일한 대표 예시를 위한 토큰 예산이 부족할 수 있습니다. 우선순위:

1. 서로 쉽게 혼동되는 클래스
2. 모델이 존재를 잊을 수 있는 희귀 클래스
3. 경계가 명확하지 않은 클래스

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

## 8. 토큰 예산 관리

### 8.1 토큰 트레이드오프

예시에 사용된 모든 토큰은 응답이나 추가 맥락에 사용할 수 없는 토큰입니다. 특히 더 짧은 컨텍스트 윈도우에서 이 예산 관리가 중요합니다.

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

### 8.2 압축 전략

예시가 긴 경우, 필수 정보를 잃지 않으면서 압축합니다:

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

### 8.3 절삭 전략

원하는 모든 예시를 넣을 수 없는 경우, 지능적 절삭을 사용합니다:

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

## 9. 제로샷 vs 퓨샷: 과제 비교

### 9.1 의사결정 프레임워크

| 과제 유형 | 권장 | 근거 |
|-----------|------|------|
| 표준 분류 (감성, 주제) | 제로샷 | 훈련 데이터에 잘 표현됨 |
| 사용자 정의 분류 체계 분류 | 퓨샷 | 모델이 당신의 특정 레이블을 학습해야 함 |
| 데이터 추출 (표준 엔티티) | 제로샷 | NER이 잘 훈련됨 |
| 데이터 추출 (사용자 정의 형식) | 퓨샷 | 형식 시연이 필요함 |
| 번역 | 제로샷 | 잘 훈련됨, 예시가 거의 도움이 안 됨 |
| 코드 생성 | 제로샷 | 강력한 명령어 따르기 |
| 텍스트 요약 | 제로샷 | 잘 훈련됨, 하지만 퓨샷이 스타일에 도움 |
| 스타일 전환 | 퓨샷 | 스타일은 설명하기 어렵지만 시연하기 쉬움 |
| 복잡한 구조화 출력 | 퓨샷 | JSON 스키마는 구체적 예시가 필요 |
| 추론 과제 | 제로샷 (CoT 포함) | 추론 예시보다 지시가 더 도움 |

### 9.2 실험적 비교

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

### 9.3 하이브리드 접근법

최선의 접근법은 종종 단순한 경우에 제로샷을 사용하고 더 어려운 경우에 동적으로 예시를 추가하는 하이브리드입니다:

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

## 연습문제

### 연습문제 1: 제로샷 설계

고객 이메일을 정확히 7개 카테고리 중 하나로 분류하는 제로샷 프롬프트를 설계하세요: `billing`, `technical`, `shipping`, `account`, `feedback`, `partnership`, `other`. 프롬프트는 예시 없이 안정적으로 작동해야 합니다. 모호한 경우에 대한 명시적 처리를 포함하세요.

<details><summary>정답 보기</summary>

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

핵심 기법:
- 각 카테고리에 명확한 정의가 있습니다 (모호성 감소)
- 다중 카테고리 이메일에 대한 명시적 규칙 (주요 의도)
- "other"는 사용 규칙이 있는 범용입니다 (과도 사용 방지)
- 출력 형식이 단일 단어로 제약됩니다
- 이메일이 XML 태그로 구분됩니다

</details>

### 연습문제 2: 퓨샷 예시 선택

독성 분류기를 위한 다음 10개의 레이블된 예시 풀에서 퓨샷 프롬프트에 최적인 5개 예시를 선택하세요. 선택 기준을 설명하세요.

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

<details><summary>정답 보기</summary>

**선택된 예시 (근거 포함):**

1. **예시 5**: "I disagree with your point about tax policy" -> not_toxic
   *경계 사례*: 반대 의견은 독성이 아닙니다. 모델이 반대 의견을 독성으로 잘못 표시하는 경우가 많기 때문에 이것이 가장 중요한 예시입니다.

2. **예시 6**: "Your argument is trash, just like your face" -> toxic
   *경계 사례*: 주장에 대한 의견으로 시작하지만(not_toxic일 수 있음) 인신공격으로 확대됩니다. 경계선이 어디인지 모델에게 가르칩니다.

3. **예시 7**: "Maybe try reading a book sometime, just saying" -> not_toxic
   *경계 사례*: 이것은 수동적-공격적이고 거만하지만 대부분의 가이드라인에서 독성이 아닙니다. 모델의 임계값을 교정하는 데 중요합니다.

4. **예시 9**: "People like you are what's wrong with this world" -> toxic
   *경계 사례*: 욕설이 없지만 비인간화합니다. 독성이 단지 나쁜 단어에 관한 것이 아님을 가르칩니다.

5. **예시 2**: "You're an absolute idiot and should be banned" -> toxic
   *명확한 긍정 예시*: 명확한 독성 의도가 있는 직접적 모욕. "명백히 독성인" 스펙트럼의 끝을 앵커링합니다.

**왜 이 5개인가?**

- **레이블 균형**: 2개 not_toxic, 3개 toxic (독성 쪽으로 약간 불균형 — 모델의 기본값이 비독성으로 분류하는 경향이 있어 독성에 민감하도록 하고 싶기 때문)
- **경계 사례**: 5개 예시 중 4개가 결정 경계 근처에 있어 모델에 가장 많은 안내가 필요한 곳
- **독성 패턴의 다양성**: 인신 모욕(#2), 의견과 혼합된 인신 공격(#6), 비인간화(#9)
- **제외됨**: 예시 1, 3, 8 (너무 명백 — 학습 신호 없음), 예시 4 (극단적, 전형적인 경계 사례를 대표하지 않음)

</details>

### 연습문제 3: 동적 퓨샷 구현

코사인 유사도를 사용하여 동적 퓨샷 예시 선택을 구현하는 함수를 작성하세요. 함수는 쿼리 문자열, 예시 풀(`text`, `label`, `embedding` 필드가 있는 딕셔너리 리스트)을 받고, 레이블 균형을 보장하면서 상위 K개의 가장 유사한 예시로 포맷된 퓨샷 프롬프트를 반환해야 합니다.

<details><summary>정답 보기</summary>

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

핵심 설계 결정:
1. **유사성 우선 선택**이 관련성을 보장합니다
2. **레이블 균형**이 다수 클래스 편향을 방지합니다
3. **레이블별 교차 배치**가 최신 편향을 줄입니다
4. **폴백 채우기**가 항상 K개 예시에 도달하도록 보장합니다
5. **구성 가능한 max_per_label**이 균형 조정을 허용합니다

</details>

### 연습문제 4: 토큰 예산 최적화

4096 토큰 컨텍스트 윈도우(소형 모델)가 있습니다. 시스템 프롬프트가 200 토큰을 사용하고, 응답에 512 토큰이 필요하며, 쿼리 평균은 50 토큰입니다. 각 퓨샷 예시는 평균 80 토큰입니다. 계산하세요: (a) 넣을 수 있는 최대 예시 수, (b) 예시를 50 토큰으로 압축하면 몇 개를 더 넣을 수 있는지, (c) 쿼리 길이에 따라 동적으로 K를 결정하는 함수를 작성하세요.

<details><summary>정답 보기</summary>

**(a) 80 토큰 예시에서 최대 예시 수:**
```
Available = 4096 - 200 (system) - 512 (response) - 50 (query) = 3334 tokens
Max examples = 3334 // 80 = 41 examples
```

**(b) 50 토큰으로 압축된 예시:**
```
Max examples = 3334 // 50 = 66 examples
Additional examples = 66 - 41 = 25 more examples
```

**(c) 동적 K 함수:**

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

### 연습문제 5: 비교 분석

5개 카테고리의 이메일 의도 분류 과제에서 제로샷, 3-샷, 5-샷 프롬프팅을 비교하는 실험을 설계하세요. API 호출을 목킹하여 정확도, 클래스별 정밀도를 계산하고 결과 요약 표를 생성하는 완전한 평가 코드를 작성하세요.

<details><summary>정답 보기</summary>

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

이 유형의 실험에서 얻는 핵심 통찰:
- 제로샷 기준선은 모델이 이미 잘 수행하는 곳을 보여줍니다
- 0-샷에서 3-샷으로의 점프가 일반적으로 가장 큰 개선입니다
- 클래스별 정밀도는 어떤 카테고리가 예시로부터 가장 많이 혜택을 받는지 보여줍니다
- 분류 과제의 경우 일반적으로 K=5 이후에 수확 체감이 나타납니다

</details>

---

**이전**: [프롬프트 기초](./01_Prompt_Fundamentals.md) | **다음**: [사고의 연쇄](./03_Chain_of_Thought.md)

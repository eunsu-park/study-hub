# 24. API & Evaluation

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare commercial LLM API providers (OpenAI, Anthropic, Google) across pricing, context length, and capability dimensions
2. Implement robust API client wrappers that handle rate limiting, retries, token counting, and cost tracking
3. Apply prompt engineering techniques and caching strategies to optimize API costs in production applications
4. Evaluate LLM performance using standard benchmarks (MMLU, HellaSwag, HumanEval) and explain what each benchmark measures
5. Design a comprehensive LLM evaluation pipeline that combines automated benchmarks with LLM-as-a-judge and human evaluation

---

## Overview

This lesson covers how to use commercial LLM APIs, cost optimization, and benchmarks and methodologies for evaluating LLM performance.

---

## 1. Commercial LLM APIs

### 1.1 Major Provider Comparison

```
API Provider Comparison (2024):
┌────────────────────────────────────────────────────────────────┐
│  Provider    │ Model          │ Input/1M  │ Output/1M │ Context │
├──────────────┼────────────────┼───────────┼───────────┼─────────┤
│  OpenAI      │ GPT-4 Turbo    │ $10       │ $30       │ 128K    │
│              │ GPT-4o         │ $5        │ $15       │ 128K    │
│              │ GPT-3.5 Turbo  │ $0.50     │ $1.50     │ 16K     │
├──────────────┼────────────────┼───────────┼───────────┼─────────┤
│  Anthropic   │ Claude 3 Opus  │ $15       │ $75       │ 200K    │
│              │ Claude 3 Sonnet│ $3        │ $15       │ 200K    │
│              │ Claude 3 Haiku │ $0.25     │ $1.25     │ 200K    │
├──────────────┼────────────────┼───────────┼───────────┼─────────┤
│  Google      │ Gemini 1.5 Pro │ $3.50     │ $10.50    │ 1M      │
│              │ Gemini 1.5 Flash│ $0.35    │ $1.05     │ 1M      │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 OpenAI API

```python
from openai import OpenAI
import tiktoken

class OpenAIClient:
    """OpenAI API client"""

    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key)
        self.token_encoder = tiktoken.get_encoding("cl100k_base")

    def chat(
        self,
        messages: list,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> dict:
        """Chat completion"""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        return {
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason
        }

    def stream_chat(self, messages: list, model: str = "gpt-4o", **kwargs):
        """Streaming chat"""
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **kwargs
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def count_tokens(self, text: str) -> int:
        """Count tokens"""
        return len(self.token_encoder.encode(text))

    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "gpt-4o"
    ) -> float:
        """Estimate cost"""
        pricing = {
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5}
        }

        if model not in pricing:
            return 0.0

        cost = (
            prompt_tokens * pricing[model]["input"] / 1_000_000 +
            completion_tokens * pricing[model]["output"] / 1_000_000
        )

        return cost


# Function calling
def function_calling_example():
    client = OpenAI()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What's the weather in Seoul?"}],
        tools=tools,
        tool_choice="auto"
    )

    # Handle tool call
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        print(f"Function: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
```

### 1.3 Anthropic API

```python
from anthropic import Anthropic

class AnthropicClient:
    """Anthropic Claude API client"""

    def __init__(self, api_key: str = None):
        self.client = Anthropic(api_key=api_key)

    def chat(
        self,
        messages: list,
        model: str = "claude-3-sonnet-20240229",
        max_tokens: int = 1000,
        system: str = None,
        **kwargs
    ) -> dict:
        """Chat"""
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or "You are a helpful assistant.",
            messages=messages,
            **kwargs
        )

        return {
            "content": response.content[0].text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            },
            "model": response.model,
            "stop_reason": response.stop_reason
        }

    def stream_chat(self, messages: list, **kwargs):
        """Streaming"""
        with self.client.messages.stream(
            messages=messages,
            **kwargs
        ) as stream:
            for text in stream.text_stream:
                yield text

    def vision(
        self,
        image_url: str,
        prompt: str,
        model: str = "claude-3-sonnet-20240229"
    ) -> str:
        """Vision API"""
        import base64
        import httpx

        # Load image
        if image_url.startswith("http"):
            image_data = base64.standard_b64encode(
                httpx.get(image_url).content
            ).decode("utf-8")
        else:
            with open(image_url, "rb") as f:
                image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        response = self.client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )

        return response.content[0].text
```

### 1.4 Google Gemini API

```python
import google.generativeai as genai

class GeminiClient:
    """Google Gemini API client"""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')

    def chat(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> dict:
        """Chat"""
        # Convert OpenAI format to Gemini format
        history = []
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})

        chat = self.model.start_chat(history=history)
        response = chat.send_message(
            messages[-1]["content"],
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )

        return {
            "content": response.text,
            "usage": {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count
            }
        }

    def multimodal(
        self,
        image_path: str,
        prompt: str
    ) -> str:
        """Multimodal input"""
        import PIL.Image

        img = PIL.Image.open(image_path)
        response = self.model.generate_content([prompt, img])

        return response.text
```

---

## 2. Cost Optimization

### 2.1 Cost Monitoring

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List
import json

@dataclass
class UsageRecord:
    """API usage record"""
    timestamp: datetime
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    request_type: str = "chat"

class CostTracker:
    """Cost tracker"""

    def __init__(self):
        self.records: List[UsageRecord] = []
        self.pricing = {
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "claude-3-opus": {"input": 15.0, "output": 75.0},
            "claude-3-sonnet": {"input": 3.0, "output": 15.0},
            "claude-3-haiku": {"input": 0.25, "output": 1.25}
        }

    def log_request(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        request_type: str = "chat"
    ):
        """Log request"""
        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)

        record = UsageRecord(
            timestamp=datetime.now(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            request_type=request_type
        )

        self.records.append(record)
        return cost

    def _calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """Calculate cost"""
        if model not in self.pricing:
            return 0.0

        pricing = self.pricing[model]
        return (
            prompt_tokens * pricing["input"] / 1_000_000 +
            completion_tokens * pricing["output"] / 1_000_000
        )

    def get_summary(self, period: str = "day") -> Dict:
        """Usage summary"""
        from collections import defaultdict

        summary = defaultdict(lambda: {"tokens": 0, "cost": 0, "requests": 0})

        for record in self.records:
            model = record.model
            summary[model]["tokens"] += record.prompt_tokens + record.completion_tokens
            summary[model]["cost"] += record.cost
            summary[model]["requests"] += 1

        return dict(summary)

    def set_budget_alert(self, daily_limit: float):
        """Set daily budget alert"""
        today_cost = sum(
            r.cost for r in self.records
            if r.timestamp.date() == datetime.now().date()
        )

        if today_cost > daily_limit:
            return f"Warning: Daily budget exceeded: ${today_cost:.2f} / ${daily_limit:.2f}"

        return None
```

### 2.2 Optimization Strategies

```python
class CostOptimizer:
    """Cost optimization strategies"""

    def __init__(self):
        self.cache = {}

    def semantic_cache(self, query: str, threshold: float = 0.95):
        """Semantic caching"""
        # Find similar previous query
        from sentence_transformers import SentenceTransformer
        import numpy as np

        if not hasattr(self, 'encoder'):
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        query_emb = self.encoder.encode(query)

        for cached_query, (cached_emb, response) in self.cache.items():
            similarity = np.dot(query_emb, cached_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(cached_emb)
            )
            if similarity > threshold:
                return response

        return None

    def add_to_cache(self, query: str, response: str):
        """Add to cache"""
        if hasattr(self, 'encoder'):
            emb = self.encoder.encode(query)
            self.cache[query] = (emb, response)

    def select_model(
        self,
        task_complexity: str,
        latency_requirement: str = "normal"
    ) -> str:
        """Select model appropriate for task"""
        model_map = {
            # (complexity, latency) -> model
            ("simple", "fast"): "gpt-3.5-turbo",
            ("simple", "normal"): "gpt-3.5-turbo",
            ("medium", "fast"): "claude-3-haiku",
            ("medium", "normal"): "gpt-4o",
            ("complex", "fast"): "gpt-4o",
            ("complex", "normal"): "claude-3-opus",
        }

        return model_map.get(
            (task_complexity, latency_requirement),
            "gpt-4o"
        )

    def prompt_compression(self, text: str, target_ratio: float = 0.5) -> str:
        """Prompt compression"""
        # Can use LLMLingua etc.
        # Here using simple summarization approach
        words = text.split()
        target_len = int(len(words) * target_ratio)

        # Select important sentences (needs more sophisticated method in practice)
        sentences = text.split('.')
        if len(sentences) <= 2:
            return text

        # Keep first and last sentences
        compressed = sentences[0] + '.' + sentences[-1]
        return compressed
```

---

## 3. LLM Evaluation

### 3.1 Benchmarks

```
Major Benchmarks:
┌────────────────────────────────────────────────────────────────┐
│  General                                                        │
│  - MMLU: 57 subjects, multiple choice                          │
│  - HellaSwag: Commonsense reasoning                            │
│  - WinoGrande: Coreference resolution                          │
│                                                                │
│  Reasoning                                                      │
│  - GSM8K: Grade school math                                    │
│  - MATH: Competition math                                       │
│  - ARC: Science questions                                       │
│                                                                │
│  Coding                                                         │
│  - HumanEval: Python code generation                           │
│  - MBPP: Python problems                                       │
│  - CodeContests: Competitive programming                       │
│                                                                │
│  Chat/Instruction                                               │
│  - MT-Bench: Multi-turn conversation                           │
│  - AlpacaEval: Instruction following                           │
│  - Chatbot Arena: Human preference                             │
└────────────────────────────────────────────────────────────────┘
```

### 3.2 Automated Evaluation

```python
import re
from typing import List, Dict

class LLMEvaluator:
    """LLM automated evaluation"""

    def __init__(self, model_client):
        self.client = model_client

    def evaluate_factuality(
        self,
        question: str,
        answer: str,
        reference: str
    ) -> Dict:
        """Factuality evaluation"""
        prompt = f"""Evaluate if the answer is factually consistent with the reference.

Question: {question}
Answer: {answer}
Reference: {reference}

Score from 1-5 where:
1 = Completely incorrect
3 = Partially correct
5 = Completely correct

Provide your score and brief explanation.
Format: Score: X
Explanation: ..."""

        response = self.client.chat([{"role": "user", "content": prompt}])
        text = response["content"]

        # Extract score
        score_match = re.search(r'Score:\s*(\d)', text)
        score = int(score_match.group(1)) if score_match else 3

        return {
            "score": score,
            "explanation": text
        }

    def evaluate_helpfulness(
        self,
        instruction: str,
        response: str
    ) -> Dict:
        """Helpfulness evaluation"""
        prompt = f"""Evaluate how helpful and complete the response is.

Instruction: {instruction}
Response: {response}

Rate on these criteria (1-5 each):
1. Relevance: Does it address the instruction?
2. Completeness: Does it fully answer?
3. Clarity: Is it well-written and clear?
4. Accuracy: Is the information correct?

Format:
Relevance: X
Completeness: X
Clarity: X
Accuracy: X
Overall: X"""

        response = self.client.chat([{"role": "user", "content": prompt}])
        text = response["content"]

        # Parse scores
        scores = {}
        for criterion in ["Relevance", "Completeness", "Clarity", "Accuracy", "Overall"]:
            match = re.search(rf'{criterion}:\s*(\d)', text)
            scores[criterion.lower()] = int(match.group(1)) if match else 3

        return scores

    def pairwise_comparison(
        self,
        instruction: str,
        response_a: str,
        response_b: str
    ) -> str:
        """Pairwise comparison"""
        prompt = f"""Compare these two responses to the instruction.

Instruction: {instruction}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Consider helpfulness, accuracy, and clarity.
Answer with:
- "A" if Response A is better
- "B" if Response B is better
- "TIE" if they are equally good

Your choice:"""

        response = self.client.chat([{"role": "user", "content": prompt}])
        text = response["content"].strip().upper()

        if "A" in text and "B" not in text:
            return "A"
        elif "B" in text and "A" not in text:
            return "B"
        else:
            return "TIE"


# MT-Bench style evaluation
class MTBenchEvaluator:
    """MT-Bench style multi-turn evaluation"""

    def __init__(self, judge_model):
        self.judge = judge_model

    def evaluate_conversation(
        self,
        conversation: List[Dict]
    ) -> Dict:
        """Evaluate conversation"""
        # Evaluate each turn
        turn_scores = []

        for i, turn in enumerate(conversation):
            if turn["role"] == "assistant":
                context = conversation[:i+1]
                score = self._evaluate_turn(context)
                turn_scores.append(score)

        return {
            "turn_scores": turn_scores,
            "average": sum(turn_scores) / len(turn_scores) if turn_scores else 0
        }

    def _evaluate_turn(self, context: List[Dict]) -> float:
        """Evaluate individual turn"""
        # Compose evaluation prompt
        context_str = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in context
        ])

        prompt = f"""Rate the assistant's last response on a scale of 1-10.

Conversation:
{context_str}

Consider:
- Helpfulness
- Relevance
- Accuracy
- Depth

Score (1-10):"""

        response = self.judge.chat([{"role": "user", "content": prompt}])
        score_match = re.search(r'\d+', response["content"])

        return float(score_match.group()) if score_match else 5.0
```

### 3.3 Human Evaluation

```python
from dataclasses import dataclass
from typing import Optional
import random

@dataclass
class EvaluationItem:
    """Evaluation item"""
    id: str
    instruction: str
    response_a: str
    response_b: str
    model_a: str
    model_b: str
    winner: Optional[str] = None
    annotator: Optional[str] = None

class HumanEvaluation:
    """Human evaluation management"""

    def __init__(self):
        self.items: List[EvaluationItem] = []
        self.results: Dict[str, int] = {}

    def add_comparison(
        self,
        instruction: str,
        responses: Dict[str, str]  # {model_name: response}
    ):
        """Add comparison item"""
        models = list(responses.keys())
        if len(models) != 2:
            raise ValueError("Exactly 2 models required")

        # Randomize order (prevent bias)
        if random.random() > 0.5:
            models = models[::-1]

        item = EvaluationItem(
            id=str(len(self.items)),
            instruction=instruction,
            response_a=responses[models[0]],
            response_b=responses[models[1]],
            model_a=models[0],
            model_b=models[1]
        )

        self.items.append(item)

    def record_judgment(
        self,
        item_id: str,
        winner: str,  # "A", "B", or "TIE"
        annotator: str
    ):
        """Record evaluation result"""
        for item in self.items:
            if item.id == item_id:
                item.winner = winner
                item.annotator = annotator

                # Record winning model
                if winner == "A":
                    winning_model = item.model_a
                elif winner == "B":
                    winning_model = item.model_b
                else:
                    winning_model = "TIE"

                self.results[winning_model] = self.results.get(winning_model, 0) + 1
                break

    def get_elo_ratings(self) -> Dict[str, float]:
        """Calculate Elo ratings"""
        # Initial ratings
        ratings = {}
        for item in self.items:
            ratings[item.model_a] = 1500
            ratings[item.model_b] = 1500

        K = 32  # K-factor

        for item in self.items:
            if item.winner is None:
                continue

            ra = ratings[item.model_a]
            rb = ratings[item.model_b]

            # Expected scores
            ea = 1 / (1 + 10 ** ((rb - ra) / 400))
            eb = 1 / (1 + 10 ** ((ra - rb) / 400))

            # Actual scores
            if item.winner == "A":
                sa, sb = 1, 0
            elif item.winner == "B":
                sa, sb = 0, 1
            else:
                sa, sb = 0.5, 0.5

            # Update ratings
            ratings[item.model_a] += K * (sa - ea)
            ratings[item.model_b] += K * (sb - eb)

        return ratings
```

---

## Key Summary

### API Usage Checklist
```
□ Manage API keys as environment variables
□ Pre-calculate token counts
□ Set up cost monitoring
□ Handle rate limits
□ Error handling and retries
□ Implement caching strategy
```

### Evaluation Method Selection
```
- Multiple choice questions → Accuracy
- Generation tasks → LLM-as-Judge
- Chat/conversation → MT-Bench/Chatbot Arena
- Coding → pass@k, HumanEval
- Production → A/B testing
```

---

## References

1. [OpenAI API Documentation](https://platform.openai.com/docs)
2. [Anthropic Claude Documentation](https://docs.anthropic.com/)
3. Zheng et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"

---

## Exercises

### Exercise 1: API Cost Calculation and Model Selection
Your team builds a document summarization service. Each request involves a 2,000-token document (input) and produces a 300-token summary (output). You need to handle 10,000 requests per day.

A) Calculate the daily cost for: GPT-4o ($5/1M input, $15/1M output) vs. Claude 3 Haiku ($0.25/1M input, $1.25/1M output).
B) At what daily request volume does the cost difference become significant enough to justify switching from GPT-4o to Haiku (>$100/day savings)?
C) If 20% of requests are "complex" (require reasoning, high quality), propose a routing strategy.

<details>
<summary>Show Answer</summary>

**A) Daily cost at 10,000 requests**:

```python
requests_per_day = 10_000
input_tokens_per_req = 2_000
output_tokens_per_req = 300

# GPT-4o: $5/1M input, $15/1M output
gpt4o_input_cost = (requests_per_day * input_tokens_per_req / 1_000_000) * 5
                 = (10_000 * 2_000 / 1_000_000) * 5 = 20M tokens * $5/M = $100/day

gpt4o_output_cost = (requests_per_day * output_tokens_per_req / 1_000_000) * 15
                  = (10_000 * 300 / 1_000_000) * 15 = 3M tokens * $15/M = $45/day

gpt4o_total = $100 + $45 = $145/day

# Claude 3 Haiku: $0.25/1M input, $1.25/1M output
haiku_input_cost = 20M tokens * $0.25/M = $5/day
haiku_output_cost = 3M tokens * $1.25/M = $3.75/day
haiku_total = $5 + $3.75 = $8.75/day

# Daily savings with Haiku: $145 - $8.75 = $136.25/day
```

**B) Volume threshold for >$100/day savings**:

```python
# Cost per request:
gpt4o_per_req = (2000/1M * $5) + (300/1M * $15) = $0.01 + $0.0045 = $0.0145
haiku_per_req = (2000/1M * $0.25) + (300/1M * $1.25) = $0.0005 + $0.000375 = $0.000875

savings_per_req = $0.0145 - $0.000875 = $0.013625

# To save >$100/day:
min_requests = $100 / $0.013625 ≈ 7,339 requests/day
```
**Answer: At 7,339+ requests/day, switching saves >$100/day.** At 10K requests/day you save $136/day = ~$4,100/month.

**C) Routing strategy for mixed complexity**:

```python
def route_request(document: str, task_complexity: str) -> str:
    """Route to appropriate model based on complexity"""
    if task_complexity == "complex":  # 20% of requests
        # Legal contracts, technical papers, multi-document synthesis
        return "claude-3-sonnet"  # Balance of quality and cost
    else:  # 80% of requests
        # News articles, product descriptions, standard reports
        return "claude-3-haiku"

# Cost with routing (10K requests/day):
# 2,000 complex → Sonnet ($3/1M in, $15/1M out)
#   2,000 * (2000 * $3/M + 300 * $15/M) = $12 + $9 = $21/day
# 8,000 simple → Haiku
#   8,000 * (2000 * $0.25/M + 300 * $1.25/M) = $4 + $3 = $7/day
# Total with routing: $28/day vs. $145/day all-GPT4o → 80% cost reduction
```

</details>

### Exercise 2: LLM-as-a-Judge Bias Analysis
LLM-as-a-Judge uses a powerful LLM to score model outputs. Identify three specific biases that can distort LLM judge evaluations, and propose a mitigation strategy for each.

<details>
<summary>Show Answer</summary>

**Bias 1: Position bias (order bias)**

The LLM judge tends to favor the response that appears first (or last) in the prompt, regardless of quality.

- **Example**: When comparing Response A vs. Response B, the same judge scores A=7, B=5 in one order, but A=5, B=7 when the order is reversed.
- **Mitigation**: Always evaluate both orderings (A vs. B AND B vs. A) and average the results. For pairwise comparison, if the judge is inconsistent across orderings, treat it as a tie. This is called **positional debiasing** and is used in Chatbot Arena.

**Bias 2: Verbosity bias (length preference)**

LLM judges tend to rate longer, more detailed responses as higher quality, even when a concise answer is more appropriate.

- **Example**: A 500-word answer is scored higher than a 50-word answer even when the 50-word answer is more accurate and directly answers the question.
- **Mitigation**: Add explicit scoring criteria that penalize unnecessary length: "A high-quality answer is one that accurately answers the question concisely. Longer answers are not automatically better." Also provide a reference answer length as a calibration anchor.

**Bias 3: Self-enhancement bias (if judge = same family as evaluated model)**

If GPT-4 is used to judge GPT-4 vs. Claude-3, it may systematically favor its own model's style and content.

- **Example**: GPT-4 judge scores GPT-4o answers +0.3 points higher than equally good Claude responses on average.
- **Mitigation**: Use multiple judges from different model families and ensemble their scores. For critical evaluations, include at least one judge from a different provider than the evaluated model. Report judge-model agreement (if judges from different families agree, the result is more trustworthy).

</details>

### Exercise 3: Benchmark Selection for Specific Use Cases
You are evaluating an LLM for three different product deployments. Choose the most relevant benchmark from the list {MMLU, HumanEval, MT-Bench, GSM8K, HellaSwag, MATH} for each, and explain what a high score on that benchmark guarantees — and what it does NOT guarantee.

| Deployment | Primary Use | Best Benchmark | Guarantees | Does NOT guarantee |
|------------|------------|----------------|------------|-------------------|
| A) AI coding assistant for Python developers | Code completion, debugging | ??? | ??? | ??? |
| B) Customer service chatbot for multi-turn support | Dialogue, policy adherence | ??? | ??? | ??? |
| C) Financial analysis assistant for investment reports | Quantitative reasoning | ??? | ??? | ??? |

<details>
<summary>Show Answer</summary>

| Deployment | Best Benchmark | Guarantees | Does NOT guarantee |
|------------|----------------|------------|-------------------|
| A) AI coding assistant | **HumanEval** (Python code generation, pass@k) | Ability to generate syntactically correct, functionally complete Python functions for common algorithmic tasks | Performance on domain-specific code (finance, medical APIs), ability to debug existing code, code review quality, performance on long (100+ line) functions |
| B) Customer service chatbot | **MT-Bench** (multi-turn conversation quality, 1-10 scoring) | Quality of multi-turn dialogue coherence, ability to maintain context across conversation turns, handling of follow-up questions | Adherence to specific company policies or prohibited topics, actual customer satisfaction, handling of ambiguous real-world requests, performance on domain-specific vocabulary |
| C) Financial analysis assistant | **GSM8K + MATH** (mathematical reasoning at different difficulty levels) | Accurate arithmetic, algebraic manipulation, structured mathematical reasoning, solving word problems with quantitative content | Domain-specific financial knowledge (CAPM, Black-Scholes), regulatory compliance awareness, ability to handle real financial data formats, detecting data inconsistencies in reports |

**Key insight**: Every benchmark evaluates a proxy for real-world capability. A model that scores 90% on HumanEval may still fail on company-specific codebases. Always supplement benchmarks with domain-specific evaluation sets representative of your actual deployment.

</details>

### Exercise 4: Evaluation Pipeline Design
Design a production evaluation pipeline for a customer-facing legal document Q&A system. The system retrieves relevant clauses and generates answers. Specify the evaluation dimensions, metrics, and scoring thresholds.

<details>
<summary>Show Answer</summary>

**Production Evaluation Pipeline**:

```
Evaluation Pipeline:
┌──────────────────────────────────────────────────┐
│  Input: (question, retrieved_clauses, answer)    │
│                                                  │
│  Dimension 1: Factual Accuracy                   │
│  Dimension 2: Legal Citation Correctness         │
│  Dimension 3: Completeness                       │
│  Dimension 4: Safety (no harmful advice)         │
│  Dimension 5: Fluency                            │
└──────────────────────────────────────────────────┘
```

**Evaluation dimensions and metrics**:

| Dimension | Method | Metric | Pass Threshold |
|-----------|--------|--------|----------------|
| Factual accuracy | LLM-as-Judge with reference answers | 1-5 score | ≥ 4.0 average |
| Citation correctness | Rule-based: check if cited clause # appears in retrieved docs | Precision/Recall | Citation precision ≥ 90% |
| Completeness | LLM-as-Judge: "Does the answer address all parts of the question?" | Binary + score | ≥ 80% rated "complete" |
| Safety | Keyword filters + LLM classifier for "definitive legal advice" | Binary flag | 0% rate of definitive advice without disclaimer |
| Fluency | Automated: perplexity + grammar check | Score 1-5 | ≥ 4.0 |

**Offline vs. Online evaluation**:

```python
# Offline (development):
# - Golden test set of 200 legal Q&A pairs (human-curated)
# - Run after every model update
# - Gate: all 5 dimensions must pass thresholds

# Online (production):
# - Sample 1% of live queries for human review
# - Track:
#   - User satisfaction (thumbs up/down)
#   - Escalation rate (how often user asks follow-up because answer was unclear)
#   - Refusal rate (model adds too many disclaimers → user can't get answer)
# - Alert if accuracy drops > 5% below baseline
```

**Critical safety rule for legal domain**: Any answer that could be construed as specific legal advice must include a disclaimer: "This is not legal advice. Consult a licensed attorney for your specific situation." The evaluation pipeline must flag and reject any answer that makes a definitive legal conclusion without this caveat.

</details>
4. Chen et al. (2021). "Evaluating Large Language Models Trained on Code" (HumanEval)

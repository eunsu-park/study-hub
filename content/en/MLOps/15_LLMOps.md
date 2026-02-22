# LLMOps

## Overview

LLMOps extends MLOps practices for Large Language Model applications. Unlike traditional ML models trained on structured data, LLMs require distinct operational patterns: prompt engineering and versioning, retrieval-augmented generation (RAG) pipelines, evaluation without ground truth, guardrails, and cost management. This lesson covers the operational lifecycle of LLM-powered applications.

---

## 1. LLMOps vs Traditional MLOps

### 1.1 Key Differences

```python
"""
Traditional MLOps vs LLMOps:

| Aspect              | Traditional MLOps          | LLMOps                        |
|---------------------|----------------------------|--------------------------------|
| Model               | Train from scratch         | Use pretrained / fine-tune     |
| Input               | Structured features        | Natural language prompts       |
| Configuration       | Hyperparameters            | Prompts + system instructions  |
| Evaluation          | Accuracy, F1, AUC          | LLM-as-judge, human eval      |
| Versioning          | Model weights + data       | Prompts + RAG configs + model  |
| Cost                | Training compute           | Inference tokens (pay-per-use) |
| Latency             | ms (batch inference)       | Seconds (autoregressive)       |
| Failure modes       | Wrong prediction           | Hallucination, toxicity, PII   |
| Monitoring          | Data/concept drift         | Prompt injection, cost drift   |

LLMOps Pipeline:
  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
  │  Prompt   │──▶│  RAG /   │──▶│  LLM     │──▶│  Guard-  │──▶ Response
  │  Template │   │  Context │   │  Call     │   │  rails   │
  └──────────┘   └──────────┘   └──────────┘   └──────────┘
       ↑              ↑              ↑              ↑
  Versioned      Vector DB      Model Config    Safety Rules
"""
```

---

## 2. Prompt Management

### 2.1 Prompt Registry

```python
"""
Prompt Registry: Version-controlled prompt templates.

Why not hardcode prompts?
  - Prompts change frequently (tuning, A/B tests)
  - Need rollback capability
  - Multiple environments (dev/staging/prod)
  - Audit trail for compliance

Architecture:
  ┌──────────────┐     ┌──────────────┐
  │  Prompt       │     │  Application  │
  │  Registry     │────▶│  Server       │
  │  (DB/Git/API) │     │              │
  └──────────────┘     └──────────────┘
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path


class PromptRegistry:
    """File-based prompt registry with versioning."""

    def __init__(self, registry_dir="prompts"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.registry_dir / "index.json"
        self._index = self._load_index()

    def _load_index(self):
        if self._index_path.exists():
            return json.loads(self._index_path.read_text())
        return {}

    def _save_index(self):
        self._index_path.write_text(json.dumps(self._index, indent=2))

    def register(self, name, template, metadata=None):
        """Register a new prompt version."""
        content_hash = hashlib.sha256(template.encode()).hexdigest()[:12]
        version = len(self._index.get(name, {}).get("versions", [])) + 1

        entry = {
            "version": version,
            "template": template,
            "hash": content_hash,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        # Save template file
        prompt_dir = self.registry_dir / name
        prompt_dir.mkdir(exist_ok=True)
        (prompt_dir / f"v{version}.txt").write_text(template)

        # Update index
        if name not in self._index:
            self._index[name] = {"active_version": 1, "versions": []}
        self._index[name]["versions"].append(entry)
        self._index[name]["active_version"] = version
        self._save_index()

        print(f"Registered {name} v{version} (hash={content_hash})")
        return version

    def get(self, name, version=None):
        """Get a prompt template (default: active version)."""
        if name not in self._index:
            raise KeyError(f"Prompt '{name}' not found")

        if version is None:
            version = self._index[name]["active_version"]

        versions = self._index[name]["versions"]
        entry = versions[version - 1]
        return entry["template"]

    def rollback(self, name, version):
        """Set active version to a previous version."""
        if version < 1 or version > len(self._index[name]["versions"]):
            raise ValueError(f"Invalid version: {version}")
        self._index[name]["active_version"] = version
        self._save_index()
        print(f"Rolled back {name} to v{version}")

    def list_versions(self, name):
        """List all versions of a prompt."""
        if name not in self._index:
            return []
        active = self._index[name]["active_version"]
        for v in self._index[name]["versions"]:
            marker = " ← active" if v["version"] == active else ""
            print(f"  v{v['version']} ({v['created_at']}) hash={v['hash']}{marker}")
        return self._index[name]["versions"]


# Usage
registry = PromptRegistry()
registry.register(
    "summarizer",
    template="Summarize the following text in {num_sentences} sentences:\n\n{text}",
    metadata={"model": "claude-sonnet-4-20250514", "author": "team-a"},
)
registry.register(
    "summarizer",
    template="You are a concise summarizer. Given the text below, "
             "produce exactly {num_sentences} sentences:\n\n{text}",
    metadata={"model": "claude-sonnet-4-20250514", "author": "team-a", "note": "improved clarity"},
)
registry.list_versions("summarizer")
```

### 2.2 Prompt Testing

```python
"""Test prompts systematically before deployment."""

import json


class PromptTestSuite:
    """Run test cases against a prompt template."""

    def __init__(self, prompt_template, model_fn):
        self.template = prompt_template
        self.model_fn = model_fn  # function(prompt_str) → response_str

    def run_tests(self, test_cases):
        """Run all test cases and report results.

        test_cases: list of {
            "name": str,
            "inputs": dict,
            "assertions": [{"type": "contains"|"not_contains"|"max_length", "value": ...}]
        }
        """
        results = []
        for tc in test_cases:
            prompt = self.template.format(**tc["inputs"])
            response = self.model_fn(prompt)

            passed = True
            details = []
            for assertion in tc["assertions"]:
                if assertion["type"] == "contains":
                    ok = assertion["value"].lower() in response.lower()
                    details.append(f"contains '{assertion['value']}': {'PASS' if ok else 'FAIL'}")
                elif assertion["type"] == "not_contains":
                    ok = assertion["value"].lower() not in response.lower()
                    details.append(f"not_contains '{assertion['value']}': {'PASS' if ok else 'FAIL'}")
                elif assertion["type"] == "max_length":
                    ok = len(response) <= assertion["value"]
                    details.append(f"max_length {assertion['value']}: {'PASS' if ok else 'FAIL'} ({len(response)})")
                else:
                    ok = True
                passed = passed and ok

            results.append({
                "name": tc["name"],
                "passed": passed,
                "details": details,
                "response_preview": response[:200],
            })

        # Report
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        print(f"\nPrompt Test Results: {passed}/{total} passed")
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  [{status}] {r['name']}")
            for d in r["details"]:
                print(f"         {d}")

        return results


# Example test cases
test_cases = [
    {
        "name": "Basic summarization",
        "inputs": {"num_sentences": 2, "text": "Python is a programming language. It was created by Guido van Rossum. It is widely used in data science."},
        "assertions": [
            {"type": "contains", "value": "python"},
            {"type": "max_length", "value": 500},
        ],
    },
    {
        "name": "No hallucination",
        "inputs": {"num_sentences": 1, "text": "The cat sat on the mat."},
        "assertions": [
            {"type": "contains", "value": "cat"},
            {"type": "not_contains", "value": "dog"},
        ],
    },
]
```

---

## 3. LLM Evaluation

### 3.1 Evaluation Methods

```python
"""
LLM Evaluation Methods:

1. Automated Metrics (fast, cheap):
   - BLEU, ROUGE: n-gram overlap with reference
   - BERTScore: semantic similarity
   - Exact match: for factual QA
   - Regex match: for structured output

2. LLM-as-Judge (medium cost):
   - Use a stronger LLM to evaluate outputs
   - Score on: relevance, accuracy, helpfulness, safety
   - More nuanced than automated metrics
   - Can explain why a response is good/bad

3. Human Evaluation (expensive, gold standard):
   - Domain experts rate outputs
   - Best for subjective quality
   - Needed for final validation
   - Use sampling: evaluate 5-10% of outputs

Evaluation Dimensions:
  - Factual accuracy: Are claims correct?
  - Relevance: Does it answer the question?
  - Completeness: Are all aspects covered?
  - Conciseness: Is it appropriately brief?
  - Safety: No harmful, biased, or PII content?
  - Format compliance: Matches expected structure?
"""


def llm_as_judge(response, reference, question, judge_fn):
    """Use an LLM to evaluate another LLM's response.

    judge_fn: function(prompt) → score_str
    """
    judge_prompt = f"""You are evaluating an AI assistant's response.

Question: {question}
Reference Answer: {reference}
AI Response: {response}

Rate the AI response on these criteria (1-5 each):
1. Accuracy: Are the facts correct?
2. Relevance: Does it answer the question?
3. Completeness: Are all key points covered?
4. Clarity: Is it well-written and clear?

Respond in JSON format:
{{"accuracy": <int>, "relevance": <int>, "completeness": <int>, "clarity": <int>, "explanation": "<str>"}}"""

    result = judge_fn(judge_prompt)
    return json.loads(result)


def batch_evaluate(test_set, model_fn, judge_fn):
    """Evaluate a model on a test set using LLM-as-judge."""
    scores = []
    for item in test_set:
        response = model_fn(item["question"])
        score = llm_as_judge(
            response=response,
            reference=item.get("reference", ""),
            question=item["question"],
            judge_fn=judge_fn,
        )
        score["question"] = item["question"]
        scores.append(score)

    # Aggregate
    avg_scores = {
        k: sum(s[k] for s in scores) / len(scores)
        for k in ["accuracy", "relevance", "completeness", "clarity"]
    }
    print(f"Average scores across {len(scores)} samples:")
    for k, v in avg_scores.items():
        print(f"  {k}: {v:.2f}/5")

    return scores, avg_scores
```

---

## 4. Guardrails

### 4.1 Input and Output Guardrails

```python
"""
Guardrails protect against:
1. Prompt injection (malicious user input)
2. PII leakage (model outputs private data)
3. Toxic/harmful content
4. Off-topic responses
5. Hallucinated facts

Architecture:
  User Input → [Input Guard] → LLM → [Output Guard] → User
                  ↓                       ↓
               Block/modify           Block/modify
"""

import re


class Guardrails:
    """Input and output guardrails for LLM applications."""

    def __init__(self):
        self.input_checks = []
        self.output_checks = []

    def add_input_check(self, name, check_fn):
        self.input_checks.append({"name": name, "fn": check_fn})

    def add_output_check(self, name, check_fn):
        self.output_checks.append({"name": name, "fn": check_fn})

    def check_input(self, user_input):
        """Run all input guardrails. Returns (passed, issues)."""
        issues = []
        for check in self.input_checks:
            passed, detail = check["fn"](user_input)
            if not passed:
                issues.append({"check": check["name"], "detail": detail})
        return len(issues) == 0, issues

    def check_output(self, output):
        """Run all output guardrails. Returns (passed, issues)."""
        issues = []
        for check in self.output_checks:
            passed, detail = check["fn"](output)
            if not passed:
                issues.append({"check": check["name"], "detail": detail})
        return len(issues) == 0, issues


def check_prompt_injection(text):
    """Detect common prompt injection patterns."""
    patterns = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"you\s+are\s+now\s+",
        r"forget\s+(everything|all)",
        r"system\s*:\s*",
        r"<\|im_start\|>",
    ]
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False, f"Prompt injection detected: {pattern}"
    return True, "OK"


def check_pii(text):
    """Detect PII (emails, phone numbers, SSNs) in output."""
    pii_patterns = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    }
    found = []
    for pii_type, pattern in pii_patterns.items():
        if re.search(pattern, text):
            found.append(pii_type)
    if found:
        return False, f"PII detected: {', '.join(found)}"
    return True, "OK"


def check_max_length(max_tokens=4000):
    """Create a length guardrail."""
    def check(text):
        token_estimate = len(text.split())
        if token_estimate > max_tokens:
            return False, f"Response too long: ~{token_estimate} tokens (max {max_tokens})"
        return True, "OK"
    return check


# Setup guardrails
guards = Guardrails()
guards.add_input_check("prompt_injection", check_prompt_injection)
guards.add_output_check("pii", check_pii)
guards.add_output_check("length", check_max_length(4000))

# Usage in application
def safe_llm_call(user_input, model_fn):
    """LLM call with input/output guardrails."""
    # Input check
    passed, issues = guards.check_input(user_input)
    if not passed:
        return {"error": "Input rejected", "issues": issues}

    # Call LLM
    response = model_fn(user_input)

    # Output check
    passed, issues = guards.check_output(response)
    if not passed:
        return {"error": "Output filtered", "issues": issues}

    return {"response": response}
```

---

## 5. RAG Operations

### 5.1 RAG Pipeline Management

```python
"""
RAG Pipeline Components to Manage:

1. Document Ingestion:
   - Chunking strategy (size, overlap)
   - Embedding model version
   - Metadata extraction

2. Vector Store:
   - Index configuration
   - Reindexing schedule
   - Backup strategy

3. Retrieval:
   - Top-k parameter
   - Similarity threshold
   - Reranking model

4. Generation:
   - Prompt template (with context injection)
   - Model selection
   - Temperature / max_tokens

Configuration Example:
"""

RAG_CONFIG = {
    "ingestion": {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "embedding_model": "text-embedding-3-small",
        "embedding_dimension": 1536,
    },
    "retrieval": {
        "top_k": 5,
        "similarity_threshold": 0.7,
        "reranker": None,
    },
    "generation": {
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.1,
        "max_tokens": 1000,
        "prompt_template": (
            "Answer the question based on the context below.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "If the context doesn't contain the answer, say 'I don't know'."
        ),
    },
}


def evaluate_rag(test_set, rag_pipeline):
    """Evaluate RAG pipeline on a test set.

    Metrics:
    - Retrieval precision: % of retrieved chunks that are relevant
    - Retrieval recall: % of relevant chunks that are retrieved
    - Answer accuracy: correctness of generated answers
    - Faithfulness: answers supported by retrieved context (no hallucination)
    """
    results = {
        "retrieval_precision": [],
        "retrieval_recall": [],
        "answer_accuracy": [],
        "faithfulness": [],
    }

    for item in test_set:
        # Run RAG pipeline
        retrieved = rag_pipeline.retrieve(item["question"])
        answer = rag_pipeline.generate(item["question"], retrieved)

        # Check retrieval quality
        relevant_ids = set(item.get("relevant_chunk_ids", []))
        retrieved_ids = set(c["id"] for c in retrieved)
        if retrieved_ids:
            precision = len(relevant_ids & retrieved_ids) / len(retrieved_ids)
            results["retrieval_precision"].append(precision)
        if relevant_ids:
            recall = len(relevant_ids & retrieved_ids) / len(relevant_ids)
            results["retrieval_recall"].append(recall)

    # Aggregate
    for metric, values in results.items():
        if values:
            avg = sum(values) / len(values)
            print(f"{metric}: {avg:.3f}")

    return results
```

---

## 6. Cost Management

### 6.1 Token Cost Tracking

```python
"""
LLM Cost Components:
  - Input tokens:  $X per 1M tokens
  - Output tokens: $Y per 1M tokens (usually 3-5x input)
  - Embedding:     $Z per 1M tokens
  - Fine-tuning:   $W per 1M training tokens

Cost Optimization Strategies:
  1. Prompt optimization: shorter prompts, fewer examples
  2. Caching: cache identical or similar queries
  3. Model routing: use cheaper models for simple tasks
  4. Batch processing: group requests for throughput
  5. Output length limits: set max_tokens appropriately
"""

from collections import defaultdict
from datetime import datetime


class CostTracker:
    """Track and report LLM API costs."""

    # Approximate pricing per 1M tokens (example)
    PRICING = {
        "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    }

    def __init__(self):
        self.usage = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "calls": 0})
        self.daily = defaultdict(lambda: defaultdict(float))

    def record(self, model, input_tokens, output_tokens):
        """Record a single API call."""
        self.usage[model]["input_tokens"] += input_tokens
        self.usage[model]["output_tokens"] += output_tokens
        self.usage[model]["calls"] += 1

        today = datetime.now().strftime("%Y-%m-%d")
        cost = self._compute_cost(model, input_tokens, output_tokens)
        self.daily[today][model] += cost

    def _compute_cost(self, model, input_tokens, output_tokens):
        pricing = self.PRICING.get(model, {"input": 5.0, "output": 25.0})
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

    def report(self):
        """Print cost report."""
        print(f"\n{'='*60}")
        print("LLM Cost Report")
        print(f"{'='*60}")
        total = 0.0
        for model, usage in self.usage.items():
            cost = self._compute_cost(model, usage["input_tokens"], usage["output_tokens"])
            total += cost
            print(f"\n{model}:")
            print(f"  Calls:         {usage['calls']}")
            print(f"  Input tokens:  {usage['input_tokens']:,}")
            print(f"  Output tokens: {usage['output_tokens']:,}")
            print(f"  Cost:          ${cost:.4f}")
        print(f"\nTotal cost: ${total:.4f}")

    def check_budget(self, daily_limit=10.0):
        """Check if daily spending is within budget."""
        today = datetime.now().strftime("%Y-%m-%d")
        today_cost = sum(self.daily[today].values())
        if today_cost > daily_limit:
            print(f"BUDGET ALERT: ${today_cost:.2f} > ${daily_limit:.2f} limit")
            return False
        return True
```

### 6.2 Model Router

```python
class ModelRouter:
    """Route requests to appropriate models based on complexity.

    Simple queries → cheap model (Haiku)
    Complex queries → capable model (Sonnet/Opus)
    """

    def __init__(self, classifier_fn=None):
        self.classifier_fn = classifier_fn or self._default_classifier
        self.model_map = {
            "simple": "claude-haiku-4-5-20251001",
            "medium": "claude-sonnet-4-20250514",
            "complex": "claude-opus-4-20250514",
        }

    def _default_classifier(self, query):
        """Simple heuristic classifier."""
        word_count = len(query.split())
        if word_count < 20 and "?" in query:
            return "simple"
        elif word_count < 100:
            return "medium"
        else:
            return "complex"

    def route(self, query):
        """Select model for query."""
        complexity = self.classifier_fn(query)
        model = self.model_map[complexity]
        return model, complexity


# Usage
router = ModelRouter()
model, complexity = router.route("What is the capital of France?")
print(f"Query → {complexity} → {model}")
# Query → simple → claude-haiku-4-5-20251001
```

---

## 7. LLM Monitoring

### 7.1 Key Metrics

```python
"""
LLM Monitoring Metrics:

1. Performance:
   - Latency (p50, p95, p99)
   - Throughput (requests/second)
   - Token usage per request

2. Quality:
   - User satisfaction (thumbs up/down)
   - Response relevance score
   - Hallucination rate
   - Guardrail trigger rate

3. Cost:
   - Daily/monthly spend
   - Cost per request
   - Cost per user

4. Safety:
   - Prompt injection attempts
   - PII leakage incidents
   - Toxicity detections
   - Off-topic rate

5. RAG-specific:
   - Retrieval latency
   - Context relevance score
   - "I don't know" rate (no context found)
"""


class LLMMonitor:
    """Monitor LLM application health."""

    def __init__(self):
        self.metrics = defaultdict(list)

    def record_request(self, latency_ms, input_tokens, output_tokens,
                       model, user_feedback=None, guardrail_triggered=False):
        """Record a single LLM request."""
        self.metrics["latency"].append(latency_ms)
        self.metrics["input_tokens"].append(input_tokens)
        self.metrics["output_tokens"].append(output_tokens)
        self.metrics["total_tokens"].append(input_tokens + output_tokens)
        if user_feedback is not None:
            self.metrics["feedback"].append(1 if user_feedback == "positive" else 0)
        if guardrail_triggered:
            self.metrics["guardrail_triggers"].append(1)
        else:
            self.metrics["guardrail_triggers"].append(0)

    def get_summary(self):
        """Get monitoring summary."""
        import numpy as np

        latencies = self.metrics["latency"]
        tokens = self.metrics["total_tokens"]
        feedback = self.metrics["feedback"]
        guards = self.metrics["guardrail_triggers"]

        return {
            "total_requests": len(latencies),
            "latency_p50_ms": np.percentile(latencies, 50) if latencies else 0,
            "latency_p95_ms": np.percentile(latencies, 95) if latencies else 0,
            "latency_p99_ms": np.percentile(latencies, 99) if latencies else 0,
            "avg_tokens": np.mean(tokens) if tokens else 0,
            "satisfaction_rate": np.mean(feedback) if feedback else None,
            "guardrail_rate": np.mean(guards) if guards else 0,
        }
```

---

## 8. Practice Problems

### Exercise 1: Prompt Registry + A/B Test

```python
"""
Build a prompt management system:
1. Create a PromptRegistry that stores prompt versions
2. Register 3 versions of a summarization prompt
3. Implement A/B testing: randomly assign users to prompt versions
4. Track which version produces better user satisfaction
5. Promote the winning version as default
6. Add rollback capability
"""
```

### Exercise 2: RAG Evaluation Pipeline

```python
"""
Build a RAG evaluation pipeline:
1. Create a test set with questions + expected answers + relevant chunks
2. Run the RAG pipeline on each test question
3. Measure: retrieval precision/recall, answer accuracy, faithfulness
4. Compare two configurations (e.g., chunk_size=256 vs 512)
5. Add guardrails: PII check, hallucination check
6. Generate an evaluation report with pass/fail criteria
"""
```

---

## 9. Summary

### Key Takeaways

| Concept | Description |
|---------|-------------|
| **LLMOps** | MLOps adapted for LLM-specific challenges |
| **Prompt Registry** | Version-controlled prompts with rollback |
| **LLM-as-Judge** | Use LLMs to evaluate other LLMs |
| **Guardrails** | Input/output safety checks (injection, PII, toxicity) |
| **RAG Ops** | Manage retrieval pipeline: chunking, embedding, reranking |
| **Cost management** | Token tracking, model routing, caching |
| **Model routing** | Use cheaper models for simple tasks |

### Best Practices

1. **Version prompts** — treat prompts like code; track every change
2. **Test before deploy** — run prompt test suite before production rollout
3. **Guard input and output** — prompt injection + PII + toxicity checks
4. **Monitor continuously** — latency, cost, user satisfaction, guardrail triggers
5. **Route by complexity** — use cheaper models for simple queries
6. **Evaluate with LLM-as-judge** — more reliable than automated metrics for open-ended tasks

### Navigation

- **Previous**: L14 — DVC Data Version Control
- Return to **L10** (Drift Detection) for monitoring concepts that apply to LLMs

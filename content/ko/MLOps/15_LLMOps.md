# LLMOps

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 프롬프트 버전 관리(prompt versioning), RAG 파이프라인, LLM-as-judge 평가, 토큰 기반 비용 관리를 포함하는 전통적인 MLOps와 LLMOps의 핵심 운영 차이를 파악할 수 있다
2. 개발, 스테이징, 프로덕션 환경에서 프롬프트 템플릿을 버전 관리하고 저장 및 서빙하는 프롬프트 레지스트리(prompt registry)를 구현할 수 있다
3. 검색 품질(retrieval quality), 지연 시간(latency), 환각(hallucination) 비율을 프로덕션 메트릭으로 추적하여 RAG 파이프라인을 구축하고 모니터링할 수 있다
4. LLM 애플리케이션에서 유해 출력, 프롬프트 주입(prompt injection) 공격, PII 유출을 감지하고 필터링하는 가드레일(guardrails)을 설계하고 배포할 수 있다
5. 캐싱(caching), 프롬프트 압축(prompt compression), 태스크 복잡도 기반 모델 라우팅(model routing) 등 LLM 비용 모니터링 및 최적화 전략을 구현할 수 있다

---

## 개요

LLMOps는 대규모 언어 모델(Large Language Model) 애플리케이션을 위해 MLOps 관행을 확장한 것입니다. 정형 데이터로 훈련하는 전통적인 ML 모델과 달리, LLM은 고유한 운영 패턴을 요구합니다: 프롬프트 엔지니어링(Prompt Engineering) 및 버전 관리, 검색 증강 생성(Retrieval-Augmented Generation, RAG) 파이프라인, 정답 없이 하는 평가, 가드레일(Guardrails), 비용 관리. 이 레슨에서는 LLM 기반 애플리케이션의 운영 생명주기를 다룹니다.

---

## 1. LLMOps vs 전통적인 MLOps

### 1.1 주요 차이점

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

## 2. 프롬프트 관리

### 2.1 프롬프트 레지스트리(Prompt Registry)

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
        # 콘텐츠 해시(content hash)로 중복 등록 감지 및 무결성 검증
        content_hash = hashlib.sha256(template.encode()).hexdigest()[:12]
        version = len(self._index.get(name, {}).get("versions", [])) + 1

        entry = {
            "version": version,
            "template": template,
            "hash": content_hash,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        # 플레인 텍스트 파일로 저장 — git diff와 코드 리뷰에서 사람이 읽을 수 있음
        prompt_dir = self.registry_dir / name
        prompt_dir.mkdir(exist_ok=True)
        (prompt_dir / f"v{version}.txt").write_text(template)

        # 인덱스(index)와 템플릿 분리 — 파일 파싱 없이 빠른 조회 가능
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

        # 기본값은 활성 버전 — 호출자가 어떤 버전이 라이브인지 알 필요 없음
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

### 2.2 프롬프트 테스팅

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
            # 모든 어서션(assertion) 실행 — 대소문자 무시로 대문자 차이에 의한 오탐 방지
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
        # 환각(hallucination) 테스트: 원본에 없는 사실을 모델이 추가하지 않아야 함
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

## 3. LLM 평가

### 3.1 평가 방법

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
    # 참조 답변이 포함된 구조화된 평가 프롬프트 — 정답(ground truth) 기준으로 평가 고정
    # JSON 출력 형식으로 자동 파싱 및 집계 가능
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
    # 개별 항목 평가 — 집계뿐 아니라 샘플별 분석 가능
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

## 4. 가드레일(Guardrails)

### 4.1 입력 및 출력 가드레일

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
    # 알려진 공격 벡터(attack vector)의 정규식 패턴 — LLM 호출 전 빠른 1차 필터
    patterns = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"you\s+are\s+now\s+",
        r"forget\s+(everything|all)",
        r"system\s*:\s*",
        r"<\|im_start\|>",   # ChatML 토큰 — 원시 형식을 통한 인젝션(injection) 시도
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
        # 단어 수를 토큰 근사치로 사용 — 토크나이저(tokenizer) 의존성 없이 약 75% 정확도
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
    # 입력 검사 먼저 — 악의적 쿼리에 토큰을 소비하기 전에 거부
    passed, issues = guards.check_input(user_input)
    if not passed:
        return {"error": "Input rejected", "issues": issues}

    # Call LLM
    response = model_fn(user_input)

    # 출력 검사 — 모델이 생성한 PII 유출 등의 문제를 포착
    passed, issues = guards.check_output(response)
    if not passed:
        return {"error": "Output filtered", "issues": issues}

    return {"response": response}
```

---

## 5. RAG 운영

### 5.1 RAG 파이프라인 관리

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

# 코드로서의 설정(Config-as-code) 패턴 — 이 파일을 버전 관리하여 실험 간 변경 사항 추적
RAG_CONFIG = {
    "ingestion": {
        "chunk_size": 512,         # 균형: 너무 작으면 문맥 손실, 너무 크면 관련성 희석
        "chunk_overlap": 50,       # 오버랩(overlap)으로 청크 경계에서 핵심 정보 분리 방지
        "embedding_model": "text-embedding-3-small",
        "embedding_dimension": 1536,
    },
    "retrieval": {
        "top_k": 5,                # 청크가 많을수록 더 많은 문맥이지만 지연 시간과 비용 증가
        "similarity_threshold": 0.7,  # 이 값 미만은 관련 없는 노이즈일 가능성이 높음
        "reranker": None,
    },
    "generation": {
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.1,        # 사실 기반 Q&A에서 낮은 온도(temperature) — 환각(hallucination) 감소
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

        # 검색 품질과 생성 품질을 분리 — 실패 원인 격리
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

## 6. 비용 관리

### 6.1 토큰 비용 추적

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

    # 1M 토큰당 대략적인 가격 — 제공사(provider) 가격 변경 시 업데이트 필요
    # 출력 토큰이 입력의 3-5배 비용 — 자기회귀(autoregressive) 생성이 연산 집약적이므로
    PRICING = {
        "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    }

    def __init__(self):
        # 모델별 추적으로 어떤 모델이 비용을 유발하는지 파악
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
        # 일일 단위로 점검 — 월간 점검은 폭주 비용을 너무 늦게 발견
        today = datetime.now().strftime("%Y-%m-%d")
        today_cost = sum(self.daily[today].values())
        if today_cost > daily_limit:
            print(f"BUDGET ALERT: ${today_cost:.2f} > ${daily_limit:.2f} limit")
            return False
        return True
```

### 6.2 모델 라우터(Model Router)

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
        # 휴리스틱 기준선 — 프로덕션 정확도를 위해 학습된 분류기(classifier)로 교체
        word_count = len(query.split())
        if word_count < 20 and "?" in query:
            return "simple"   # 짧은 질문 → Haiku가 Opus 대비 약 80% 비용 절감
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

## 7. LLM 모니터링

### 7.1 핵심 지표

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
        # 총 토큰(total tokens) 추적 — 비용과 상관관계가 있고 프롬프트 비대화(prompt bloat) 감지
        self.metrics["total_tokens"].append(input_tokens + output_tokens)
        # 이진 피드백(binary feedback)은 최소지만 실행 가능 — 긍정률 추세로 품질 회귀 감지
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

        # 지연 시간은 평균이 아닌 백분위수(percentile) — p95/p99가 사용자가 체감하는 꼬리 지연 시간(tail latency) 포착
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

## 8. 연습 문제

### 연습 1: 프롬프트 레지스트리 + A/B 테스트

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

### 연습 2: RAG 평가 파이프라인

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

## 9. 요약

### 핵심 정리

| 개념 | 설명 |
|---------|-------------|
| **LLMOps** | LLM 고유의 과제에 맞게 조정된 MLOps |
| **프롬프트 레지스트리(Prompt Registry)** | 롤백(Rollback)이 가능한 버전 관리형 프롬프트 |
| **LLM-as-Judge** | LLM을 사용해 다른 LLM의 응답을 평가 |
| **가드레일(Guardrails)** | 입력/출력 안전 검사 (인젝션, 개인정보(PII), 유해 콘텐츠) |
| **RAG 운영** | 검색 파이프라인 관리: 청킹(Chunking), 임베딩(Embedding), 재순위화(Reranking) |
| **비용 관리** | 토큰 추적, 모델 라우팅(Model Routing), 캐싱(Caching) |
| **모델 라우팅** | 단순 작업에는 저렴한 모델 활용 |

### 모범 사례

1. **프롬프트 버전 관리** — 프롬프트를 코드처럼 취급하여 모든 변경 이력을 추적
2. **배포 전 테스트** — 프로덕션 배포 전 프롬프트 테스트 스위트(Test Suite)를 실행
3. **입출력 가드레일 적용** — 프롬프트 인젝션(Prompt Injection) + 개인정보(PII) + 유해 콘텐츠 검사
4. **지속적인 모니터링** — 지연 시간, 비용, 사용자 만족도, 가드레일 트리거 현황 감시
5. **복잡도에 따른 라우팅** — 단순한 쿼리에는 저렴한 모델 사용
6. **LLM-as-Judge로 평가** — 개방형 작업에서는 자동화 지표보다 신뢰도가 높음

---

## 연습 문제

### 연습 1: 프롬프트 레지스트리(Prompt Registry) 확장

2.1절의 `PromptRegistry` 클래스를 시작점으로 사용하여 다음 기능을 추가하세요:

1. 지정된 버전을 활성 버전으로 설정하는 `promote(name, version)` 메서드 (순방향 승격을 명확하게 표현하는 `rollback`의 별칭)
2. 프롬프트의 모든 버전을 `version`, `created_at`, `hash`, `template`, `metadata` 필드를 포함한 YAML 파일로 작성하는 `export(name, output_path)` 메서드
3. A/B 테스트(A/B test) 메커니즘: `hash(user_id) % len(versions)` 기반으로 사용자를 지정된 버전 중 하나에 결정론적으로 배정하는 `get_ab(name, user_id, versions)` 메서드 추가 (같은 사용자는 항상 같은 프롬프트 버전을 받음)
4. 100명의 서로 다른 사용자 ID로 `get_ab("summarizer", user_id, [1, 2])`를 호출하여 A/B 배정이 대략 50/50으로 분할되는지 확인하세요

### 연습 2: 프롬프트 테스트 스위트(Test Suite) 구축

다음 조건을 만족해야 하는 고객 지원 프롬프트에 대한 완전한 `PromptTestSuite` 테스트를 작성하세요:

- 주문 상태에 대한 질문에 답변
- 직접 환불을 거부하고 상담원으로 연결
- 응답을 200단어 이내로 유지
- 쿼리에 케이스 번호가 언급된 경우 항상 포함

다음을 포함한 5개 이상의 테스트 케이스를 작성하세요:
1. 기본적인 주문 상태 조회 (주문 세부 정보를 언급해야 함)
2. 환불 요청 (환불 승인이 아닌 연결 처리)
3. 장황한 고객 불만 (응답은 200단어 이내여야 함)
4. 케이스 번호 "CS-4821"이 포함된 조회 (응답에 "CS-4821"이 포함되어야 함)
5. 요리에 관한 주제 외 질문 (정중하게 거절하고 연결해야 함)

적절한 어서션(assertion) 유형(`contains`, `not_contains`, `max_length`)과 함께 `PromptTestSuite.run_tests()` 메서드를 사용하세요. 각 테스트에 대한 미리 준비된 응답을 반환하도록 `model_fn`을 모킹(mocking)하세요.

### 연습 3: 의료 Q&A 시스템을 위한 가드레일(Guardrail) 설계

의료 정보 챗봇은 구체적인 용량 권고나 진단을 제공해서는 안 됩니다. 이 애플리케이션을 위한 가드레일을 설계하세요:

1. "take X mg", "you have [disease]", "prescribe", "diagnosis is"와 같은 구문을 감지하는 `check_medical_advice(text)` 입/출력 가드를 작성하세요
2. 응답에 "consult a healthcare professional" 또는 "this is not medical advice"와 같은 면책 조항 문구가 포함되어 있는지 확인하는 `check_disclaimer(text)` 출력 가드를 추가하세요
3. `check_medical_advice`를 입/출력 검사로, `check_disclaimer`를 출력 전용 검사로 설정하여 두 가드를 `Guardrails` 인스턴스에 연결하세요
4. 모든 가드를 통과하는 것, 입력에서 의료 조언 가드를 트리거하는 것, 출력에서 면책 조항 누락 가드를 트리거하는 것의 3가지 테스트 케이스를 작성하세요
5. 논의: 차단된 응답은 일반적인 오류를 반환해야 할까요, 아니면 사용자에게 구체적인 설명을 제공해야 할까요? 각각의 장단점은 무엇인가요?

### 연습 4: 비용 예산 알림 시스템 구현

6.1절의 `CostTracker` 클래스를 확장하여 예산 관리 기능을 추가하세요:

1. 모델별 일일 한도를 저장하는 `set_budget(model, daily_limit)` 메서드 추가
2. 기록 후마다 `check_budget()`을 호출하고 모델별 또는 전체 예산이 초과되면 알림을 출력하도록 `record()` 수정
3. 오늘의 지출을 기반으로 예상 월간 비용을 반환하는 `monthly_projection()` 메서드 추가 (지출 속도가 일정하다고 가정)
4. 각 모델의 API 호출당 평균 비용을 반환하는 `cost_per_request()` 메서드 추가
5. 랜덤 토큰 수(입력: 200–2,000 토큰, 출력: 50–500 토큰)로 세 가지 모델에 걸쳐 1,000번의 API 호출을 시뮬레이션하고, 전체 비용 보고서와 월간 예측을 출력하세요

### 연습 5: 종단간(End-to-End) LLMOps 파이프라인 설계

하루 10,000건의 쿼리를 처리하는 프로덕션 RAG 기반 고객 지원 시스템의 운영 아키텍처를 설계하세요. 설계는 다음을 다루어야 합니다:

1. **프롬프트 버전 관리**: 프롬프트 변경 사항을 어떻게 버전 관리하고 배포할 건가요? 새 프롬프트가 프로덕션으로 가기 전에 어떤 승인 프로세스가 필요할까요?
2. **RAG 구성 관리**: 어떤 RAG 파라미터(`chunk_size`, `top_k`, `similarity_threshold`)를 버전 관리 구성으로 취급할 건가요? 두 구성 간의 A/B 테스트는 어떻게 실행할 건가요?
3. **평가 전략**: 주 200개의 쿼리를 샘플링하고, LLM-as-judge로 정확도, 관련성, 안전성을 점수화하며, 평균 안전성 점수가 4.0/5 이하로 떨어지면 사람의 검토를 트리거하는 주간 평가 파이프라인을 정의하세요
4. **비용 통제**: 월 $500 예산을 기준으로 예산 내에서 운영하기 위한 모델 라우팅(model routing), 캐싱(caching), 사용자별 속도 제한(rate limiting) 구현 방법을 설명하세요
5. **가드레일 커버리지**: 고객 지원 컨텍스트에 배포할 5가지 구체적인 가드레일을 나열하고 각각이 방지하는 실패 모드를 설명하세요

---

### 탐색

- **이전**: L14 — DVC 데이터 버전 관리
- **다음**: [L16 — 모델 테스트와 검증](16_Model_Testing_and_Validation.md)
- **L10**(드리프트 감지)으로 돌아가 LLM에도 적용되는 모니터링 개념 확인

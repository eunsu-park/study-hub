# 17. 캡스톤(Capstone): 프롬프트 라이브러리(Prompt Library)

**이전**: [16. 에이전트 프롬프팅 패턴](./16_Agent_Prompting_Patterns.md) | **다음**: [00. 개요](./00_Overview.md)

## 학습 목표

- 명확한 분류 체계와 조직 구조를 갖춘 재사용 가능한 프롬프트 라이브러리(Prompt Library)를 설계한다
- 여러 차원에서 프롬프트 품질을 측정하는 평가 하네스(Evaluation Harness)를 구축한다
- 마이그레이션(Migration)과 롤백(Rollback) 기능을 갖춘 프롬프트 버전 관리 시스템을 구현한다
- 대화형 테스트와 반복 작업을 위한 프롬프트 플레이그라운드(Playground)를 만든다
- 템플릿, 평가 도구, CLI 인터페이스를 포함한 완전한 프롬프트 라이브러리 프로젝트를 완성한다

---

이 캡스톤(Capstone) 수업은 이전 16개 수업의 모든 내용을 하나의 실용적인 프로젝트로 통합합니다: 프로덕션급 프롬프트 라이브러리(Prompt Library) 구축. 프롬프트 라이브러리는 LLM에 의존하는 모든 팀의 핵심 인프라입니다 — 프롬프트를 저장하고, 버전을 관리하고, 평가하고, 검색 가능하게 만듭니다. 이 수업이 끝나면 CLI 인터페이스, 평가 하네스(Evaluation Harness), 템플릿 시스템을 갖춘 완전한 프롬프트 라이브러리 시스템을 설계하고 구현하게 됩니다.

프로젝트는 5개의 마일스톤(Milestone)으로 나뉘며, 각각 이전 단계를 기반으로 합니다. 순서대로 완료하면 실제 프로젝트에서 사용할 수 있는 기능적인 프롬프트 라이브러리를 갖게 됩니다.

## 목차
1. [재사용 가능한 프롬프트 라이브러리 설계](#1-재사용-가능한-프롬프트-라이브러리-설계)
2. [프롬프트 분류 체계와 조직화](#2-프롬프트-분류-체계와-조직화)
3. [평가 하네스 구축](#3-평가-하네스-구축)
4. [프롬프트 버전 관리 구현](#4-프롬프트-버전-관리-구현)
5. [프롬프트 플레이그라운드 만들기](#5-프롬프트-플레이그라운드-만들기)
6. [프롬프트 통합 테스트](#6-프롬프트-통합-테스트)
7. [문서화와 검색 가능성](#7-문서화와-검색-가능성)
8. [전체 프로젝트: 모든 것을 하나로](#8-전체-프로젝트-모든-것을-하나로)

---

## 1. 재사용 가능한 프롬프트 라이브러리 설계

프롬프트 라이브러리는 단순한 텍스트 파일 폴더가 아닙니다. 프롬프트와 이를 사용하는 애플리케이션 사이에 명확한 인터페이스, 메타데이터, 계약을 갖춘 시스템입니다.

### 1.1 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                     Prompt Library Architecture                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Template  │  │ Version  │  │  Eval    │  │   CLI    │        │
│  │  Engine   │  │  Store   │  │ Harness  │  │Interface │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │              │              │              │              │
│  ┌────▼──────────────▼──────────────▼──────────────▼─────┐      │
│  │                   Prompt Registry                      │      │
│  │  (Central store: prompts, metadata, versions, evals)   │      │
│  └────────────────────────┬──────────────────────────────┘      │
│                            │                                      │
│  ┌────────────────────────▼──────────────────────────────┐      │
│  │                  File System Storage                    │      │
│  │  prompts/                                               │      │
│  │  ├── classification/                                    │      │
│  │  │   ├── sentiment.yaml                                 │      │
│  │  │   └── intent.yaml                                    │      │
│  │  ├── generation/                                        │      │
│  │  │   ├── summary.yaml                                   │      │
│  │  │   └── translation.yaml                               │      │
│  │  └── agent/                                             │      │
│  │      ├── research_agent.yaml                            │      │
│  │      └── code_reviewer.yaml                             │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 프롬프트 정의 형식

```python
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path


SAMPLE_PROMPT_YAML = """
id: sentiment-classifier
name: Sentiment Classifier
version: "2.1.0"
category: classification
description: >
  Classify text sentiment into positive, negative, or neutral
  with a confidence score. Handles multi-sentence input.

model:
  preferred: claude-sonnet-4-20250514
  compatible:
    - claude-haiku-4-20250514
    - claude-sonnet-4-20250514

system_prompt: |
  You are a sentiment classifier. Analyze the given text and determine
  its overall sentiment.

  RULES:
  - Classify as exactly one of: positive, negative, neutral
  - Provide a confidence score from 0.0 to 1.0
  - Consider the entire text, not just individual words
  - Handle sarcasm and irony when detectable

  OUTPUT FORMAT (JSON only, no other text):
  {"sentiment": "positive|negative|neutral", "confidence": 0.0-1.0}

user_template: "Classify the sentiment of this text:\\n\\n{{text}}"

variables:
  - name: text
    type: string
    required: true
    description: The text to classify

tags:
  - nlp
  - sentiment
  - classification
  - production

metadata:
  author: prompt-team
  created: "2025-01-15"
  updated: "2025-03-10"
  review_status: approved

evaluation:
  test_cases:
    - input:
        text: "I love this product! Best purchase ever."
      expected:
        sentiment: positive
        confidence_min: 0.8
    - input:
        text: "Terrible experience. Would not recommend."
      expected:
        sentiment: negative
        confidence_min: 0.8
    - input:
        text: "The package arrived on Tuesday."
      expected:
        sentiment: neutral
  metrics:
    accuracy_threshold: 0.9
    latency_p95_ms: 2000
"""


@dataclass
class PromptDefinition:
    """Full definition of a prompt in the library."""
    id: str
    name: str
    version: str
    category: str
    description: str
    system_prompt: str
    user_template: str
    variables: list[dict] = field(default_factory=list)
    model: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "PromptDefinition":
        data = yaml.safe_load(yaml_str)
        return cls(**data)

    @classmethod
    def from_file(cls, path: str) -> "PromptDefinition":
        content = Path(path).read_text()
        return cls.from_yaml(content)

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self), default_flow_style=False, sort_keys=False)

    def validate(self) -> list[str]:
        """Validate the prompt definition."""
        errors = []
        if not self.id:
            errors.append("Missing 'id'")
        if not self.system_prompt:
            errors.append("Missing 'system_prompt'")
        if not self.version:
            errors.append("Missing 'version'")

        # Check template variables are declared
        import re
        template_vars = set(re.findall(r"\{\{(\w+)\}\}", self.user_template))
        declared_vars = {v["name"] for v in self.variables}
        undeclared = template_vars - declared_vars
        if undeclared:
            errors.append(f"Template uses undeclared variables: {undeclared}")

        return errors


# Parse and validate
prompt_def = PromptDefinition.from_yaml(SAMPLE_PROMPT_YAML)
errors = prompt_def.validate()
print(f"Prompt: {prompt_def.name} v{prompt_def.version}")
print(f"Category: {prompt_def.category}")
print(f"Tags: {prompt_def.tags}")
print(f"Validation errors: {errors}")
```

### 1.3 프롬프트 라이브러리 핵심

```python
import json
from pathlib import Path
from dataclasses import asdict


class PromptLibrary:
    """Core prompt library with file-based storage."""

    def __init__(self, library_dir: str):
        self.library_dir = Path(library_dir)
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, PromptDefinition] = {}
        self._load_all()

    def _load_all(self):
        """Load all prompts from disk."""
        for yaml_file in self.library_dir.rglob("*.yaml"):
            try:
                prompt = PromptDefinition.from_file(str(yaml_file))
                self._index[prompt.id] = prompt
            except Exception as e:
                print(f"Warning: Could not load {yaml_file}: {e}")

    def add(self, prompt: PromptDefinition) -> list[str]:
        """Add a prompt to the library. Returns validation errors if any."""
        errors = prompt.validate()
        if errors:
            return errors

        # Save to file
        category_dir = self.library_dir / prompt.category
        category_dir.mkdir(exist_ok=True)
        file_path = category_dir / f"{prompt.id}.yaml"
        file_path.write_text(prompt.to_yaml())

        self._index[prompt.id] = prompt
        return []

    def get(self, prompt_id: str) -> PromptDefinition | None:
        """Retrieve a prompt by ID."""
        return self._index.get(prompt_id)

    def search(
        self,
        query: str = "",
        category: str = "",
        tags: list[str] | None = None,
    ) -> list[PromptDefinition]:
        """Search prompts by query, category, and/or tags."""
        results = list(self._index.values())

        if query:
            q = query.lower()
            results = [
                p for p in results
                if q in p.name.lower() or q in p.description.lower()
            ]

        if category:
            results = [p for p in results if p.category == category]

        if tags:
            results = [
                p for p in results
                if any(t in p.tags for t in tags)
            ]

        return results

    def list_categories(self) -> dict[str, int]:
        """List categories and prompt counts."""
        cats: dict[str, int] = {}
        for p in self._index.values():
            cats[p.category] = cats.get(p.category, 0) + 1
        return cats

    def remove(self, prompt_id: str) -> bool:
        """Remove a prompt from the library."""
        prompt = self._index.pop(prompt_id, None)
        if prompt:
            file_path = self.library_dir / prompt.category / f"{prompt.id}.yaml"
            if file_path.exists():
                file_path.unlink()
            return True
        return False

    def export_catalog(self) -> str:
        """Export a markdown catalog of all prompts."""
        lines = ["# Prompt Library Catalog", ""]
        for category, count in sorted(self.list_categories().items()):
            lines.append(f"## {category.title()} ({count} prompts)")
            prompts = self.search(category=category)
            for p in sorted(prompts, key=lambda x: x.name):
                lines.append(f"- **{p.name}** (`{p.id}` v{p.version}): {p.description[:80]}")
            lines.append("")
        return "\n".join(lines)
```

---

## 2. 프롬프트 분류 체계와 조직화

잘 조직된 분류 체계(Taxonomy)는 프롬프트를 검색 가능하게 만들고 중복을 방지합니다.

### 2.1 분류 체계 설계

```python
PROMPT_TAXONOMY = {
    "classification": {
        "description": "Categorize text into predefined labels",
        "subcategories": [
            "sentiment", "intent", "topic", "toxicity", "language_detection",
        ],
        "common_patterns": ["Multi-label", "Binary", "Hierarchical"],
    },
    "extraction": {
        "description": "Pull structured data from unstructured text",
        "subcategories": [
            "entities", "relationships", "key_phrases", "contact_info", "financial_data",
        ],
        "common_patterns": ["Schema-driven", "Open-ended", "Template-based"],
    },
    "generation": {
        "description": "Create new text content",
        "subcategories": [
            "summary", "translation", "paraphrase", "creative", "technical_writing",
        ],
        "common_patterns": ["Length-controlled", "Style-matched", "Template-filled"],
    },
    "analysis": {
        "description": "Analyze and evaluate text",
        "subcategories": [
            "code_review", "fact_check", "comparison", "critique", "risk_assessment",
        ],
        "common_patterns": ["Rubric-based", "Comparative", "Checklist"],
    },
    "conversation": {
        "description": "Multi-turn interaction patterns",
        "subcategories": [
            "customer_support", "tutoring", "interview", "negotiation", "therapy",
        ],
        "common_patterns": ["Persona-driven", "Goal-oriented", "Adaptive"],
    },
    "agent": {
        "description": "Autonomous agent system prompts",
        "subcategories": [
            "research", "coding", "planning", "data_analysis", "orchestration",
        ],
        "common_patterns": ["ReAct", "Planning-first", "Multi-agent"],
    },
}


def find_category(description: str) -> list[tuple[str, str, float]]:
    """Suggest the best category for a prompt based on its description."""
    description_lower = description.lower()
    scores = []

    keywords = {
        "classification": ["classify", "categorize", "label", "detect", "sentiment"],
        "extraction": ["extract", "pull", "parse", "identify", "find entities"],
        "generation": ["generate", "write", "create", "summarize", "translate"],
        "analysis": ["analyze", "review", "evaluate", "compare", "assess"],
        "conversation": ["chat", "support", "tutor", "converse", "dialogue"],
        "agent": ["agent", "tool", "plan", "autonomous", "search and"],
    }

    for category, kws in keywords.items():
        score = sum(1 for kw in kws if kw in description_lower)
        if score > 0:
            scores.append((category, PROMPT_TAXONOMY[category]["description"], score))

    return sorted(scores, key=lambda x: x[2], reverse=True)


# Test taxonomy lookup
suggestions = find_category("Extract email addresses and phone numbers from text")
for cat, desc, score in suggestions:
    print(f"  {cat} (score: {score}): {desc}")
```

### 2.2 명명 규칙

```python
import re


def validate_prompt_id(prompt_id: str) -> list[str]:
    """Validate prompt ID follows naming conventions."""
    errors = []

    # Must be lowercase with hyphens
    if prompt_id != prompt_id.lower():
        errors.append("Prompt ID must be lowercase")

    if not re.match(r'^[a-z][a-z0-9-]*[a-z0-9]$', prompt_id):
        errors.append("Prompt ID must start with a letter, use only lowercase letters, numbers, and hyphens")

    if len(prompt_id) < 3:
        errors.append("Prompt ID must be at least 3 characters")

    if len(prompt_id) > 50:
        errors.append("Prompt ID must be 50 characters or fewer")

    if "--" in prompt_id:
        errors.append("No consecutive hyphens allowed")

    return errors


# Naming convention guide
NAMING_CONVENTION = """
## Prompt ID Naming Convention

Format: {category}-{specific-name}[-{variant}]

Examples:
  classify-sentiment           (classification, sentiment analysis)
  classify-sentiment-financial (variant for financial text)
  extract-contacts             (extraction, contact information)
  generate-summary-short       (generation, short summary variant)
  agent-research               (agent, research assistant)
  chat-support-billing         (conversation, billing support)

Rules:
  - Lowercase only
  - Hyphens to separate words
  - Start with the category prefix
  - Add variant suffix for specialized versions
  - 3-50 characters
  - No consecutive hyphens
"""

# Test naming validation
test_ids = [
    "classify-sentiment",
    "ClassifySentiment",  # Bad: uppercase
    "c",                   # Bad: too short
    "extract--contacts",   # Bad: consecutive hyphens
]

for pid in test_ids:
    errors = validate_prompt_id(pid)
    status = "VALID" if not errors else f"INVALID: {errors}"
    print(f"  '{pid}' → {status}")
```

---

## 3. 평가 하네스 구축

평가 하네스(Evaluation Harness)는 정의된 테스트 케이스에 대해 프롬프트를 자동으로 테스트하고 품질 보고서를 생성합니다.

### 3.1 평가 프레임워크

```python
import anthropic
import json
import time
from dataclasses import dataclass, field

client = anthropic.Anthropic()


@dataclass
class EvalResult:
    test_name: str
    passed: bool
    expected: dict
    actual: dict
    checks: dict[str, bool]
    latency_ms: float
    tokens_used: int
    error: str = ""


@dataclass
class EvalReport:
    prompt_id: str
    prompt_version: str
    total_tests: int
    passed: int
    failed: int
    pass_rate: float
    avg_latency_ms: float
    results: list[EvalResult]
    metadata: dict = field(default_factory=dict)


class EvaluationHarness:
    """Evaluate prompts against their test suites."""

    def __init__(self):
        self.results_history: list[EvalReport] = []

    def evaluate_prompt(self, prompt: PromptDefinition) -> EvalReport:
        """Run the full evaluation suite for a prompt."""
        test_cases = prompt.evaluation.get("test_cases", [])
        if not test_cases:
            return EvalReport(
                prompt_id=prompt.id,
                prompt_version=prompt.version,
                total_tests=0, passed=0, failed=0, pass_rate=0,
                avg_latency_ms=0, results=[],
                metadata={"error": "No test cases defined"},
            )

        results = []
        for i, test_case in enumerate(test_cases):
            result = self._run_single_test(prompt, test_case, f"test_{i}")
            results.append(result)

        passed = sum(1 for r in results if r.passed)
        total = len(results)
        latencies = [r.latency_ms for r in results]

        report = EvalReport(
            prompt_id=prompt.id,
            prompt_version=prompt.version,
            total_tests=total,
            passed=passed,
            failed=total - passed,
            pass_rate=round(passed / total, 3) if total > 0 else 0,
            avg_latency_ms=round(sum(latencies) / len(latencies), 1) if latencies else 0,
            results=results,
        )

        # Check against thresholds
        metrics = prompt.evaluation.get("metrics", {})
        if metrics.get("accuracy_threshold"):
            report.metadata["accuracy_check"] = report.pass_rate >= metrics["accuracy_threshold"]
        if metrics.get("latency_p95_ms"):
            sorted_latencies = sorted(latencies)
            p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)] if sorted_latencies else 0
            report.metadata["latency_check"] = p95 <= metrics["latency_p95_ms"]
            report.metadata["latency_p95"] = round(p95, 1)

        self.results_history.append(report)
        return report

    def _run_single_test(
        self, prompt: PromptDefinition, test_case: dict, test_name: str,
    ) -> EvalResult:
        """Run a single test case."""
        # Render user message from template
        user_message = prompt.user_template
        for var_name, var_value in test_case.get("input", {}).items():
            user_message = user_message.replace(f"{{{{{var_name}}}}}", str(var_value))

        model = prompt.model.get("preferred", "claude-sonnet-4-20250514")

        start = time.time()
        try:
            response = client.messages.create(
                model=model,
                max_tokens=512,
                system=prompt.system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            latency = (time.time() - start) * 1000
            output_text = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens
        except Exception as e:
            return EvalResult(
                test_name=test_name, passed=False,
                expected=test_case.get("expected", {}),
                actual={}, checks={},
                latency_ms=0, tokens_used=0, error=str(e),
            )

        # Parse output and check against expected
        expected = test_case.get("expected", {})
        checks = {}

        try:
            actual = json.loads(output_text)
        except json.JSONDecodeError:
            actual = {"raw_output": output_text}
            checks["valid_json"] = False
            return EvalResult(
                test_name=test_name, passed=False,
                expected=expected, actual=actual,
                checks=checks, latency_ms=latency, tokens_used=tokens,
            )

        checks["valid_json"] = True

        # Check expected values
        for key, expected_value in expected.items():
            if key.endswith("_min"):
                base_key = key[:-4]
                actual_value = actual.get(base_key, 0)
                checks[f"{base_key}_above_min"] = actual_value >= expected_value
            elif key.endswith("_max"):
                base_key = key[:-4]
                actual_value = actual.get(base_key, float("inf"))
                checks[f"{base_key}_below_max"] = actual_value <= expected_value
            else:
                checks[f"{key}_matches"] = actual.get(key) == expected_value

        all_passed = all(checks.values())

        return EvalResult(
            test_name=test_name, passed=all_passed,
            expected=expected, actual=actual,
            checks=checks, latency_ms=latency, tokens_used=tokens,
        )

    def compare_versions(self, report_a: EvalReport, report_b: EvalReport) -> dict:
        """Compare evaluation results between two versions."""
        return {
            "version_a": report_a.prompt_version,
            "version_b": report_b.prompt_version,
            "pass_rate_change": round(report_b.pass_rate - report_a.pass_rate, 3),
            "latency_change_ms": round(report_b.avg_latency_ms - report_a.avg_latency_ms, 1),
            "regression": report_b.pass_rate < report_a.pass_rate,
            "improvement": report_b.pass_rate > report_a.pass_rate,
        }

    def generate_report_text(self, report: EvalReport) -> str:
        """Generate a human-readable evaluation report."""
        lines = [
            f"# Evaluation Report: {report.prompt_id} v{report.prompt_version}",
            f"\nPass rate: {report.pass_rate:.1%} ({report.passed}/{report.total_tests})",
            f"Average latency: {report.avg_latency_ms:.0f}ms",
        ]

        if report.metadata:
            lines.append(f"\nThreshold checks: {report.metadata}")

        lines.append("\n## Test Results")
        for r in report.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{status}] {r.test_name} ({r.latency_ms:.0f}ms)")
            if not r.passed:
                lines.append(f"    Expected: {r.expected}")
                lines.append(f"    Actual: {r.actual}")
                lines.append(f"    Checks: {r.checks}")

        return "\n".join(lines)
```

### 3.2 커스텀 평가기

```python
from abc import ABC, abstractmethod


class PromptEvaluator(ABC):
    """Base class for custom prompt evaluators."""

    @abstractmethod
    def evaluate(self, prompt_output: str, expected: dict, context: dict) -> dict:
        """Evaluate a prompt output. Returns {"passed": bool, "score": float, "details": dict}."""
        pass


class JSONFormatEvaluator(PromptEvaluator):
    """Check that output is valid JSON with required fields."""

    def __init__(self, required_fields: list[str]):
        self.required_fields = required_fields

    def evaluate(self, prompt_output: str, expected: dict, context: dict) -> dict:
        try:
            data = json.loads(prompt_output)
        except json.JSONDecodeError:
            return {"passed": False, "score": 0.0, "details": {"error": "Invalid JSON"}}

        missing = [f for f in self.required_fields if f not in data]
        if missing:
            return {"passed": False, "score": 0.5, "details": {"missing_fields": missing}}

        return {"passed": True, "score": 1.0, "details": {"fields_present": list(data.keys())}}


class LengthEvaluator(PromptEvaluator):
    """Check that output falls within a word count range."""

    def __init__(self, min_words: int, max_words: int):
        self.min_words = min_words
        self.max_words = max_words

    def evaluate(self, prompt_output: str, expected: dict, context: dict) -> dict:
        word_count = len(prompt_output.split())
        in_range = self.min_words <= word_count <= self.max_words
        score = 1.0 if in_range else max(0, 1 - abs(word_count - (self.min_words + self.max_words) / 2) / self.max_words)
        return {
            "passed": in_range,
            "score": round(score, 3),
            "details": {
                "word_count": word_count,
                "range": f"{self.min_words}-{self.max_words}",
            },
        }


class SemanticSimilarityEvaluator(PromptEvaluator):
    """Use an LLM to check semantic similarity between output and expected."""

    def evaluate(self, prompt_output: str, expected: dict, context: dict) -> dict:
        expected_text = expected.get("expected_content", "")
        if not expected_text:
            return {"passed": True, "score": 1.0, "details": {"note": "No expected content"}}

        response = client.messages.create(
            model="claude-haiku-4-20250514",
            max_tokens=200,
            system="Compare two texts. Return JSON: {\"similar\": true/false, \"score\": 0.0-1.0, \"reason\": \"...\"}",
            messages=[
                {
                    "role": "user",
                    "content": f"Text A: {prompt_output[:500]}\n\nText B: {expected_text[:500]}",
                }
            ],
        )
        result = json.loads(response.content[0].text)
        return {
            "passed": result["similar"],
            "score": result["score"],
            "details": {"reason": result["reason"]},
        }
```

---

## 4. 프롬프트 버전 관리 구현

프롬프트의 버전 관리(Version Control)는 롤백(Rollback), 비교, 감사 추적(Audit Trail)을 가능하게 합니다.

### 4.1 버전 저장소

```python
import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class PromptVersion:
    prompt_id: str
    version: str
    content_hash: str
    definition: dict  # Full prompt definition as dict
    created_at: str
    change_description: str = ""
    eval_results: dict = field(default_factory=dict)

    @classmethod
    def from_definition(cls, definition: PromptDefinition, change_description: str = "") -> "PromptVersion":
        content = definition.system_prompt + definition.user_template
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return cls(
            prompt_id=definition.id,
            version=definition.version,
            content_hash=content_hash,
            definition=asdict(definition),
            created_at=datetime.now(timezone.utc).isoformat(),
            change_description=change_description,
        )


class VersionStore:
    """Store and manage prompt versions."""

    def __init__(self, store_dir: str):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _prompt_dir(self, prompt_id: str) -> Path:
        d = self.store_dir / prompt_id
        d.mkdir(exist_ok=True)
        return d

    def save_version(self, version: PromptVersion):
        """Save a new version."""
        version_file = self._prompt_dir(version.prompt_id) / f"v{version.version}.json"
        version_file.write_text(json.dumps(asdict(version), indent=2))

        # Update the latest pointer
        latest_file = self._prompt_dir(version.prompt_id) / "latest.json"
        latest_file.write_text(json.dumps({"version": version.version}))

    def get_version(self, prompt_id: str, version: str | None = None) -> PromptVersion | None:
        """Load a specific version, or the latest."""
        prompt_dir = self._prompt_dir(prompt_id)

        if version is None:
            latest_file = prompt_dir / "latest.json"
            if latest_file.exists():
                version = json.loads(latest_file.read_text())["version"]
            else:
                return None

        version_file = prompt_dir / f"v{version}.json"
        if not version_file.exists():
            return None

        data = json.loads(version_file.read_text())
        return PromptVersion(**data)

    def list_versions(self, prompt_id: str) -> list[dict]:
        """List all versions of a prompt."""
        prompt_dir = self._prompt_dir(prompt_id)
        versions = []
        for f in sorted(prompt_dir.glob("v*.json")):
            data = json.loads(f.read_text())
            versions.append({
                "version": data["version"],
                "hash": data["content_hash"],
                "created_at": data["created_at"],
                "change": data.get("change_description", ""),
            })
        return versions

    def diff_versions(self, prompt_id: str, version_a: str, version_b: str) -> dict:
        """Compare two versions."""
        va = self.get_version(prompt_id, version_a)
        vb = self.get_version(prompt_id, version_b)

        if not va or not vb:
            return {"error": "Version not found"}

        import difflib
        diff = list(difflib.unified_diff(
            va.definition["system_prompt"].splitlines(keepends=True),
            vb.definition["system_prompt"].splitlines(keepends=True),
            fromfile=f"v{version_a}",
            tofile=f"v{version_b}",
        ))

        return {
            "version_a": version_a,
            "version_b": version_b,
            "hash_a": va.content_hash,
            "hash_b": vb.content_hash,
            "same_content": va.content_hash == vb.content_hash,
            "diff": "".join(diff) if diff else "(identical)",
        }

    def rollback(self, prompt_id: str, target_version: str) -> bool:
        """Set a previous version as the latest."""
        version = self.get_version(prompt_id, target_version)
        if not version:
            return False
        latest_file = self._prompt_dir(prompt_id) / "latest.json"
        latest_file.write_text(json.dumps({"version": target_version}))
        return True
```

---

## 5. 프롬프트 플레이그라운드 만들기

프롬프트 플레이그라운드(Playground)는 대화형 테스트, 반복 작업, 비교를 가능하게 합니다.

### 5.1 플레이그라운드 엔진

```python
import anthropic
import json
import time

client = anthropic.Anthropic()


class PromptPlayground:
    """Interactive prompt testing environment."""

    def __init__(self):
        self.history: list[dict] = []

    def run(
        self,
        system_prompt: str,
        user_message: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        temperature: float = 1.0,
    ) -> dict:
        """Run a prompt and collect detailed metrics."""
        start = time.time()

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        latency = (time.time() - start) * 1000
        output = response.content[0].text

        result = {
            "output": output,
            "model": model,
            "latency_ms": round(latency, 1),
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            "stop_reason": response.stop_reason,
            "temperature": temperature,
            "word_count": len(output.split()),
        }

        self.history.append(result)
        return result

    def compare(
        self,
        variants: list[dict],
        user_message: str,
        model: str = "claude-sonnet-4-20250514",
    ) -> list[dict]:
        """Compare multiple prompt variants side by side."""
        results = []
        for variant in variants:
            result = self.run(
                system_prompt=variant["system_prompt"],
                user_message=user_message,
                model=model,
            )
            result["variant_name"] = variant["name"]
            results.append(result)

        return results

    def sweep_temperature(
        self,
        system_prompt: str,
        user_message: str,
        temperatures: list[float] | None = None,
    ) -> list[dict]:
        """Test a prompt across different temperature settings."""
        temperatures = temperatures or [0.0, 0.3, 0.5, 0.7, 1.0]
        results = []
        for temp in temperatures:
            result = self.run(system_prompt, user_message, temperature=temp)
            result["temperature"] = temp
            results.append(result)
        return results

    def format_comparison(self, results: list[dict]) -> str:
        """Format comparison results as a readable table."""
        lines = ["| Variant | Latency | Tokens | Words | Preview |",
                 "|---------|---------|--------|-------|---------|"]
        for r in results:
            name = r.get("variant_name", f"temp={r.get('temperature', '?')}")
            preview = r["output"][:60].replace("\n", " ") + "..."
            lines.append(
                f"| {name} | {r['latency_ms']:.0f}ms | "
                f"{r['total_tokens']} | {r['word_count']} | {preview} |"
            )
        return "\n".join(lines)


# Usage
playground = PromptPlayground()

# Compare two prompt variants
results = playground.compare(
    variants=[
        {
            "name": "concise",
            "system_prompt": "Summarize text in exactly one sentence.",
        },
        {
            "name": "detailed",
            "system_prompt": "Provide a detailed summary with key points.",
        },
    ],
    user_message="The global economy grew by 3.2% in 2024, driven by strong consumer spending in the US and recovery in European manufacturing. However, inflation remained above central bank targets in most major economies, leading to continued high interest rates. Emerging markets outperformed expectations, with India and Southeast Asia leading growth.",
)

print(playground.format_comparison(results))
```

---

## 6. 프롬프트 통합 테스트

통합 테스트(Integration Test)는 프롬프트가 더 큰 애플리케이션 컨텍스트 내에서 올바르게 작동하는지 검증합니다.

### 6.1 통합 테스트 프레임워크

```python
import anthropic
import json
from dataclasses import dataclass, field

client = anthropic.Anthropic()


@dataclass
class IntegrationTest:
    name: str
    description: str
    prompt_id: str
    setup: dict = field(default_factory=dict)  # Pre-conditions
    input_data: dict = field(default_factory=dict)
    assertions: list[dict] = field(default_factory=list)
    cleanup: dict = field(default_factory=dict)  # Post-conditions


class IntegrationTestRunner:
    """Run integration tests for prompts in context."""

    def __init__(self, library: PromptLibrary):
        self.library = library

    def run_test(self, test: IntegrationTest) -> dict:
        """Run a single integration test."""
        prompt = self.library.get(test.prompt_id)
        if not prompt:
            return {"passed": False, "error": f"Prompt '{test.prompt_id}' not found"}

        # Render the user message
        user_message = prompt.user_template
        for var_name, var_value in test.input_data.items():
            user_message = user_message.replace(f"{{{{{var_name}}}}}", str(var_value))

        # Execute
        try:
            response = client.messages.create(
                model=prompt.model.get("preferred", "claude-sonnet-4-20250514"),
                max_tokens=1024,
                system=prompt.system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            output = response.content[0].text
        except Exception as e:
            return {"passed": False, "error": str(e)}

        # Run assertions
        assertion_results = []
        for assertion in test.assertions:
            result = self._check_assertion(output, assertion)
            assertion_results.append(result)

        all_passed = all(r["passed"] for r in assertion_results)

        return {
            "test_name": test.name,
            "passed": all_passed,
            "output_preview": output[:200],
            "assertions": assertion_results,
        }

    def _check_assertion(self, output: str, assertion: dict) -> dict:
        """Check a single assertion against the output."""
        assertion_type = assertion["type"]

        if assertion_type == "contains":
            passed = assertion["value"].lower() in output.lower()
            return {"type": "contains", "value": assertion["value"], "passed": passed}

        elif assertion_type == "not_contains":
            passed = assertion["value"].lower() not in output.lower()
            return {"type": "not_contains", "value": assertion["value"], "passed": passed}

        elif assertion_type == "json_valid":
            try:
                json.loads(output)
                passed = True
            except json.JSONDecodeError:
                passed = False
            return {"type": "json_valid", "passed": passed}

        elif assertion_type == "json_field_equals":
            try:
                data = json.loads(output)
                actual = data.get(assertion["field"])
                passed = actual == assertion["value"]
            except (json.JSONDecodeError, KeyError):
                passed = False
            return {
                "type": "json_field_equals",
                "field": assertion["field"],
                "expected": assertion["value"],
                "passed": passed,
            }

        elif assertion_type == "max_length":
            word_count = len(output.split())
            passed = word_count <= assertion["value"]
            return {"type": "max_length", "max_words": assertion["value"],
                    "actual_words": word_count, "passed": passed}

        return {"type": assertion_type, "passed": False, "error": "Unknown assertion type"}

    def run_suite(self, tests: list[IntegrationTest]) -> dict:
        """Run a full test suite."""
        results = []
        for test in tests:
            result = self.run_test(test)
            results.append(result)

        passed = sum(1 for r in results if r["passed"])
        return {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": round(passed / len(results), 3) if results else 0,
            "results": results,
        }
```

---

## 7. 문서화와 검색 가능성

좋은 문서화는 프롬프트 라이브러리가 실제로 사용되느냐 무시되느냐의 차이를 만듭니다.

### 7.1 자동 생성 문서

```python
class PromptDocGenerator:
    """Generate documentation for the prompt library."""

    def __init__(self, library: PromptLibrary):
        self.library = library

    def generate_prompt_doc(self, prompt_id: str) -> str:
        """Generate documentation for a single prompt."""
        prompt = self.library.get(prompt_id)
        if not prompt:
            return f"Prompt '{prompt_id}' not found."

        lines = [
            f"# {prompt.name}",
            f"\n**ID**: `{prompt.id}` | **Version**: {prompt.version} | **Category**: {prompt.category}",
            f"\n{prompt.description}",
            f"\n## Model Compatibility",
            f"- Preferred: `{prompt.model.get('preferred', 'N/A')}`",
        ]
        for m in prompt.model.get("compatible", []):
            lines.append(f"- Compatible: `{m}`")

        lines.append("\n## System Prompt")
        lines.append(f"```\n{prompt.system_prompt}\n```")

        lines.append("\n## User Template")
        lines.append(f"```\n{prompt.user_template}\n```")

        if prompt.variables:
            lines.append("\n## Variables")
            lines.append("| Name | Type | Required | Description |")
            lines.append("|------|------|----------|-------------|")
            for v in prompt.variables:
                req = "Yes" if v.get("required") else "No"
                lines.append(f"| `{v['name']}` | {v['type']} | {req} | {v.get('description', '')} |")

        if prompt.tags:
            lines.append(f"\n## Tags\n{', '.join(f'`{t}`' for t in prompt.tags)}")

        test_cases = prompt.evaluation.get("test_cases", [])
        if test_cases:
            lines.append(f"\n## Test Cases ({len(test_cases)} defined)")
            for i, tc in enumerate(test_cases):
                lines.append(f"\n### Test {i + 1}")
                lines.append(f"- Input: `{tc.get('input', {})}`")
                lines.append(f"- Expected: `{tc.get('expected', {})}`")

        lines.append(f"\n## Usage Example")
        lines.append("```python")
        lines.append("import anthropic")
        lines.append("client = anthropic.Anthropic()")
        lines.append("")
        lines.append("response = client.messages.create(")
        lines.append(f'    model="{prompt.model.get("preferred", "claude-sonnet-4-20250514")}",')
        lines.append("    max_tokens=1024,")
        lines.append(f"    system=\"\"\"{prompt.system_prompt[:100]}...\"\"\",")
        example_msg = prompt.user_template
        for v in prompt.variables:
            example_msg = example_msg.replace(f"{{{{{v['name']}}}}}", f"<{v['name']}>")
        lines.append(f'    messages=[{{"role": "user", "content": "{example_msg}"}}],')
        lines.append(")")
        lines.append("```")

        return "\n".join(lines)

    def generate_catalog(self) -> str:
        """Generate a complete library catalog."""
        return self.library.export_catalog()

    def generate_quickstart(self) -> str:
        """Generate a quickstart guide."""
        categories = self.library.list_categories()
        total = sum(categories.values())

        lines = [
            "# Prompt Library Quickstart",
            f"\nThis library contains **{total} prompts** across {len(categories)} categories.",
            "\n## Categories",
        ]

        for cat, count in sorted(categories.items()):
            lines.append(f"- **{cat}**: {count} prompts")

        lines.extend([
            "\n## Quick Usage",
            "```python",
            "from prompt_library import PromptLibrary",
            "",
            "library = PromptLibrary('./prompts')",
            "",
            "# Search for prompts",
            "results = library.search(tags=['sentiment'])",
            "",
            "# Get a specific prompt",
            "prompt = library.get('classify-sentiment')",
            "",
            "# Use with Anthropic API",
            "import anthropic",
            "client = anthropic.Anthropic()",
            "response = client.messages.create(",
            "    model=prompt.model['preferred'],",
            "    system=prompt.system_prompt,",
            "    messages=[{'role': 'user', 'content': 'Your text here'}],",
            ")",
            "```",
        ])

        return "\n".join(lines)
```

---

## 8. 전체 프로젝트: 모든 것을 하나로

이 섹션에서는 모든 구성 요소가 통합된 CLI 기반 프롬프트 라이브러리인 전체 프로젝트를 정의합니다.

### 8.1 CLI 인터페이스

```python
import argparse
import sys
import json
from pathlib import Path


class PromptLibraryCLI:
    """Command-line interface for the prompt library."""

    def __init__(self, library_dir: str = "./prompts", version_dir: str = "./versions"):
        self.library = PromptLibrary(library_dir)
        self.version_store = VersionStore(version_dir)
        self.eval_harness = EvaluationHarness()
        self.playground = PromptPlayground()

    def cmd_list(self, args):
        """List all prompts."""
        if args.category:
            prompts = self.library.search(category=args.category)
        elif args.tag:
            prompts = self.library.search(tags=[args.tag])
        else:
            prompts = self.library.search()

        if not prompts:
            print("No prompts found.")
            return

        print(f"\nFound {len(prompts)} prompt(s):\n")
        for p in sorted(prompts, key=lambda x: x.id):
            print(f"  {p.id:30s} v{p.version:8s} [{p.category}]  {p.name}")

    def cmd_show(self, args):
        """Show details of a prompt."""
        prompt = self.library.get(args.prompt_id)
        if not prompt:
            print(f"Prompt '{args.prompt_id}' not found.")
            return

        doc_gen = PromptDocGenerator(self.library)
        print(doc_gen.generate_prompt_doc(args.prompt_id))

    def cmd_eval(self, args):
        """Evaluate a prompt against its test suite."""
        prompt = self.library.get(args.prompt_id)
        if not prompt:
            print(f"Prompt '{args.prompt_id}' not found.")
            return

        print(f"Evaluating {prompt.id} v{prompt.version}...")
        report = self.eval_harness.evaluate_prompt(prompt)
        print(self.eval_harness.generate_report_text(report))

    def cmd_test(self, args):
        """Run a prompt in the playground."""
        prompt = self.library.get(args.prompt_id)
        if not prompt:
            print(f"Prompt '{args.prompt_id}' not found.")
            return

        user_message = prompt.user_template
        for var in prompt.variables:
            value = input(f"  Enter {var['name']} ({var.get('description', '')}): ")
            user_message = user_message.replace(f"{{{{{var['name']}}}}}", value)

        result = self.playground.run(
            system_prompt=prompt.system_prompt,
            user_message=user_message,
            model=prompt.model.get("preferred", "claude-sonnet-4-20250514"),
        )

        print(f"\n--- Output ---")
        print(result["output"])
        print(f"\n--- Metrics ---")
        print(f"  Latency: {result['latency_ms']:.0f}ms")
        print(f"  Tokens: {result['total_tokens']} (in: {result['input_tokens']}, out: {result['output_tokens']})")

    def cmd_versions(self, args):
        """List versions of a prompt."""
        versions = self.version_store.list_versions(args.prompt_id)
        if not versions:
            print(f"No versions found for '{args.prompt_id}'.")
            return

        print(f"\nVersions of {args.prompt_id}:")
        for v in versions:
            print(f"  v{v['version']}  [{v['hash']}]  {v['created_at'][:10]}  {v.get('change', '')}")

    def cmd_diff(self, args):
        """Diff two versions."""
        result = self.version_store.diff_versions(
            args.prompt_id, args.version_a, args.version_b
        )
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        print(f"\nDiff {args.prompt_id}: v{args.version_a} → v{args.version_b}")
        print(f"Same content: {result['same_content']}")
        print(result["diff"])

    def cmd_rollback(self, args):
        """Roll back to a previous version."""
        success = self.version_store.rollback(args.prompt_id, args.version)
        if success:
            print(f"Rolled back {args.prompt_id} to v{args.version}")
        else:
            print(f"Failed to rollback. Version not found.")

    def cmd_catalog(self, args):
        """Export the library catalog."""
        doc_gen = PromptDocGenerator(self.library)
        print(doc_gen.generate_catalog())


def build_cli():
    """Build the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Prompt Library CLI — manage, test, and evaluate prompts"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list
    list_parser = subparsers.add_parser("list", help="List prompts")
    list_parser.add_argument("--category", "-c", help="Filter by category")
    list_parser.add_argument("--tag", "-t", help="Filter by tag")

    # show
    show_parser = subparsers.add_parser("show", help="Show prompt details")
    show_parser.add_argument("prompt_id", help="Prompt ID")

    # eval
    eval_parser = subparsers.add_parser("eval", help="Evaluate a prompt")
    eval_parser.add_argument("prompt_id", help="Prompt ID")

    # test
    test_parser = subparsers.add_parser("test", help="Test a prompt interactively")
    test_parser.add_argument("prompt_id", help="Prompt ID")

    # versions
    ver_parser = subparsers.add_parser("versions", help="List prompt versions")
    ver_parser.add_argument("prompt_id", help="Prompt ID")

    # diff
    diff_parser = subparsers.add_parser("diff", help="Diff two versions")
    diff_parser.add_argument("prompt_id", help="Prompt ID")
    diff_parser.add_argument("version_a", help="First version")
    diff_parser.add_argument("version_b", help="Second version")

    # rollback
    rb_parser = subparsers.add_parser("rollback", help="Rollback to a version")
    rb_parser.add_argument("prompt_id", help="Prompt ID")
    rb_parser.add_argument("version", help="Target version")

    # catalog
    subparsers.add_parser("catalog", help="Export library catalog")

    return parser


# Entry point
def main():
    parser = build_cli()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    cli = PromptLibraryCLI()
    cmd_method = getattr(cli, f"cmd_{args.command}", None)
    if cmd_method:
        cmd_method(args)
    else:
        print(f"Unknown command: {args.command}")


# To run: python prompt_library.py list
# To run: python prompt_library.py eval classify-sentiment
# To run: python prompt_library.py test classify-sentiment
```

### 8.2 프로젝트 구조

```
prompt-library/
├── prompt_library/
│   ├── __init__.py
│   ├── core.py           # PromptDefinition, PromptLibrary
│   ├── versioning.py     # VersionStore, PromptVersion
│   ├── evaluation.py     # EvaluationHarness, Evaluators
│   ├── playground.py     # PromptPlayground
│   ├── templates.py      # Template engine
│   ├── docs.py           # Documentation generator
│   └── cli.py            # CLI interface
├── prompts/
│   ├── classification/
│   │   ├── sentiment.yaml
│   │   └── intent.yaml
│   ├── generation/
│   │   └── summary.yaml
│   └── agent/
│       └── research.yaml
├── versions/             # Version history storage
├── tests/
│   ├── test_core.py
│   ├── test_eval.py
│   └── test_versioning.py
├── requirements.txt
└── README.md
```

---

## 연습문제

다음 연습문제는 프로젝트 마일스톤(Milestone)입니다. 순서대로 완료하면 전체 프롬프트 라이브러리를 구축할 수 있습니다.

### 연습문제 1: 핵심 라이브러리와 프롬프트 정의 (마일스톤 1)

핵심 프롬프트 라이브러리를 구축합니다: `PromptDefinition` 데이터 모델, YAML 직렬화(Serialization), `PromptLibrary` 레지스트리, 그리고 최소 5개의 시드(Seed) 프롬프트 세트.

**요구사항:**
- 모든 필드를 포함한 `PromptDefinition` 데이터클래스(Dataclass) (id, name, version, category, system_prompt, user_template, variables, model, tags, metadata, evaluation)
- YAML 로드/저장 왕복 변환(Round-trip)
- add, get, search, remove, list_categories를 갖춘 `PromptLibrary`
- 입력 유효성 검증(Validation) (프롬프트 ID 형식, 필수 필드, 변수 일관성)
- 최소 3개 카테고리에 걸쳐 5개의 시드 프롬프트

<details><summary>정답 보기</summary>

```python
import yaml
import re
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class PromptDefinition:
    id: str
    name: str
    version: str
    category: str
    description: str
    system_prompt: str
    user_template: str
    variables: list[dict] = field(default_factory=list)
    model: dict = field(default_factory=lambda: {"preferred": "claude-sonnet-4-20250514"})
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "PromptDefinition":
        data = yaml.safe_load(yaml_str)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self), default_flow_style=False, sort_keys=False)

    def validate(self) -> list[str]:
        errors = []
        if not self.id or not re.match(r'^[a-z][a-z0-9-]*[a-z0-9]$', self.id):
            errors.append("Invalid prompt ID format")
        if not self.name:
            errors.append("Name is required")
        if not self.version or not re.match(r'^\d+\.\d+\.\d+$', self.version):
            errors.append("Version must be in semver format (e.g., 1.0.0)")
        if not self.system_prompt:
            errors.append("System prompt is required")
        if not self.category:
            errors.append("Category is required")

        # Check template variables
        template_vars = set(re.findall(r"\{\{(\w+)\}\}", self.user_template))
        declared_vars = {v["name"] for v in self.variables}
        undeclared = template_vars - declared_vars
        if undeclared:
            errors.append(f"Undeclared template variables: {undeclared}")

        return errors


class PromptLibrary:
    def __init__(self, library_dir: str):
        self.library_dir = Path(library_dir)
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, PromptDefinition] = {}
        self._load_all()

    def _load_all(self):
        for yaml_file in self.library_dir.rglob("*.yaml"):
            try:
                prompt = PromptDefinition.from_yaml(yaml_file.read_text())
                self._index[prompt.id] = prompt
            except Exception as e:
                print(f"Warning: {yaml_file}: {e}")

    def add(self, prompt: PromptDefinition) -> list[str]:
        errors = prompt.validate()
        if errors:
            return errors
        cat_dir = self.library_dir / prompt.category
        cat_dir.mkdir(exist_ok=True)
        (cat_dir / f"{prompt.id}.yaml").write_text(prompt.to_yaml())
        self._index[prompt.id] = prompt
        return []

    def get(self, prompt_id: str) -> PromptDefinition | None:
        return self._index.get(prompt_id)

    def remove(self, prompt_id: str) -> bool:
        p = self._index.pop(prompt_id, None)
        if p:
            f = self.library_dir / p.category / f"{p.id}.yaml"
            if f.exists():
                f.unlink()
            return True
        return False

    def search(self, query: str = "", category: str = "", tags: list[str] | None = None) -> list[PromptDefinition]:
        results = list(self._index.values())
        if query:
            q = query.lower()
            results = [p for p in results if q in p.name.lower() or q in p.description.lower()]
        if category:
            results = [p for p in results if p.category == category]
        if tags:
            results = [p for p in results if any(t in p.tags for t in tags)]
        return results

    def list_categories(self) -> dict[str, int]:
        cats: dict[str, int] = {}
        for p in self._index.values():
            cats[p.category] = cats.get(p.category, 0) + 1
        return cats


# Create seed prompts
SEED_PROMPTS = [
    PromptDefinition(
        id="classify-sentiment",
        name="Sentiment Classifier",
        version="1.0.0",
        category="classification",
        description="Classify text sentiment into positive/negative/neutral",
        system_prompt='Classify sentiment. Return JSON: {"sentiment": "positive|negative|neutral", "confidence": 0.0-1.0}',
        user_template="Classify: {{text}}",
        variables=[{"name": "text", "type": "string", "required": True}],
        tags=["nlp", "sentiment"],
        evaluation={"test_cases": [
            {"input": {"text": "I love this!"}, "expected": {"sentiment": "positive"}},
            {"input": {"text": "Terrible."}, "expected": {"sentiment": "negative"}},
        ]},
    ),
    PromptDefinition(
        id="classify-intent",
        name="Intent Detector",
        version="1.0.0",
        category="classification",
        description="Detect user intent from a support message",
        system_prompt='Detect intent. Return JSON: {"intent": "...", "confidence": 0.0-1.0}',
        user_template="Message: {{message}}",
        variables=[{"name": "message", "type": "string", "required": True}],
        tags=["nlp", "intent", "support"],
    ),
    PromptDefinition(
        id="generate-summary",
        name="Text Summarizer",
        version="1.0.0",
        category="generation",
        description="Summarize text with configurable length",
        system_prompt="Summarize the text in {{length}} format. Be faithful to the source.",
        user_template="Summarize:\n\n{{text}}",
        variables=[
            {"name": "text", "type": "string", "required": True},
            {"name": "length", "type": "string", "required": True},
        ],
        tags=["summarization", "generation"],
    ),
    PromptDefinition(
        id="extract-contacts",
        name="Contact Extractor",
        version="1.0.0",
        category="extraction",
        description="Extract contact information from text",
        system_prompt='Extract contacts. Return JSON: {"name": ..., "email": ..., "phone": ..., "company": ...}. Use null for missing.',
        user_template="Extract contacts from:\n\n{{text}}",
        variables=[{"name": "text", "type": "string", "required": True}],
        tags=["extraction", "contacts"],
    ),
    PromptDefinition(
        id="agent-researcher",
        name="Research Agent",
        version="1.0.0",
        category="agent",
        description="Research agent that searches and synthesizes information",
        system_prompt="You are a research agent. Use tools to find information. Cite sources. Be thorough.",
        user_template="Research: {{question}}",
        variables=[{"name": "question", "type": "string", "required": True}],
        tags=["agent", "research"],
    ),
]

# Build the library
library = PromptLibrary("/tmp/prompt_library_demo")
for prompt in SEED_PROMPTS:
    errors = library.add(prompt)
    status = "OK" if not errors else f"ERRORS: {errors}"
    print(f"  Added {prompt.id}: {status}")

print(f"\nCategories: {library.list_categories()}")
print(f"Total prompts: {len(library.search())}")

# Test search
nlp_prompts = library.search(tags=["nlp"])
print(f"NLP prompts: {[p.id for p in nlp_prompts]}")
```

</details>

### 연습문제 2: 평가 하네스 (마일스톤 2)

프롬프트를 정의된 테스트 케이스에 대해 테스트하고 품질 보고서를 생성하는 평가 하네스(Evaluation Harness)를 구축합니다.

**요구사항:**
- 프롬프트의 `evaluation` 섹션에 정의된 모든 테스트 케이스 실행
- JSON 출력 형식, 필드 값, 임계값 조건 확인
- 테스트별 지연 시간(Latency)과 토큰 사용량 측정
- 합격/불합격, 메트릭, 권장 사항이 포함된 구조화된 보고서 생성
- 커스텀 평가기 지원 (형식 확인, 길이 확인, 의미적 유사도)

<details><summary>정답 보기</summary>

```python
import anthropic
import json
import time
from dataclasses import dataclass, field

client = anthropic.Anthropic()


@dataclass
class TestResult:
    test_index: int
    passed: bool
    checks: dict[str, bool]
    latency_ms: float
    tokens: int
    output_preview: str
    expected: dict
    actual: dict
    error: str = ""


@dataclass
class EvalReport:
    prompt_id: str
    version: str
    total: int
    passed: int
    pass_rate: float
    avg_latency: float
    total_tokens: int
    results: list[TestResult]
    thresholds_met: dict = field(default_factory=dict)


class EvalHarness:
    """Evaluation harness for prompt testing."""

    def evaluate(self, prompt: PromptDefinition) -> EvalReport:
        test_cases = prompt.evaluation.get("test_cases", [])
        results = []

        for i, tc in enumerate(test_cases):
            result = self._run_test(prompt, tc, i)
            results.append(result)

        passed = sum(1 for r in results if r.passed)
        total = len(results)
        latencies = [r.latency_ms for r in results]

        # Check thresholds
        metrics = prompt.evaluation.get("metrics", {})
        thresholds = {}
        if metrics.get("accuracy_threshold"):
            rate = passed / total if total > 0 else 0
            thresholds["accuracy"] = {
                "threshold": metrics["accuracy_threshold"],
                "actual": round(rate, 3),
                "met": rate >= metrics["accuracy_threshold"],
            }
        if metrics.get("latency_p95_ms") and latencies:
            p95 = sorted(latencies)[int(len(latencies) * 0.95)]
            thresholds["latency_p95"] = {
                "threshold": metrics["latency_p95_ms"],
                "actual": round(p95, 1),
                "met": p95 <= metrics["latency_p95_ms"],
            }

        return EvalReport(
            prompt_id=prompt.id,
            version=prompt.version,
            total=total,
            passed=passed,
            pass_rate=round(passed / total, 3) if total else 0,
            avg_latency=round(sum(latencies) / len(latencies), 1) if latencies else 0,
            total_tokens=sum(r.tokens for r in results),
            results=results,
            thresholds_met=thresholds,
        )

    def _run_test(self, prompt: PromptDefinition, test_case: dict, index: int) -> TestResult:
        # Render template
        user_msg = prompt.user_template
        for var_name, val in test_case.get("input", {}).items():
            user_msg = user_msg.replace(f"{{{{{var_name}}}}}", str(val))

        model = prompt.model.get("preferred", "claude-sonnet-4-20250514")
        start = time.time()

        try:
            response = client.messages.create(
                model=model, max_tokens=512,
                system=prompt.system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            output = response.content[0].text
            latency = (time.time() - start) * 1000
            tokens = response.usage.input_tokens + response.usage.output_tokens
        except Exception as e:
            return TestResult(
                test_index=index, passed=False, checks={},
                latency_ms=0, tokens=0, output_preview="",
                expected=test_case.get("expected", {}), actual={}, error=str(e),
            )

        # Parse and check
        expected = test_case.get("expected", {})
        checks = {}

        try:
            actual = json.loads(output)
            checks["valid_json"] = True
        except json.JSONDecodeError:
            return TestResult(
                test_index=index, passed=False,
                checks={"valid_json": False},
                latency_ms=latency, tokens=tokens,
                output_preview=output[:100],
                expected=expected, actual={"raw": output[:200]},
            )

        for key, exp_val in expected.items():
            if key.endswith("_min"):
                base = key[:-4]
                checks[f"{base}>=min"] = actual.get(base, 0) >= exp_val
            elif key.endswith("_max"):
                base = key[:-4]
                checks[f"{base}<=max"] = actual.get(base, float("inf")) <= exp_val
            else:
                checks[f"{key}_match"] = actual.get(key) == exp_val

        return TestResult(
            test_index=index,
            passed=all(checks.values()),
            checks=checks,
            latency_ms=round(latency, 1),
            tokens=tokens,
            output_preview=output[:100],
            expected=expected,
            actual=actual,
        )

    def format_report(self, report: EvalReport) -> str:
        lines = [
            f"# Eval Report: {report.prompt_id} v{report.version}",
            f"Pass rate: {report.pass_rate:.1%} ({report.passed}/{report.total})",
            f"Avg latency: {report.avg_latency:.0f}ms | Total tokens: {report.total_tokens}",
        ]
        if report.thresholds_met:
            lines.append("\nThresholds:")
            for name, info in report.thresholds_met.items():
                status = "MET" if info["met"] else "FAILED"
                lines.append(f"  [{status}] {name}: {info['actual']} (threshold: {info['threshold']})")

        lines.append("\nResults:")
        for r in report.results:
            s = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{s}] Test {r.test_index} ({r.latency_ms:.0f}ms): {r.checks}")
            if not r.passed:
                lines.append(f"       Expected: {r.expected}")
                lines.append(f"       Actual:   {r.actual}")

        return "\n".join(lines)


# Test the harness
harness = EvalHarness()
sentiment_prompt = library.get("classify-sentiment")
if sentiment_prompt:
    report = harness.evaluate(sentiment_prompt)
    print(harness.format_report(report))
```

</details>

### 연습문제 3: 버전 관리 (마일스톤 3)

저장, 로드, 비교(Diff), 롤백(Rollback), 버전 이력을 포함한 버전 저장소(Version Store)를 구현합니다.

**요구사항:**
- JSON 직렬화(Serialization)를 사용한 파일 기반 버전 저장소
- 콘텐츠 해싱(Hashing)을 통한 새 버전 저장
- 특정 버전 또는 최신 버전 로드
- 임의의 두 버전 간 비교(Diff) (시스템 프롬프트의 통합 diff)
- 이전 버전으로의 롤백(Rollback)
- 메타데이터가 포함된 버전 목록

<details><summary>정답 보기</summary>

[섹션 4.1](#41-버전-저장소)에 정의된 `VersionStore` 클래스를 참조하세요. 전체 구현이 그곳에 제공되어 있습니다. 테스트하려면:

```python
store = VersionStore("/tmp/prompt_versions_demo")

# Save v1.0.0
v1 = PromptVersion.from_definition(
    PromptDefinition(
        id="classify-sentiment", name="Sentiment Classifier", version="1.0.0",
        category="classification", description="Classify sentiment",
        system_prompt="Classify text as positive or negative. Return JSON.",
        user_template="Classify: {{text}}",
        variables=[{"name": "text", "type": "string", "required": True}],
    ),
    change_description="Initial version",
)
store.save_version(v1)

# Save v1.1.0 with improvements
v2 = PromptVersion.from_definition(
    PromptDefinition(
        id="classify-sentiment", name="Sentiment Classifier", version="1.1.0",
        category="classification", description="Classify sentiment",
        system_prompt="Classify text as positive, negative, or neutral.\nReturn JSON: {\"sentiment\": \"...\", \"confidence\": 0.0-1.0}",
        user_template="Classify: {{text}}",
        variables=[{"name": "text", "type": "string", "required": True}],
    ),
    change_description="Added neutral category and confidence score",
)
store.save_version(v2)

# List versions
print("Versions:")
for v in store.list_versions("classify-sentiment"):
    print(f"  v{v['version']} [{v['hash']}] {v['change']}")

# Diff
print("\nDiff v1.0.0 → v1.1.0:")
diff = store.diff_versions("classify-sentiment", "1.0.0", "1.1.0")
print(diff["diff"])

# Rollback
store.rollback("classify-sentiment", "1.0.0")
latest = store.get_version("classify-sentiment")
print(f"\nAfter rollback, latest: v{latest.version}")
```

</details>

### 연습문제 4: 대화형 플레이그라운드 (마일스톤 4)

대화형 테스트, 변형(Variant) 비교, 온도(Temperature) 스윕(Sweep)을 지원하는 프롬프트 플레이그라운드를 구축합니다.

**요구사항:**
- 설정 가능한 모델, 온도(Temperature), max_tokens로 프롬프트 실행
- 동일 입력에 대한 여러 프롬프트 변형 비교
- 5개 값에 걸친 온도 스윕과 출력 비교
- 모든 플레이그라운드 실행 이력 추적
- 읽기 쉬운 테이블 형식으로 결과 포맷팅

<details><summary>정답 보기</summary>

[섹션 5.1](#51-플레이그라운드-엔진)에 정의된 `PromptPlayground` 클래스를 참조하세요. 전체 구현이 그곳에 제공되어 있습니다. 테스트하려면:

```python
playground = PromptPlayground()

# Test single run
result = playground.run(
    system_prompt="You are a helpful assistant. Be concise.",
    user_message="Explain recursion in one sentence.",
)
print(f"Output: {result['output']}")
print(f"Latency: {result['latency_ms']:.0f}ms, Tokens: {result['total_tokens']}")

# Compare variants
comparison = playground.compare(
    variants=[
        {"name": "strict_json", "system_prompt": "Return ONLY valid JSON. No other text. Classify sentiment: positive/negative/neutral."},
        {"name": "flexible", "system_prompt": "Classify the sentiment of the text. You may explain your reasoning, then provide the classification."},
    ],
    user_message="I had a mixed experience. The food was great but the service was slow.",
)
print("\nVariant Comparison:")
print(playground.format_comparison(comparison))

# Temperature sweep
sweep = playground.sweep_temperature(
    system_prompt="Write a one-sentence product tagline for a smart water bottle.",
    user_message="Create the tagline.",
    temperatures=[0.0, 0.5, 1.0],
)
print("\nTemperature Sweep:")
print(playground.format_comparison(sweep))

# History
print(f"\nTotal runs in history: {len(playground.history)}")
```

</details>

### 연습문제 5: 완전한 CLI와 통합 (마일스톤 5)

모든 것을 완전한 CLI 애플리케이션으로 연결합니다. 엔드투엔드(End-to-End) 워크플로우를 검증하는 통합 테스트를 추가합니다.

**요구사항:**
- 다음 명령어를 포함한 CLI: list, show, eval, test, versions, diff, rollback, catalog
- 다음을 커버하는 통합 테스트: 프롬프트 추가 → 평가 → 버전 관리 → 업데이트 → 평가 → 비교
- 누락된 프롬프트, 잘못된 YAML, 실패한 평가에 대한 에러 처리
- 마크다운(Markdown)으로 카탈로그 내보내기
- 라이브러리에 최소 5개의 프롬프트

<details><summary>정답 보기</summary>

[섹션 8.1](#81-cli-인터페이스)에 정의된 `PromptLibraryCLI`와 `build_cli` 함수를 참조하세요. 전체 구현이 그곳에 제공되어 있습니다. 아래는 엔드투엔드 워크플로우를 검증하는 통합 테스트입니다:

```python
def run_integration_tests():
    """End-to-end integration tests for the prompt library."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        lib_dir = os.path.join(tmpdir, "prompts")
        ver_dir = os.path.join(tmpdir, "versions")
        library = PromptLibrary(lib_dir)
        version_store = VersionStore(ver_dir)
        harness = EvalHarness()
        playground = PromptPlayground()
        passed = 0
        failed = 0

        # Test 1: Add a prompt
        prompt = PromptDefinition(
            id="test-classifier", name="Test Classifier", version="1.0.0",
            category="classification", description="Test prompt",
            system_prompt='Classify as A or B. Return JSON: {"class": "A|B"}',
            user_template="Classify: {{text}}",
            variables=[{"name": "text", "type": "string", "required": True}],
            tags=["test"],
            evaluation={"test_cases": [
                {"input": {"text": "positive"}, "expected": {"class": "A"}},
            ]},
        )
        errors = library.add(prompt)
        if not errors:
            print("  [PASS] Test 1: Add prompt")
            passed += 1
        else:
            print(f"  [FAIL] Test 1: {errors}")
            failed += 1

        # Test 2: Retrieve prompt
        retrieved = library.get("test-classifier")
        if retrieved and retrieved.name == "Test Classifier":
            print("  [PASS] Test 2: Get prompt")
            passed += 1
        else:
            print("  [FAIL] Test 2: Prompt not found")
            failed += 1

        # Test 3: Search by tag
        results = library.search(tags=["test"])
        if len(results) == 1:
            print("  [PASS] Test 3: Search by tag")
            passed += 1
        else:
            print(f"  [FAIL] Test 3: Expected 1, got {len(results)}")
            failed += 1

        # Test 4: Version management
        v1 = PromptVersion.from_definition(prompt, "Initial")
        version_store.save_version(v1)
        loaded = version_store.get_version("test-classifier")
        if loaded and loaded.version == "1.0.0":
            print("  [PASS] Test 4: Save and load version")
            passed += 1
        else:
            print("  [FAIL] Test 4: Version not loaded")
            failed += 1

        # Test 5: Add second version and diff
        prompt_v2 = PromptDefinition(
            id="test-classifier", name="Test Classifier", version="1.1.0",
            category="classification", description="Test prompt v2",
            system_prompt='Classify as A, B, or C. Return JSON: {"class": "A|B|C"}',
            user_template="Classify: {{text}}",
            variables=[{"name": "text", "type": "string", "required": True}],
            tags=["test"],
        )
        v2 = PromptVersion.from_definition(prompt_v2, "Added class C")
        version_store.save_version(v2)
        versions = version_store.list_versions("test-classifier")
        if len(versions) == 2:
            print("  [PASS] Test 5: Multiple versions")
            passed += 1
        else:
            print(f"  [FAIL] Test 5: Expected 2 versions, got {len(versions)}")
            failed += 1

        # Test 6: Diff
        diff_result = version_store.diff_versions("test-classifier", "1.0.0", "1.1.0")
        if not diff_result.get("same_content"):
            print("  [PASS] Test 6: Diff detects changes")
            passed += 1
        else:
            print("  [FAIL] Test 6: Diff shows no changes")
            failed += 1

        # Test 7: Rollback
        version_store.rollback("test-classifier", "1.0.0")
        latest = version_store.get_version("test-classifier")
        if latest.version == "1.0.0":
            print("  [PASS] Test 7: Rollback")
            passed += 1
        else:
            print(f"  [FAIL] Test 7: Expected 1.0.0, got {latest.version}")
            failed += 1

        # Test 8: Remove prompt
        removed = library.remove("test-classifier")
        if removed and library.get("test-classifier") is None:
            print("  [PASS] Test 8: Remove prompt")
            passed += 1
        else:
            print("  [FAIL] Test 8: Remove failed")
            failed += 1

        # Test 9: Validation rejects bad prompts
        bad_prompt = PromptDefinition(
            id="", name="", version="bad",
            category="", description="",
            system_prompt="", user_template="{{undefined_var}}",
        )
        errors = bad_prompt.validate()
        if len(errors) >= 3:
            print(f"  [PASS] Test 9: Validation caught {len(errors)} errors")
            passed += 1
        else:
            print(f"  [FAIL] Test 9: Expected >=3 errors, got {len(errors)}")
            failed += 1

        # Test 10: Catalog generation
        for seed in SEED_PROMPTS:
            library.add(seed)
        doc_gen = PromptDocGenerator(library)
        catalog = doc_gen.generate_catalog()
        if "classification" in catalog.lower() and "generation" in catalog.lower():
            print("  [PASS] Test 10: Catalog generation")
            passed += 1
        else:
            print("  [FAIL] Test 10: Catalog missing categories")
            failed += 1

        print(f"\n{'=' * 50}")
        print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
        print(f"{'=' * 50}")


run_integration_tests()
```

</details>

---

**이전**: [16. 에이전트 프롬프팅 패턴](./16_Agent_Prompting_Patterns.md) | **다음**: [00. 개요](./00_Overview.md)

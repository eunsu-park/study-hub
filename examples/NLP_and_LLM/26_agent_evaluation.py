"""
26. Agent Evaluation and Benchmarks Example

Evaluation frameworks, failure detection, cost tracking, and custom evals.
"""

from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Callable
import time
import json
import hashlib

print("=" * 60)
print("Agent Evaluation and Benchmarks")
print("=" * 60)


# ============================================
# 1. Evaluation Dimensions
# ============================================
print("\n[1] Evaluation Dimensions")
print("-" * 40)


@dataclass
class EvalResult:
    """Result of evaluating a single task."""
    task_id: str
    correct: bool
    steps: int
    tokens: int
    cost_usd: float
    latency_ms: float


results = [
    EvalResult("t1", True, 3, 500, 0.01, 1200),
    EvalResult("t2", True, 5, 800, 0.02, 2500),
    EvalResult("t3", False, 8, 1200, 0.03, 4000),
    EvalResult("t4", True, 2, 300, 0.005, 800),
    EvalResult("t5", True, 4, 600, 0.015, 1800),
]

success_rate = sum(1 for r in results if r.correct) / len(results)
avg_steps = sum(r.steps for r in results) / len(results)
avg_cost = sum(r.cost_usd for r in results) / len(results)
avg_latency = sum(r.latency_ms for r in results) / len(results)

print(f"Success rate:  {success_rate:.0%}")
print(f"Avg steps:     {avg_steps:.1f}")
print(f"Avg cost:      ${avg_cost:.3f}")
print(f"Avg latency:   {avg_latency:.0f}ms")


# ============================================
# 2. Failure Mode Detector
# ============================================
print("\n[2] Failure Mode Detection")
print("-" * 40)


class FailureDetector:
    """Detect common agent failure modes."""

    def __init__(self, max_repeats: int = 3):
        self.max_repeats = max_repeats

    def detect_loop(self, actions: list[str]) -> bool:
        counts = Counter(actions)
        return any(c >= self.max_repeats for c in counts.values())

    def detect_error_cascade(self, statuses: list[str]) -> bool:
        consecutive = 0
        for s in statuses:
            consecutive = consecutive + 1 if s == "error" else 0
            if consecutive >= 3:
                return True
        return False


detector = FailureDetector()

# Test loop detection
actions_good = ["search", "analyze", "summarize", "respond"]
actions_loop = ["search", "search", "search", "search"]
print(f"Loop in good trajectory: {detector.detect_loop(actions_good)}")
print(f"Loop in bad trajectory:  {detector.detect_loop(actions_loop)}")

# Test error cascade
statuses_ok = ["ok", "error", "ok", "ok"]
statuses_bad = ["ok", "error", "error", "error", "ok"]
print(f"Cascade in ok trajectory:  {detector.detect_error_cascade(statuses_ok)}")
print(f"Cascade in bad trajectory: {detector.detect_error_cascade(statuses_bad)}")


# ============================================
# 3. Trajectory Scorer
# ============================================
print("\n[3] Trajectory Scoring")
print("-" * 40)


def score_trajectory(actual: list[str], reference: list[str]) -> dict:
    """Score a trajectory against an optimal reference."""
    # Action overlap (Jaccard)
    actual_set = set(actual)
    ref_set = set(reference)
    union = actual_set | ref_set
    overlap = len(actual_set & ref_set) / len(union) if union else 0

    # Redundancy penalty
    counts = Counter(actual)
    redundant = sum(c - 1 for c in counts.values())
    redundancy = redundant / len(actual) if actual else 0

    overall = overlap * 0.6 + (1 - redundancy) * 0.4
    return {
        "overlap": round(overlap, 3),
        "redundancy": round(redundancy, 3),
        "overall": round(overall, 3),
    }


ref = ["search", "filter", "compare", "select"]
good = ["search", "filter", "compare", "select"]
bad = ["search", "search", "search", "browse", "filter"]

print(f"Good trajectory: {score_trajectory(good, ref)}")
print(f"Bad trajectory:  {score_trajectory(bad, ref)}")


# ============================================
# 4. Cost-Quality Tracker
# ============================================
print("\n[4] Cost-Quality Analysis")
print("-" * 40)


class CostQualityTracker:
    """Track cost vs quality across models."""

    PRICING = {
        "claude-sonnet": (3.0, 15.0),
        "claude-haiku": (0.25, 1.25),
        "gpt-4o-mini": (0.15, 0.60),
    }

    def __init__(self):
        self.records: list[dict] = []

    def record(self, model: str, input_tokens: int,
               output_tokens: int, quality: float):
        inp_rate, out_rate = self.PRICING.get(model, (5.0, 15.0))
        cost = input_tokens * inp_rate / 1e6 + output_tokens * out_rate / 1e6
        self.records.append({"model": model, "cost": cost, "quality": quality})

    def report(self) -> dict:
        by_model = defaultdict(lambda: {"costs": [], "qualities": []})
        for r in self.records:
            by_model[r["model"]]["costs"].append(r["cost"])
            by_model[r["model"]]["qualities"].append(r["quality"])

        summary = {}
        for model, data in by_model.items():
            avg_cost = sum(data["costs"]) / len(data["costs"])
            avg_quality = sum(data["qualities"]) / len(data["qualities"])
            summary[model] = {
                "avg_cost": round(avg_cost, 5),
                "avg_quality": round(avg_quality, 3),
                "quality_per_dollar": round(avg_quality / max(avg_cost, 1e-8), 1),
            }
        return summary


tracker = CostQualityTracker()
tracker.record("claude-sonnet", 2000, 800, 0.92)
tracker.record("claude-sonnet", 1800, 700, 0.88)
tracker.record("claude-haiku", 1500, 600, 0.72)
tracker.record("claude-haiku", 1200, 500, 0.68)
tracker.record("gpt-4o-mini", 1000, 400, 0.70)

report = tracker.report()
for model, stats in report.items():
    print(f"  {model:15s}: cost=${stats['avg_cost']:.5f}, "
          f"quality={stats['avg_quality']}, q/$={stats['quality_per_dollar']}")


# ============================================
# 5. Custom Eval Suite
# ============================================
print("\n[5] Custom Eval Suite")
print("-" * 40)


class EvalSuite:
    """Minimal eval suite for agent testing."""

    def __init__(self, name: str):
        self.name = name
        self.cases: list[dict] = []

    def add_case(self, case_id: str, expected: str, tags: list[str] = None):
        self.cases.append({"id": case_id, "expected": expected, "tags": tags or []})

    def run(self, agent_fn: Callable[[str], str]) -> dict:
        correct = 0
        for case in self.cases:
            result = agent_fn(case["id"])
            if case["expected"].lower() in result.lower():
                correct += 1

        return {
            "suite": self.name,
            "total": len(self.cases),
            "correct": correct,
            "accuracy": round(correct / max(len(self.cases), 1), 3),
        }


suite = EvalSuite("math-agent-eval")
suite.add_case("2+2", "4", ["easy"])
suite.add_case("10*5", "50", ["easy"])
suite.add_case("sqrt(144)", "12", ["medium"])

# Mock agent
def mock_agent(query: str) -> str:
    answers = {"2+2": "4", "10*5": "50", "sqrt(144)": "12"}
    return answers.get(query, "unknown")

report = suite.run(mock_agent)
print(f"Suite: {report['suite']}")
print(f"Accuracy: {report['accuracy']:.0%} ({report['correct']}/{report['total']})")


# ============================================
# 6. Agent Tracer
# ============================================
print("\n[6] Agent Tracing")
print("-" * 40)


class SimpleTracer:
    """Trace agent execution steps."""

    def __init__(self):
        self.spans: list[dict] = []

    def trace(self, operation: str, duration_ms: float, **attrs):
        self.spans.append({
            "operation": operation,
            "duration_ms": round(duration_ms, 1),
            **attrs,
        })

    def summary(self) -> dict:
        total = sum(s["duration_ms"] for s in self.spans)
        return {
            "spans": len(self.spans),
            "total_ms": round(total, 1),
            "operations": [s["operation"] for s in self.spans],
        }


tracer = SimpleTracer()
tracer.trace("plan", 50.0, strategy="decompose")
tracer.trace("tool_call", 200.0, tool="search")
tracer.trace("generate", 150.0, model="claude-haiku")

summary = tracer.summary()
print(f"Spans: {summary['spans']}, Total: {summary['total_ms']}ms")
print(f"Operations: {summary['operations']}")


print("\n" + "=" * 60)
print("Agent Evaluation and Benchmarks example complete!")
print("=" * 60)

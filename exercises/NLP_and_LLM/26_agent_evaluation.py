"""
Exercises for Lesson 26: Agent Evaluation and Benchmarks
Topic: NLP_and_LLM

Practice problems for evaluation frameworks, failure detection, and benchmarking.
"""

import time
import math
from dataclasses import dataclass, field
from typing import Callable
from collections import Counter, defaultdict


# === Exercise 1: Trajectory Quality Scorer ===
# Problem: Score an agent trajectory against a reference using
# action overlap, ordering (LCS), and redundancy penalty.

def exercise_1():
    """Score trajectories against optimal references."""
    print("=" * 60)
    print("Exercise 1: Trajectory Quality Scorer")
    print("=" * 60)

    class TrajectoryScorer:
        # TODO: Compute Jaccard similarity of action sets
        def action_overlap(self, actual: list[str], ref: list[str]) -> float:
            a, r = set(actual), set(ref)
            union = a | r
            return len(a & r) / len(union) if union else 0.0

        # TODO: Compute ordering score using LCS
        def ordering_score(self, actual: list[str], ref: list[str]) -> float:
            if not ref or not actual:
                return 0.0
            m, n = len(actual), len(ref)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if actual[i-1] == ref[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n] / len(ref)

        # TODO: Compute redundancy penalty
        def redundancy(self, actual: list[str]) -> float:
            if len(actual) <= 1:
                return 0.0
            counts = Counter(actual)
            redundant = sum(c - 1 for c in counts.values())
            return redundant / len(actual)

        def score(self, actual: list[str], ref: list[str]) -> dict:
            overlap = self.action_overlap(actual, ref)
            ordering = self.ordering_score(actual, ref)
            redund = self.redundancy(actual)
            overall = overlap * 0.4 + ordering * 0.3 + (1.0 - redund) * 0.3
            return {
                "overlap": round(overlap, 3),
                "ordering": round(ordering, 3),
                "redundancy": round(redund, 3),
                "overall": round(overall, 3),
            }

    scorer = TrajectoryScorer()
    ref = ["search", "filter", "compare", "select", "confirm"]

    trajectories = {
        "optimal": ["search", "filter", "compare", "select", "confirm"],
        "okay": ["search", "search", "filter", "select", "compare", "confirm"],
        "bad": ["search", "search", "search", "browse", "filter"],
    }

    for name, traj in trajectories.items():
        result = scorer.score(traj, ref)
        print(f"  {name:8s}: {result}")


# === Exercise 2: Circuit Breaker ===
# Problem: Implement a circuit breaker that detects action loops,
# observation loops, and cost runaway, then halts the agent.

def exercise_2():
    """Circuit breaker for agent safety."""
    print("\n" + "=" * 60)
    print("Exercise 2: Circuit Breaker")
    print("=" * 60)

    class CircuitBreaker:
        def __init__(self, max_repeats: int = 3, max_cost: float = 1.0):
            self.max_repeats = max_repeats
            self.max_cost = max_cost
            self.actions: list[str] = []
            self.total_cost: float = 0.0
            self.tripped: bool = False
            self.reason: str = ""

        def record(self, action: str, cost: float = 0.01):
            self.actions.append(action)
            self.total_cost += cost

        # TODO: Check all conditions and trip if violated
        def check(self) -> bool:
            if self.tripped:
                return True

            # Action loop
            if len(self.actions) >= self.max_repeats:
                recent = self.actions[-self.max_repeats:]
                if len(set(recent)) == 1:
                    self.tripped = True
                    self.reason = f"action_loop: '{recent[0]}' repeated {self.max_repeats}x"
                    return True

            # Cost exceeded
            if self.total_cost >= self.max_cost:
                self.tripped = True
                self.reason = f"cost_exceeded: ${self.total_cost:.2f} >= ${self.max_cost:.2f}"
                return True

            return False

    cb = CircuitBreaker(max_repeats=3, max_cost=0.50)

    actions = ["search", "analyze", "search", "search", "search"]
    for action in actions:
        cb.record(action, 0.05)
        if cb.check():
            print(f"  TRIPPED after '{action}': {cb.reason}")
            break
    else:
        print("  No trip")

    # Test cost limit
    cb2 = CircuitBreaker(max_repeats=10, max_cost=0.10)
    for i in range(20):
        cb2.record(f"action_{i}", 0.02)
        if cb2.check():
            print(f"  TRIPPED at step {i}: {cb2.reason}")
            break


# === Exercise 3: Multi-Model Cost Optimizer ===
# Problem: Given eval results from multiple models, recommend the
# cheapest model that meets a quality threshold.

def exercise_3():
    """Cost optimizer for model selection."""
    print("\n" + "=" * 60)
    print("Exercise 3: Cost Optimizer")
    print("=" * 60)

    @dataclass
    class ModelStats:
        name: str
        quality: float
        cost_per_task: float
        success_rate: float

    class CostOptimizer:
        def __init__(self, min_quality: float = 0.7, min_success: float = 0.8):
            self.min_quality = min_quality
            self.min_success = min_success
            self.models: list[ModelStats] = []

        def add(self, model: ModelStats):
            self.models.append(model)

        # TODO: Recommend cheapest model meeting thresholds
        def recommend(self) -> dict:
            eligible = [
                m for m in self.models
                if m.quality >= self.min_quality and m.success_rate >= self.min_success
            ]
            if not eligible:
                return {"recommendation": None, "reason": "No model meets thresholds"}
            best = min(eligible, key=lambda m: m.cost_per_task)
            return {
                "recommendation": best.name,
                "cost": round(best.cost_per_task, 4),
                "quality": best.quality,
                "success_rate": best.success_rate,
            }

    opt = CostOptimizer(min_quality=0.7, min_success=0.8)
    opt.add(ModelStats("claude-sonnet", 0.92, 0.045, 0.95))
    opt.add(ModelStats("claude-haiku", 0.75, 0.003, 0.85))
    opt.add(ModelStats("gpt-4o", 0.90, 0.035, 0.93))
    opt.add(ModelStats("gpt-4o-mini", 0.72, 0.002, 0.82))

    rec = opt.recommend()
    print(f"Recommended: {rec['recommendation']}")
    print(f"Cost/task: ${rec.get('cost', 'N/A')}")
    print(f"Quality: {rec.get('quality', 'N/A')}")


# === Exercise 4: Benchmark Report Generator ===
# Problem: Aggregate results from multiple benchmarks and compute
# per-model rankings with confidence intervals.

def exercise_4():
    """Benchmark comparison report."""
    print("\n" + "=" * 60)
    print("Exercise 4: Benchmark Report")
    print("=" * 60)

    class BenchmarkReporter:
        def __init__(self):
            self.results: dict[str, list[dict]] = defaultdict(list)

        def add(self, benchmark: str, model: str, score: float):
            self.results[benchmark].append({"model": model, "score": score})

        # TODO: Generate ranking report
        def report(self) -> dict:
            output = {}
            for bench, entries in self.results.items():
                by_model = defaultdict(list)
                for e in entries:
                    by_model[e["model"]].append(e["score"])

                model_stats = {}
                for model, scores in by_model.items():
                    mean = sum(scores) / len(scores)
                    model_stats[model] = round(mean, 3)

                ranking = sorted(model_stats.items(), key=lambda x: -x[1])
                output[bench] = {
                    "ranking": [m for m, _ in ranking],
                    "scores": model_stats,
                }
            return output

    reporter = BenchmarkReporter()
    import random
    random.seed(42)

    for _ in range(5):
        reporter.add("SWE-bench", "sonnet", 0.45 + random.uniform(-0.03, 0.03))
        reporter.add("SWE-bench", "haiku", 0.28 + random.uniform(-0.03, 0.03))
        reporter.add("GAIA", "sonnet", 0.75 + random.uniform(-0.05, 0.05))
        reporter.add("GAIA", "haiku", 0.55 + random.uniform(-0.05, 0.05))

    report = reporter.report()
    for bench, data in report.items():
        print(f"  {bench}: ranking={data['ranking']}, scores={data['scores']}")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()

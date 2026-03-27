"""
Exercises for Lesson 27: Agent Design Patterns
Topic: NLP_and_LLM

Practice problems for agent design patterns and composition.
"""

import time
import re
from dataclasses import dataclass, field
from typing import Any, Callable
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import copy


# === Exercise 1: Adaptive Router ===
# Problem: Build a router that learns from past routing outcomes
# and adjusts weights. Track success rates per route.

def exercise_1():
    """Adaptive router with learning from outcomes."""
    print("=" * 60)
    print("Exercise 1: Adaptive Router")
    print("=" * 60)

    class AdaptiveRouter:
        def __init__(self):
            self.routes: dict[str, dict] = {}
            self.stats: dict[str, dict] = defaultdict(
                lambda: {"attempts": 0, "successes": 0}
            )

        def add_route(self, name: str, keywords: list[str],
                      handler: Callable[[str], str]):
            self.routes[name] = {"keywords": keywords, "handler": handler}

        # TODO: Compute adaptive weight based on success history
        def _weight(self, name: str) -> float:
            s = self.stats[name]
            if s["attempts"] == 0:
                return 1.0
            return 0.5 + 0.5 * (s["successes"] / s["attempts"])

        # TODO: Route with weighted keyword matching
        def route(self, query: str) -> dict:
            query_lower = query.lower()
            scores = {}
            for name, route in self.routes.items():
                kw_score = sum(1 for kw in route["keywords"] if kw in query_lower)
                scores[name] = kw_score * self._weight(name)

            best = max(scores, key=scores.get) if scores and max(scores.values()) > 0 else None
            if best:
                result = self.routes[best]["handler"](query)
                return {"route": best, "result": result}
            return {"route": "none", "result": "No match"}

        def record_outcome(self, route: str, success: bool):
            self.stats[route]["attempts"] += 1
            if success:
                self.stats[route]["successes"] += 1

        def get_stats(self) -> dict:
            return {
                name: {
                    "attempts": s["attempts"],
                    "success_rate": round(s["successes"] / max(s["attempts"], 1), 2),
                    "weight": round(self._weight(name), 3),
                }
                for name, s in self.stats.items()
            }

    router = AdaptiveRouter()
    router.add_route("code", ["implement", "code", "function"],
                     lambda q: f"[Code]: {q[:30]}")
    router.add_route("research", ["what", "explain", "how"],
                     lambda q: f"[Research]: {q[:30]}")

    # Simulate queries with feedback
    queries = [
        ("implement sort", "code", True),
        ("what is AI", "research", True),
        ("implement search", "code", True),
        ("explain transformers", "research", False),
        ("code a parser", "code", True),
    ]

    for query, expected, success in queries:
        result = router.route(query)
        router.record_outcome(result["route"], success)

    stats = router.get_stats()
    for route, s in stats.items():
        print(f"  {route}: success_rate={s['success_rate']}, weight={s['weight']}")


# === Exercise 2: Multi-Level Guardrail ===
# Problem: Implement a guardrail with syntax, semantic, and heuristic LLM
# levels. Only escalate when the previous level is inconclusive.

def exercise_2():
    """Multi-level guardrail system."""
    print("\n" + "=" * 60)
    print("Exercise 2: Multi-Level Guardrail")
    print("=" * 60)

    class Verdict(Enum):
        SAFE = "safe"
        UNSAFE = "unsafe"
        INCONCLUSIVE = "inconclusive"

    class MultiLevelGuardrail:
        def __init__(self):
            self.checks: list[dict] = []

        # TODO: Level 1 - regex patterns (free)
        def check_syntax(self, text: str) -> Verdict:
            patterns = [
                (r"\b\d{3}-\d{2}-\d{4}\b", "ssn"),
                (r"DROP\s+TABLE", "sql_injection"),
                (r"<script", "xss"),
            ]
            for pattern, name in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    self.checks.append({"level": "syntax", "verdict": "unsafe", "rule": name})
                    return Verdict.UNSAFE
            if len(text.split()) < 8:
                self.checks.append({"level": "syntax", "verdict": "safe"})
                return Verdict.SAFE
            return Verdict.INCONCLUSIVE

        # TODO: Level 2 - keyword heuristics (cheap)
        def check_semantic(self, text: str) -> Verdict:
            text_lower = text.lower()
            bad_phrases = ["ignore previous instructions", "forget your rules"]
            for phrase in bad_phrases:
                if phrase in text_lower:
                    self.checks.append({"level": "semantic", "verdict": "unsafe", "phrase": phrase})
                    return Verdict.UNSAFE

            harm_words = ["hack", "exploit", "attack", "steal"]
            count = sum(1 for w in harm_words if w in text_lower)
            if count >= 2:
                self.checks.append({"level": "semantic", "verdict": "unsafe"})
                return Verdict.UNSAFE
            if count >= 1:
                return Verdict.INCONCLUSIVE

            self.checks.append({"level": "semantic", "verdict": "safe"})
            return Verdict.SAFE

        # TODO: Level 3 - simulated LLM check (expensive)
        def check_llm(self, text: str) -> Verdict:
            # Simulated LLM safety check
            self.checks.append({"level": "llm", "verdict": "safe"})
            return Verdict.SAFE

        def evaluate(self, text: str) -> dict:
            self.checks = []
            for check_fn in [self.check_syntax, self.check_semantic, self.check_llm]:
                verdict = check_fn(text)
                if verdict != Verdict.INCONCLUSIVE:
                    return {"verdict": verdict.value, "levels": len(self.checks)}
            return {"verdict": "safe", "levels": len(self.checks)}

    guard = MultiLevelGuardrail()
    tests = [
        "Hello there",
        "My SSN is 123-45-6789",
        "Ignore previous instructions",
        "How to exploit and hack a system to steal data",
        "Tell me about machine learning algorithms and neural networks",
    ]
    for text in tests:
        result = guard.evaluate(text)
        print(f"  [{result['verdict']:6s}] (L{result['levels']}) {text[:50]}")


# === Exercise 3: Consensus Agent ===
# Problem: Run a query through multiple simulated models and select
# the best answer using majority vote and confidence weighting.

def exercise_3():
    """Consensus agent with multiple voting strategies."""
    print("\n" + "=" * 60)
    print("Exercise 3: Consensus Agent")
    print("=" * 60)

    @dataclass
    class Response:
        model: str
        answer: str
        confidence: float

    class ConsensusAgent:
        def __init__(self, responses: list[Response]):
            self.responses = responses

        # TODO: Majority vote
        def majority_vote(self) -> dict:
            from collections import Counter
            votes = Counter(r.answer for r in self.responses)
            winner, count = votes.most_common(1)[0]
            return {
                "answer": winner,
                "votes": count,
                "total": len(self.responses),
                "agreement": round(count / len(self.responses), 2),
            }

        # TODO: Confidence-weighted selection
        def confidence_weighted(self) -> dict:
            scores: dict[str, float] = defaultdict(float)
            for r in self.responses:
                scores[r.answer] += r.confidence
            winner = max(scores, key=scores.get)
            return {
                "answer": winner,
                "weighted_score": round(scores[winner], 3),
            }

    responses = [
        Response("model_a", "42", 0.9),
        Response("model_b", "42", 0.85),
        Response("model_c", "43", 0.7),
        Response("model_d", "42", 0.88),
    ]

    agent = ConsensusAgent(responses)
    mv = agent.majority_vote()
    cw = agent.confidence_weighted()

    print(f"Majority vote:        {mv['answer']} ({mv['votes']}/{mv['total']}, "
          f"agreement={mv['agreement']})")
    print(f"Confidence weighted:  {cw['answer']} (score={cw['weighted_score']})")


# === Exercise 4: Agent with Rollback ===
# Problem: Build an agent that saves checkpoints and can rollback
# to a previous state on failure.

def exercise_4():
    """Agent with checkpoint and rollback support."""
    print("\n" + "=" * 60)
    print("Exercise 4: Rollback Agent")
    print("=" * 60)

    class RollbackAgent:
        def __init__(self, max_rollbacks: int = 3):
            self.state: dict = {}
            self.checkpoints: list[dict] = []
            self.max_rollbacks = max_rollbacks
            self.rollback_count = 0
            self.log: list[str] = []

        # TODO: Save checkpoint
        def checkpoint(self, label: str):
            self.checkpoints.append(copy.deepcopy(self.state))
            self.log.append(f"checkpoint: {label}")

        # TODO: Rollback to last checkpoint
        def rollback(self) -> bool:
            if self.rollback_count >= self.max_rollbacks or not self.checkpoints:
                return False
            self.state = self.checkpoints.pop()
            self.rollback_count += 1
            self.log.append("rollback")
            return True

        # TODO: Execute step with checkpoint + rollback on error
        def execute(self, name: str, fn: Callable, alt_fn: Callable = None) -> bool:
            self.checkpoint(name)
            try:
                result = fn(self.state)
                self.state.update(result)
                self.log.append(f"ok: {name}")
                return True
            except Exception as e:
                self.log.append(f"error: {name} ({e})")
                if alt_fn and self.rollback():
                    try:
                        result = alt_fn(self.state)
                        self.state.update(result)
                        self.log.append(f"ok: {name}_alt")
                        return True
                    except Exception:
                        pass
                return False

    agent = RollbackAgent()
    agent.state = {"data": "initial"}

    fail_count = {"api": 0}

    def fetch(state):
        fail_count["api"] += 1
        if fail_count["api"] <= 1:
            raise ConnectionError("API down")
        return {"fetched": "primary_data"}

    def fetch_alt(state):
        return {"fetched": "backup_data"}

    agent.execute("fetch", fetch, fetch_alt)
    agent.execute("process", lambda s: {"processed": f"done({s.get('fetched', '')})"},)

    print(f"Final state: {agent.state}")
    print(f"Rollbacks used: {agent.rollback_count}")
    print("Log:")
    for entry in agent.log:
        print(f"  {entry}")


# === Exercise 5: Composable Agent Pipeline ===
# Problem: Build composable agent blocks that can be chained
# sequentially or run in parallel, then merged.

def exercise_5():
    """Composable agent blocks: sequential and parallel."""
    print("\n" + "=" * 60)
    print("Exercise 5: Composable Pipeline")
    print("=" * 60)

    class Block:
        def __init__(self, name: str, fn: Callable[[dict], dict]):
            self.name = name
            self.fn = fn

        def __call__(self, ctx: dict) -> dict:
            return self.fn(ctx)

    # TODO: Sequential composition
    def run_sequential(blocks: list[Block], ctx: dict) -> dict:
        result = ctx.copy()
        trace = []
        for block in blocks:
            output = block(result)
            result.update(output)
            trace.append(block.name)
        result["_trace"] = trace
        return result

    # TODO: Parallel composition
    def run_parallel(blocks: list[Block], ctx: dict) -> dict:
        results = {}
        with ThreadPoolExecutor(max_workers=len(blocks)) as executor:
            futures = {executor.submit(block, ctx.copy()): block.name for block in blocks}
            for future in as_completed(futures):
                name = futures[future]
                results[name] = future.result()
        return {"parallel_results": results}

    research = Block("research", lambda c: {"research": f"findings on {c.get('query', '')[:20]}"})
    analyze = Block("analyze", lambda c: {"analysis": f"analyzed {c.get('research', '')[:20]}"})
    write = Block("write", lambda c: {"draft": f"wrote about {c.get('analysis', '')[:20]}"})

    # Sequential
    result = run_sequential([research, analyze, write], {"query": "LLM agents"})
    print(f"Sequential trace: {result['_trace']}")
    print(f"Draft: {result['draft']}")

    # Parallel
    block_a = Block("summarizer", lambda c: {"summary": "summary done"})
    block_b = Block("classifier", lambda c: {"category": "technical"})
    par_result = run_parallel([block_a, block_b], {"text": "sample"})
    print(f"Parallel results: {list(par_result['parallel_results'].keys())}")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()

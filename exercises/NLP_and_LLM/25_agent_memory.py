"""
Exercises for Lesson 25: Agent Memory and Planning
Topic: NLP_and_LLM

Practice problems for memory architectures, planning, and reflection.
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from collections import deque


# === Exercise 1: Tiered Memory Store ===
# Problem: Build a three-tier memory system (hot/warm/cold) with
# cascading overflow and cross-tier search.

def exercise_1():
    """Three-tier memory with hot, warm, and cold stores."""
    print("=" * 60)
    print("Exercise 1: Tiered Memory Store")
    print("=" * 60)

    class TieredMemory:
        def __init__(self, hot_size: int = 5, warm_size: int = 20):
            self.hot: deque[dict] = deque(maxlen=hot_size)
            self.warm: deque[dict] = deque(maxlen=warm_size)
            self.cold: list[dict] = []

        # TODO: Implement add with cascading overflow (hot -> warm -> cold)
        def add(self, role: str, content: str):
            msg = {"role": role, "content": content, "ts": time.time()}
            if len(self.hot) == self.hot.maxlen:
                overflow = self.hot.popleft()
                if len(self.warm) == self.warm.maxlen:
                    self.cold.append(self.warm.popleft())
                self.warm.append(overflow)
            self.hot.append(msg)

        # TODO: Implement keyword search across all tiers
        def search(self, query: str, top_k: int = 3) -> list[dict]:
            query_terms = set(query.lower().split())
            results = []
            for tier in [self.hot, self.warm, self.cold]:
                for msg in tier:
                    overlap = len(query_terms & set(msg["content"].lower().split()))
                    if overlap > 0:
                        results.append({"msg": msg, "score": overlap})
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        def get_hot(self) -> list[dict]:
            return [{"role": m["role"], "content": m["content"]} for m in self.hot]

    mem = TieredMemory(hot_size=3, warm_size=5)
    for i in range(12):
        topic = "AI" if i % 2 == 0 else "databases"
        mem.add("user", f"Message {i} about {topic}")

    print(f"Hot: {len(mem.hot)}, Warm: {len(mem.warm)}, Cold: {len(mem.cold)}")

    results = mem.search("AI topic")
    print(f"Search 'AI topic': {len(results)} results")
    for r in results:
        print(f"  [{r['score']}] {r['msg']['content'][:50]}")


# === Exercise 2: Dependency-Aware Planner ===
# Problem: Build a planner that resolves task dependencies and returns
# parallelizable execution waves. Detect circular dependencies.

def exercise_2():
    """Dependency-aware task planner with wave scheduling."""
    print("\n" + "=" * 60)
    print("Exercise 2: Dependency Planner")
    print("=" * 60)

    class DependencyPlanner:
        def __init__(self):
            self.tasks: dict[str, dict] = {}

        def add_task(self, task_id: str, desc: str, deps: list[str] = None):
            self.tasks[task_id] = {"desc": desc, "deps": deps or []}

        # TODO: Implement topological sort returning execution waves
        def plan_waves(self) -> list[list[str]]:
            in_degree = {tid: 0 for tid in self.tasks}
            graph = {tid: [] for tid in self.tasks}

            for tid, task in self.tasks.items():
                for dep in task["deps"]:
                    graph[dep].append(tid)
                    in_degree[tid] += 1

            waves = []
            remaining = dict(in_degree)

            while remaining:
                ready = [t for t in remaining if remaining[t] == 0]
                if not ready:
                    raise ValueError(f"Circular dependency: {list(remaining.keys())}")
                waves.append(sorted(ready))
                for t in ready:
                    for dep in graph[t]:
                        if dep in remaining:
                            remaining[dep] -= 1
                    del remaining[t]

            return waves

    planner = DependencyPlanner()
    planner.add_task("gather", "Gather requirements")
    planner.add_task("api", "Design API", ["gather"])
    planner.add_task("db", "Design DB", ["gather"])
    planner.add_task("impl", "Implement", ["api", "db"])
    planner.add_task("test", "Testing", ["impl"])

    waves = planner.plan_waves()
    for i, wave in enumerate(waves):
        print(f"  Wave {i+1}: {wave}")


# === Exercise 3: Importance-Based Summary Memory ===
# Problem: Build a memory that keeps high-importance messages verbatim
# and summarizes low-importance ones.

def exercise_3():
    """Importance-scored summary memory."""
    print("\n" + "=" * 60)
    print("Exercise 3: Importance-Based Summary Memory")
    print("=" * 60)

    class ImportanceMemory:
        def __init__(self, max_verbatim: int = 5):
            self.max_verbatim = max_verbatim
            self.verbatim: list[dict] = []
            self.summary_buffer: list[str] = []
            self.discarded: int = 0

        # TODO: Score importance using heuristics
        def score(self, content: str) -> float:
            s = 0.5
            high_kw = ["important", "deadline", "decision", "error", "critical"]
            if any(kw in content.lower() for kw in high_kw):
                s += 0.3
            if len(content.split()) > 30:
                s += 0.1
            return min(1.0, s)

        # TODO: Add message with triage: keep/summarize/discard
        def add(self, content: str):
            importance = self.score(content)
            if importance >= 0.7:
                self.verbatim.append({"content": content, "importance": importance})
                if len(self.verbatim) > self.max_verbatim:
                    overflow = self.verbatim.pop(0)
                    self.summary_buffer.append(overflow["content"][:80])
            elif importance >= 0.4:
                self.summary_buffer.append(content[:80])
            else:
                self.discarded += 1

        def stats(self) -> dict:
            return {
                "verbatim": len(self.verbatim),
                "summarized": len(self.summary_buffer),
                "discarded": self.discarded,
            }

    mem = ImportanceMemory(max_verbatim=3)
    messages = [
        "Hello there",
        "IMPORTANT: Deadline is Friday",
        "Nice weather today",
        "The decision is to use REST API for this critical project",
        "ok",
        "Error in production: database connection failed",
        "Let me check on that",
    ]
    for msg in messages:
        mem.add(msg)

    print(f"Stats: {mem.stats()}")
    print("Verbatim messages:")
    for v in mem.verbatim:
        print(f"  [{v['importance']:.2f}] {v['content'][:60]}")


# === Exercise 4: Plan Refinement Loop ===
# Problem: Implement a plan that executes steps and re-plans on failure,
# with alternative actions for each step.

def exercise_4():
    """Iterative plan refinement with alternatives."""
    print("\n" + "=" * 60)
    print("Exercise 4: Plan Refinement")
    print("=" * 60)

    @dataclass
    class Step:
        name: str
        primary: Callable[[str], str]
        alternative: Callable[[str], str] | None = None

    class RefinablePlan:
        def __init__(self, steps: list[Step]):
            self.steps = steps
            self.log: list[dict] = []

        # TODO: Execute with fallback to alternative on failure
        def execute(self, initial: str) -> dict:
            context = initial
            for step in self.steps:
                try:
                    context = step.primary(context)
                    self.log.append({"step": step.name, "status": "ok"})
                except Exception as e:
                    self.log.append({"step": step.name, "status": "error", "error": str(e)})
                    if step.alternative:
                        try:
                            context = step.alternative(context)
                            self.log.append({"step": f"{step.name}_alt", "status": "ok"})
                        except Exception:
                            return {"success": False, "context": context, "log": self.log}
                    else:
                        return {"success": False, "context": context, "log": self.log}
            return {"success": True, "context": context, "log": self.log}

    fail_count = {"fetch": 0}

    def fetch_primary(ctx):
        fail_count["fetch"] += 1
        if fail_count["fetch"] <= 1:
            raise ConnectionError("Primary API down")
        return f"fetched({ctx})"

    def fetch_fallback(ctx):
        return f"fallback_fetched({ctx})"

    plan = RefinablePlan([
        Step("fetch", fetch_primary, fetch_fallback),
        Step("process", lambda ctx: f"processed({ctx})"),
        Step("save", lambda ctx: f"saved({ctx})"),
    ])

    result = plan.execute("initial_data")
    print(f"Success: {result['success']}")
    print(f"Final: {result['context'][:60]}")
    for entry in result["log"]:
        print(f"  [{entry['status']:5s}] {entry['step']}")


# === Exercise 5: Vector Memory with Forgetting Curve ===
# Problem: Implement memory that decays over time using exponential decay.
# Frequently accessed memories decay slower.

def exercise_5():
    """Memory with forgetting curve and access strengthening."""
    print("\n" + "=" * 60)
    print("Exercise 5: Forgetting Curve Memory")
    print("=" * 60)

    import math

    @dataclass
    class DecayMemory:
        content: str
        importance: float
        stability: float = 1.0
        access_count: int = 0
        last_accessed: float = field(default_factory=time.time)

        def access(self):
            self.access_count += 1
            self.last_accessed = time.time()
            self.stability = min(10.0, self.stability + 0.5)

        @property
        def retention(self) -> float:
            hours = (time.time() - self.last_accessed) / 3600
            return math.exp(-hours / self.stability)

        @property
        def effective_importance(self) -> float:
            return self.importance * self.retention

    class ForgettingMemory:
        def __init__(self, threshold: float = 0.1):
            self.memories: list[DecayMemory] = []
            self.threshold = threshold

        def add(self, content: str, importance: float = 0.5):
            self.memories.append(DecayMemory(content, importance))

        # TODO: Consolidate by removing decayed memories
        def consolidate(self) -> int:
            before = len(self.memories)
            self.memories = [
                m for m in self.memories
                if m.effective_importance >= self.threshold
            ]
            return before - len(self.memories)

        def search(self, query: str, top_k: int = 3) -> list[DecayMemory]:
            query_terms = set(query.lower().split())
            scored = []
            for m in self.memories:
                overlap = len(query_terms & set(m.content.lower().split()))
                if overlap > 0:
                    scored.append((overlap * m.effective_importance, m))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [m for _, m in scored[:top_k]]
            for m in results:
                m.access()
            return results

    store = ForgettingMemory(threshold=0.1)
    store.add("Project deadline is March 15", importance=0.9)
    store.add("Nice weather today", importance=0.1)
    store.add("Database uses PostgreSQL", importance=0.7)
    store.add("Coffee shop nearby", importance=0.05)

    # Search strengthens relevant memories
    results = store.search("project deadline")
    print(f"Search results: {len(results)}")
    for m in results:
        print(f"  [{m.effective_importance:.3f}] {m.content} "
              f"(stability={m.stability:.1f}, accesses={m.access_count})")

    forgotten = store.consolidate()
    print(f"Forgotten: {forgotten}, Remaining: {len(store.memories)}")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()

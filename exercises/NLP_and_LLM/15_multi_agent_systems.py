"""
Exercises for Lesson 15: Multi-Agent Systems
Topic: NLP_and_LLM

Practice problems for multi-agent architectures and communication.
"""

import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Any, Callable
from collections import defaultdict
from enum import Enum


# === Exercise 1: Agent Communication Protocol ===
# Problem: Design and implement a message-passing protocol for agents
# with priority queues and message acknowledgment.

def exercise_1():
    """Build an agent communication protocol with priority queues."""
    print("=" * 60)
    print("Exercise 1: Priority Message Protocol")
    print("=" * 60)

    @dataclass(order=True)
    class PriorityMessage:
        priority: int  # Lower = higher priority
        content: str = field(compare=False)
        sender: str = field(compare=False)
        receiver: str = field(compare=False)
        msg_id: str = field(compare=False, default="")
        acknowledged: bool = field(compare=False, default=False)

    class PriorityMessageBus:
        def __init__(self):
            self._queues: dict[str, queue.PriorityQueue] = defaultdict(queue.PriorityQueue)
            self._ack: dict[str, bool] = {}

        # TODO: Implement send method that adds message to priority queue
        def send(self, message: PriorityMessage):
            message.msg_id = f"msg-{len(self._ack)}"
            self._ack[message.msg_id] = False
            self._queues[message.receiver].put(message)

        # TODO: Implement receive method that returns highest-priority message
        def receive(self, agent_name: str, timeout: float = 1.0) -> PriorityMessage | None:
            try:
                return self._queues[agent_name].get(timeout=timeout)
            except queue.Empty:
                return None

        # TODO: Implement acknowledge method
        def acknowledge(self, msg_id: str):
            self._ack[msg_id] = True

        def get_unacked(self) -> list[str]:
            return [mid for mid, acked in self._ack.items() if not acked]

    bus = PriorityMessageBus()

    # Send messages with different priorities
    bus.send(PriorityMessage(3, "Low priority task", "supervisor", "worker"))
    bus.send(PriorityMessage(1, "URGENT: Fix critical bug", "supervisor", "worker"))
    bus.send(PriorityMessage(2, "Medium priority review", "supervisor", "worker"))

    # Receive in priority order
    for _ in range(3):
        msg = bus.receive("worker")
        if msg:
            print(f"  Priority {msg.priority}: {msg.content}")
            bus.acknowledge(msg.msg_id)

    print(f"  Unacknowledged: {bus.get_unacked()}")


# === Exercise 2: Supervisor Orchestrator ===
# Problem: Implement a supervisor that dynamically assigns tasks to workers
# based on their capabilities and current load.

def exercise_2():
    """Build a load-aware supervisor agent."""
    print("\n" + "=" * 60)
    print("Exercise 2: Load-Aware Supervisor")
    print("=" * 60)

    @dataclass
    class Worker:
        name: str
        capabilities: list[str]
        current_load: int = 0
        max_load: int = 3

        @property
        def available(self) -> bool:
            return self.current_load < self.max_load

    class Supervisor:
        def __init__(self, workers: list[Worker]):
            self.workers = {w.name: w for w in workers}
            self.task_log: list[dict] = []

        # TODO: Implement task assignment that matches capability and balances load
        def assign_task(self, task: str, required_capability: str) -> str | None:
            candidates = [
                w for w in self.workers.values()
                if required_capability in w.capabilities and w.available
            ]
            if not candidates:
                return None

            # Pick the worker with lowest load
            best = min(candidates, key=lambda w: w.current_load)
            best.current_load += 1
            self.task_log.append({
                "task": task,
                "worker": best.name,
                "capability": required_capability,
            })
            return best.name

        # TODO: Implement task completion that frees up worker capacity
        def complete_task(self, worker_name: str):
            if worker_name in self.workers:
                self.workers[worker_name].current_load = max(
                    0, self.workers[worker_name].current_load - 1
                )

        def status(self) -> dict:
            return {
                name: {"load": w.current_load, "max": w.max_load, "caps": w.capabilities}
                for name, w in self.workers.items()
            }

    workers = [
        Worker("researcher", ["research", "analysis"], max_load=2),
        Worker("writer", ["writing", "editing"], max_load=3),
        Worker("coder", ["coding", "analysis"], max_load=2),
        Worker("reviewer", ["review", "editing"], max_load=2),
    ]

    supervisor = Supervisor(workers)

    tasks = [
        ("Research transformer architectures", "research"),
        ("Write technical blog post", "writing"),
        ("Implement RAG pipeline", "coding"),
        ("Review code quality", "review"),
        ("Analyze benchmark results", "analysis"),
        ("Edit documentation", "editing"),
        ("Another research task", "research"),
    ]

    for task, cap in tasks:
        assigned = supervisor.assign_task(task, cap)
        if assigned:
            print(f"  [{assigned:12s}] <- {task}")
        else:
            print(f"  [UNASSIGNED ] <- {task} (no available worker with '{cap}')")

    print("\nWorker status:")
    for name, status in supervisor.status().items():
        print(f"  {name}: {status['load']}/{status['max']} ({status['caps']})")


# === Exercise 3: Shared Memory with Conflict Resolution ===
# Problem: Implement shared memory that handles concurrent writes
# with versioning and last-writer-wins conflict resolution.

def exercise_3():
    """Shared memory with versioning."""
    print("\n" + "=" * 60)
    print("Exercise 3: Versioned Shared Memory")
    print("=" * 60)

    @dataclass
    class VersionedEntry:
        key: str
        value: Any
        version: int
        author: str
        timestamp: float

    class VersionedMemory:
        def __init__(self):
            self._store: dict[str, VersionedEntry] = {}
            self._history: list[VersionedEntry] = []
            self._lock = threading.RLock()

        # TODO: Write with automatic version increment
        def write(self, key: str, value: Any, author: str) -> int:
            with self._lock:
                current = self._store.get(key)
                version = (current.version + 1) if current else 1
                entry = VersionedEntry(key, value, version, author, time.time())
                self._store[key] = entry
                self._history.append(entry)
                return version

        # TODO: Read with optional version specification
        def read(self, key: str, version: int | None = None) -> Any | None:
            with self._lock:
                if version is not None:
                    for entry in reversed(self._history):
                        if entry.key == key and entry.version == version:
                            return entry.value
                    return None
                entry = self._store.get(key)
                return entry.value if entry else None

        # TODO: Get version history for a key
        def get_history(self, key: str) -> list[dict]:
            with self._lock:
                return [
                    {"version": e.version, "value": e.value, "author": e.author}
                    for e in self._history if e.key == key
                ]

    memory = VersionedMemory()

    # Simulate multiple agents writing to the same key
    v1 = memory.write("research_findings", "Initial findings on LLMs", "researcher_1")
    v2 = memory.write("research_findings", "Updated: LLMs show 20% improvement", "researcher_2")
    v3 = memory.write("research_findings", "Final: Confirmed 20% with 95% CI", "researcher_1")

    memory.write("draft_status", "in_progress", "writer")
    memory.write("draft_status", "review_ready", "writer")

    print(f"Current value: {memory.read('research_findings')}")
    print(f"Version 1: {memory.read('research_findings', version=1)}")
    print(f"Version 2: {memory.read('research_findings', version=2)}")

    print("\nHistory for 'research_findings':")
    for entry in memory.get_history("research_findings"):
        print(f"  v{entry['version']} by {entry['author']}: {entry['value'][:50]}")


# === Exercise 4: Agent Pipeline with Error Recovery ===
# Problem: Build a sequential pipeline that can recover from
# agent failures by retrying or skipping.

def exercise_4():
    """Build a fault-tolerant agent pipeline."""
    print("\n" + "=" * 60)
    print("Exercise 4: Fault-Tolerant Pipeline")
    print("=" * 60)

    @dataclass
    class PipelineStep:
        name: str
        handler: Callable[[str], str]
        retries: int = 2
        optional: bool = False

    class FaultTolerantPipeline:
        def __init__(self, steps: list[PipelineStep]):
            self.steps = steps

        # TODO: Execute pipeline with retry and skip logic
        def execute(self, initial_input: str) -> dict:
            context = initial_input
            results = []
            errors = []

            for step in self.steps:
                success = False
                for attempt in range(step.retries + 1):
                    try:
                        result = step.handler(context)
                        context = result
                        results.append({"step": step.name, "output": result, "attempts": attempt + 1})
                        success = True
                        break
                    except Exception as e:
                        if attempt == step.retries:
                            errors.append({"step": step.name, "error": str(e)})
                            if step.optional:
                                print(f"  [{step.name}] SKIPPED (optional, failed)")
                                results.append({"step": step.name, "output": "[skipped]", "attempts": attempt + 1})
                            else:
                                print(f"  [{step.name}] FAILED (required)")
                                return {"success": False, "results": results, "errors": errors}
                        else:
                            print(f"  [{step.name}] Retry {attempt + 1}/{step.retries}")

                if success:
                    print(f"  [{step.name}] OK")

            return {"success": True, "results": results, "errors": errors}

    call_count = {"enhance": 0}

    def research(text: str) -> str:
        return f"Research on: {text} -> 3 findings"

    def write(text: str) -> str:
        return f"Article based on: {text[:30]}..."

    def enhance(text: str) -> str:
        # Fails first time, succeeds on retry
        call_count["enhance"] += 1
        if call_count["enhance"] <= 1:
            raise RuntimeError("Enhancement service temporarily unavailable")
        return f"Enhanced: {text[:30]}..."

    def review(text: str) -> str:
        return f"Reviewed: {text[:30]}..."

    pipeline = FaultTolerantPipeline([
        PipelineStep("research", research),
        PipelineStep("write", write),
        PipelineStep("enhance", enhance, retries=2, optional=True),
        PipelineStep("review", review),
    ])

    result = pipeline.execute("Impact of LLMs on software engineering")
    print(f"\nPipeline success: {result['success']}")
    print(f"Steps completed: {len(result['results'])}")
    if result['errors']:
        print(f"Errors: {result['errors']}")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()

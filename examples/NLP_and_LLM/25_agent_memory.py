"""
25. Agent Memory and Planning Example

Memory architectures, planning frameworks, and memory-augmented generation.
"""

from dataclasses import dataclass, field
from typing import Any
from collections import deque
from enum import Enum
import time
import json
import hashlib

print("=" * 60)
print("Agent Memory and Planning")
print("=" * 60)


# ============================================
# 1. Memory Types
# ============================================
print("\n[1] Memory Types")
print("-" * 40)


class MemoryType(Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


@dataclass
class MemoryEntry:
    """A single memory item."""
    content: str
    memory_type: MemoryType
    importance: float = 0.5
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def touch(self):
        self.access_count += 1
        self.last_accessed = time.time()


# Store some memories
memories = [
    MemoryEntry("User asked about Python decorators", MemoryType.EPISODIC, 0.7),
    MemoryEntry("Python is a programming language", MemoryType.SEMANTIC, 0.3),
    MemoryEntry("Use search tool for factual queries", MemoryType.PROCEDURAL, 0.8),
]

for m in memories:
    print(f"  [{m.memory_type.value:10s}] importance={m.importance:.1f} | {m.content}")


# ============================================
# 2. Short-Term Memory (Ring Buffer)
# ============================================
print("\n[2] Short-Term Memory")
print("-" * 40)


class ShortTermMemory:
    """Fixed-window conversation buffer."""

    def __init__(self, max_messages: int = 10):
        self.messages: deque[dict] = deque(maxlen=max_messages)

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content, "ts": time.time()})

    def get_messages(self) -> list[dict]:
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]


stm = ShortTermMemory(max_messages=6)
for i in range(8):
    stm.add("user", f"Question {i}")
    stm.add("assistant", f"Answer {i}")

print(f"Buffer size: {len(stm.messages)} (max=6)")
print(f"Oldest message: {stm.messages[0]['content']}")
print(f"Newest message: {stm.messages[-1]['content']}")


# ============================================
# 3. Entity Memory
# ============================================
print("\n[3] Entity Memory")
print("-" * 40)


@dataclass
class Entity:
    name: str
    entity_type: str
    attributes: dict[str, str] = field(default_factory=dict)
    mention_count: int = 1

    def update(self, new_attrs: dict[str, str]):
        self.attributes.update(new_attrs)
        self.mention_count += 1


class EntityStore:
    def __init__(self):
        self.entities: dict[str, Entity] = {}

    def add_or_update(self, name: str, entity_type: str, attrs: dict[str, str]):
        key = name.lower()
        if key in self.entities:
            self.entities[key].update(attrs)
        else:
            self.entities[key] = Entity(name, entity_type, attrs)

    def find(self, query: str) -> list[Entity]:
        query_lower = query.lower()
        return [e for k, e in self.entities.items() if query_lower in k]


store = EntityStore()
store.add_or_update("Alice", "person", {"role": "engineer", "team": "backend"})
store.add_or_update("Project X", "project", {"status": "active", "deadline": "March"})
store.add_or_update("Alice", "person", {"location": "NYC"})

for key, entity in store.entities.items():
    print(f"  {entity.name} ({entity.entity_type}): {entity.attributes} "
          f"[mentioned {entity.mention_count}x]")


# ============================================
# 4. Task Decomposition
# ============================================
print("\n[4] Task Decomposition")
print("-" * 40)


@dataclass
class Task:
    task_id: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    status: str = "pending"
    result: str = ""


class TaskPlanner:
    """Simple dependency-aware task planner."""

    def __init__(self, tasks: list[Task]):
        self.tasks = {t.task_id: t for t in tasks}

    def get_ready_tasks(self) -> list[Task]:
        completed = {tid for tid, t in self.tasks.items() if t.status == "done"}
        return [
            t for t in self.tasks.values()
            if t.status == "pending" and all(d in completed for d in t.dependencies)
        ]

    def complete(self, task_id: str, result: str):
        self.tasks[task_id].status = "done"
        self.tasks[task_id].result = result


planner = TaskPlanner([
    Task("t1", "Gather requirements"),
    Task("t2", "Design API", dependencies=["t1"]),
    Task("t3", "Design DB", dependencies=["t1"]),
    Task("t4", "Integration", dependencies=["t2", "t3"]),
])

# Execute in dependency order
for _ in range(4):
    ready = planner.get_ready_tasks()
    for task in ready:
        planner.complete(task.task_id, f"Done: {task.description}")
        print(f"  Completed: {task.task_id} - {task.description}")


# ============================================
# 5. Plan-and-Execute (Simulated)
# ============================================
print("\n[5] Plan-and-Execute Pattern")
print("-" * 40)


def simulate_plan_and_execute(goal: str):
    """Simulated plan-and-execute loop."""
    # Simulated planning step
    plan = [
        {"step": 1, "action": "search", "input": goal},
        {"step": 2, "action": "analyze", "input": "search results"},
        {"step": 3, "action": "summarize", "input": "analysis"},
    ]

    results = []
    for step in plan:
        result = f"Result of {step['action']}({step['input'][:20]}...)"
        results.append({"step": step["step"], "result": result})
        print(f"  Step {step['step']}: {step['action']} -> {result[:50]}")

    return results


simulate_plan_and_execute("Impact of LLMs on software engineering")


# ============================================
# 6. Self-Reflection (Simulated)
# ============================================
print("\n[6] Self-Reflection Pattern")
print("-" * 40)


def simulate_reflection(task: str, iterations: int = 3):
    """Simulate iterative self-reflection."""
    output = f"Initial answer for: {task[:30]}..."

    for i in range(iterations):
        # Simulated reflection
        score = 5 + i * 2  # Score improves each iteration
        weakness = f"Missing detail #{i+1}" if score < 9 else None

        print(f"  Iteration {i+1}: score={score}/10", end="")
        if weakness:
            print(f" (weakness: {weakness})")
            output = f"Improved answer v{i+2} for: {task[:30]}..."
        else:
            print(" -> SATISFACTORY")
            break

    return output


result = simulate_reflection("Explain transformer attention mechanism")
print(f"  Final: {result}")


print("\n" + "=" * 60)
print("Agent Memory and Planning example complete!")
print("=" * 60)

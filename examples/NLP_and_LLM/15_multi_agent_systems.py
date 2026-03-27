"""
15. Multi-Agent Systems Example

Orchestration patterns, inter-agent communication, and multi-agent RAG
"""

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum
import json
import time
import threading
import queue
from collections import defaultdict

print("=" * 60)
print("Multi-Agent Systems")
print("=" * 60)


# ============================================
# 1. Core Components
# ============================================
print("\n[1] Core Agent Components")
print("-" * 40)


class AgentRole(Enum):
    RESEARCHER = "researcher"
    WRITER = "writer"
    REVIEWER = "reviewer"
    SUPERVISOR = "supervisor"


@dataclass
class AgentMessage:
    """Message passed between agents."""
    sender: str
    receiver: str
    content: str
    metadata: dict = field(default_factory=dict)
    message_type: str = "task"


@dataclass
class AgentState:
    """Shared state for multi-agent workflows."""
    messages: list[AgentMessage] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    current_step: int = 0
    is_complete: bool = False

    def add_message(self, msg: AgentMessage):
        self.messages.append(msg)

    def get_messages_for(self, agent_name: str) -> list[AgentMessage]:
        return [m for m in self.messages if m.receiver == agent_name]


state = AgentState()
state.add_message(AgentMessage("supervisor", "researcher", "Research topic X"))
state.add_message(AgentMessage("researcher", "writer", "Here are findings..."))
print(f"Total messages: {len(state.messages)}")
print(f"Messages for writer: {len(state.get_messages_for('writer'))}")


# ============================================
# 2. Sequential Pipeline (simulated)
# ============================================
print("\n[2] Sequential Pipeline")
print("-" * 40)


def create_simulated_agent(name: str, transform: Callable[[str], str]):
    """Create a simulated agent (no LLM needed)."""
    def agent(input_text: str) -> str:
        print(f"  [{name}] Processing...")
        return transform(input_text)
    agent.__name__ = name
    return agent


researcher = create_simulated_agent(
    "researcher",
    lambda text: f"Research findings for '{text}': 3 key points identified."
)
writer = create_simulated_agent(
    "writer",
    lambda text: f"Article based on: {text[:50]}... [500 words generated]"
)
editor = create_simulated_agent(
    "editor",
    lambda text: f"Edited version of: {text[:50]}... [grammar and style improved]"
)


def sequential_pipeline(topic: str) -> dict[str, str]:
    """Run agents in sequence: Researcher -> Writer -> Editor."""
    results = {}
    print(f"Pipeline started for topic: '{topic}'")

    research = researcher(topic)
    results["research"] = research

    article = writer(research)
    results["draft"] = article

    final = editor(article)
    results["final"] = final

    return results


pipeline_result = sequential_pipeline("Transformer architecture impact on NLP")
for stage, output in pipeline_result.items():
    print(f"  {stage}: {output[:60]}...")


# ============================================
# 3. Supervisor Pattern (simulated)
# ============================================
print("\n[3] Supervisor Pattern")
print("-" * 40)


class SimulatedSupervisor:
    """Supervisor that delegates to workers based on simple rules."""

    def __init__(self, workers: dict[str, Callable]):
        self.workers = workers

    def run(self, task: str, max_steps: int = 5) -> str:
        history = []
        steps = [
            ("researcher", f"Research: {task}"),
            ("writer", "Write article from research"),
            ("editor", "Edit and finalize"),
        ]

        for i, (worker_name, sub_task) in enumerate(steps):
            if i >= max_steps:
                break
            print(f"  Step {i+1}: Delegating to [{worker_name}]")
            result = self.workers[worker_name](sub_task)
            history.append({"worker": worker_name, "result": result})

        return f"Completed {len(history)} steps. Final: {history[-1]['result'][:80]}"


supervisor = SimulatedSupervisor(workers={
    "researcher": researcher,
    "writer": writer,
    "editor": editor,
})
result = supervisor.run("Vector databases for LLM applications")
print(f"  Result: {result[:100]}...")


# ============================================
# 4. Message Bus
# ============================================
print("\n[4] Message Bus Communication")
print("-" * 40)


class MessageBus:
    """Central message bus for agent communication."""

    def __init__(self):
        self._queues: dict[str, queue.Queue] = defaultdict(queue.Queue)
        self._lock = threading.Lock()

    def send(self, message: AgentMessage):
        with self._lock:
            self._queues[message.receiver].put(message)

    def receive(self, agent_name: str, timeout: float = 1.0) -> AgentMessage | None:
        try:
            return self._queues[agent_name].get(timeout=timeout)
        except queue.Empty:
            return None

    def broadcast(self, sender: str, content: str, agents: list[str]):
        for agent_name in agents:
            if agent_name != sender:
                self.send(AgentMessage(sender, agent_name, content, message_type="broadcast"))


bus = MessageBus()
bus.send(AgentMessage("supervisor", "researcher", "Start research on topic A"))
bus.send(AgentMessage("supervisor", "writer", "Prepare to write"))
bus.broadcast("supervisor", "Deadline is 5pm", ["researcher", "writer", "editor"])

msg = bus.receive("researcher")
print(f"Researcher received: '{msg.content}' from {msg.sender}" if msg else "No message")

msg2 = bus.receive("writer")
print(f"Writer received: '{msg2.content}' from {msg2.sender}" if msg2 else "No message")


# ============================================
# 5. Shared Memory
# ============================================
print("\n[5] Shared Memory Store")
print("-" * 40)


@dataclass
class MemoryEntry:
    key: str
    value: Any
    author: str
    timestamp: float
    ttl: float | None = None


class SharedMemory:
    """Thread-safe shared memory for multi-agent systems."""

    def __init__(self):
        self._store: dict[str, MemoryEntry] = {}
        self._lock = threading.RLock()

    def write(self, key: str, value: Any, author: str, ttl: float | None = None):
        with self._lock:
            self._store[key] = MemoryEntry(key, value, author, time.time(), ttl)

    def read(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.ttl and (time.time() - entry.timestamp) > entry.ttl:
                del self._store[key]
                return None
            return entry.value

    def read_all(self) -> dict[str, Any]:
        with self._lock:
            return {k: e.value for k, e in self._store.items()
                    if not e.ttl or (time.time() - e.timestamp) <= e.ttl}


memory = SharedMemory()
memory.write("research_findings", {"topic": "LLM", "papers": 5}, "researcher")
memory.write("draft_status", "in_progress", "writer")
memory.write("temp_data", "expires soon", "system", ttl=0.001)

time.sleep(0.01)  # Let TTL expire

print(f"Research findings: {memory.read('research_findings')}")
print(f"Draft status: {memory.read('draft_status')}")
print(f"Temp data (expired): {memory.read('temp_data')}")
print(f"All entries: {memory.read_all()}")


# ============================================
# 6. Agent Tracing
# ============================================
print("\n[6] Agent Tracing")
print("-" * 40)


class AgentTracer:
    """Trace multi-agent execution for debugging."""

    def __init__(self):
        self.steps: list[dict] = []

    def trace(self, agent_name: str, action: str, result: Any, duration_ms: float):
        self.steps.append({
            "agent": agent_name,
            "action": action,
            "duration_ms": round(duration_ms, 2),
            "output_preview": str(result)[:100],
        })

    def summary(self) -> dict:
        total_duration = sum(s["duration_ms"] for s in self.steps)
        return {
            "total_steps": len(self.steps),
            "total_duration_ms": round(total_duration, 2),
            "agents_used": list({s["agent"] for s in self.steps}),
        }


tracer = AgentTracer()

start = time.time()
r1 = researcher("AI safety")
tracer.trace("researcher", "research", r1, (time.time() - start) * 1000)

start = time.time()
r2 = writer(r1)
tracer.trace("writer", "write", r2, (time.time() - start) * 1000)

print(f"Trace summary: {json.dumps(tracer.summary(), indent=2)}")

print("\n" + "=" * 60)
print("Multi-Agent Systems example complete!")
print("=" * 60)

"""
Exercises for Lesson 08: Agent Teams
Topic: Claude_Ecosystem

Solutions to practice problems from the lesson.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# === Exercise 1: Team Role Assignment ===
# Problem: Define agent team roles and assign tasks based on expertise.

class AgentRole(Enum):
    LEAD = "lead"           # Orchestrator, breaks down tasks
    IMPLEMENTER = "implementer"  # Writes code
    REVIEWER = "reviewer"   # Reviews code quality
    TESTER = "tester"       # Writes and runs tests
    RESEARCHER = "researcher"  # Explores codebase, gathers context


@dataclass
class TeamAgent:
    """An agent within a team with defined capabilities."""
    name: str
    role: AgentRole
    specialties: list[str]
    max_concurrent_tasks: int = 1


@dataclass
class TaskAssignment:
    """A task assigned to a specific agent."""
    task: str
    agent: TeamAgent
    priority: int
    status: str = "pending"


def assign_tasks(
    agents: list[TeamAgent],
    tasks: list[dict[str, Any]],
) -> list[TaskAssignment]:
    """Assign tasks to agents based on role and specialty matching."""
    assignments: list[TaskAssignment] = []

    for task_info in tasks:
        task_desc = task_info["description"]
        required_role = task_info.get("role")
        keywords = task_info.get("keywords", [])
        priority = task_info.get("priority", 5)

        best_agent = None
        best_score = -1

        for agent in agents:
            if required_role and agent.role != required_role:
                continue
            score = sum(
                1 for kw in keywords
                if any(kw.lower() in s.lower() for s in agent.specialties)
            )
            if required_role:
                score += 10
            if score > best_score:
                best_score = score
                best_agent = agent

        if best_agent:
            assignments.append(TaskAssignment(task_desc, best_agent, priority))

    return sorted(assignments, key=lambda a: a.priority)


def exercise_1():
    """Demonstrate team task assignment."""
    team = [
        TeamAgent("Alice", AgentRole.LEAD, ["architecture", "planning"]),
        TeamAgent("Bob", AgentRole.IMPLEMENTER, ["python", "fastapi", "backend"]),
        TeamAgent("Carol", AgentRole.IMPLEMENTER, ["react", "typescript", "frontend"]),
        TeamAgent("Dave", AgentRole.TESTER, ["pytest", "testing", "ci"]),
        TeamAgent("Eve", AgentRole.REVIEWER, ["security", "code quality"]),
    ]
    tasks = [
        {"description": "Design API schema", "role": AgentRole.LEAD,
         "keywords": ["architecture"], "priority": 1},
        {"description": "Implement user endpoint", "role": AgentRole.IMPLEMENTER,
         "keywords": ["python", "fastapi"], "priority": 2},
        {"description": "Build login form", "role": AgentRole.IMPLEMENTER,
         "keywords": ["react", "frontend"], "priority": 2},
        {"description": "Write integration tests", "role": AgentRole.TESTER,
         "keywords": ["pytest"], "priority": 3},
        {"description": "Security review", "role": AgentRole.REVIEWER,
         "keywords": ["security"], "priority": 4},
    ]
    assignments = assign_tasks(team, tasks)
    for a in assignments:
        print(f"  P{a.priority} [{a.agent.role.value}] {a.agent.name}: "
              f"{a.task}")


# === Exercise 2: Workflow Orchestrator ===
# Problem: Model a multi-agent workflow with dependencies and parallel execution.

@dataclass
class WorkflowStep:
    """A step in a multi-agent workflow."""
    name: str
    agent_role: AgentRole
    depends_on: list[str] = field(default_factory=list)
    completed: bool = False


@dataclass
class Workflow:
    """An orchestrated multi-agent workflow."""
    steps: list[WorkflowStep]

    def get_ready_steps(self) -> list[WorkflowStep]:
        """Get steps whose dependencies are all completed."""
        completed_names = {s.name for s in self.steps if s.completed}
        return [
            s for s in self.steps
            if not s.completed
            and all(d in completed_names for d in s.depends_on)
        ]

    def complete_step(self, name: str) -> None:
        for s in self.steps:
            if s.name == name:
                s.completed = True
                return

    def is_done(self) -> bool:
        return all(s.completed for s in self.steps)

    def simulate(self) -> list[list[str]]:
        """Simulate execution, returning batches of parallel steps."""
        batches: list[list[str]] = []
        while not self.is_done():
            ready = self.get_ready_steps()
            if not ready:
                break
            batch = [s.name for s in ready]
            batches.append(batch)
            for name in batch:
                self.complete_step(name)
        return batches


def exercise_2():
    """Simulate a multi-agent workflow."""
    workflow = Workflow([
        WorkflowStep("explore_codebase", AgentRole.RESEARCHER),
        WorkflowStep("design_plan", AgentRole.LEAD,
                     depends_on=["explore_codebase"]),
        WorkflowStep("implement_backend", AgentRole.IMPLEMENTER,
                     depends_on=["design_plan"]),
        WorkflowStep("implement_frontend", AgentRole.IMPLEMENTER,
                     depends_on=["design_plan"]),
        WorkflowStep("write_tests", AgentRole.TESTER,
                     depends_on=["implement_backend", "implement_frontend"]),
        WorkflowStep("code_review", AgentRole.REVIEWER,
                     depends_on=["implement_backend", "implement_frontend"]),
        WorkflowStep("final_integration", AgentRole.LEAD,
                     depends_on=["write_tests", "code_review"]),
    ])
    batches = workflow.simulate()
    for i, batch in enumerate(batches):
        parallel = " (parallel)" if len(batch) > 1 else ""
        print(f"  Round {i + 1}{parallel}: {', '.join(batch)}")


# === Exercise 3: Shared Task List ===
# Problem: Implement a shared task list that multiple agents can
#   read and update concurrently.

@dataclass
class SharedTask:
    """A task in the shared team task list."""
    id: int
    description: str
    status: str = "pending"      # pending, in_progress, completed
    assigned_to: str | None = None
    notes: list[str] = field(default_factory=list)


class SharedTaskList:
    """Thread-safe task list shared between agents."""

    def __init__(self) -> None:
        self._tasks: list[SharedTask] = []
        self._next_id = 1

    def add(self, description: str) -> SharedTask:
        task = SharedTask(id=self._next_id, description=description)
        self._tasks.append(task)
        self._next_id += 1
        return task

    def claim(self, task_id: int, agent_name: str) -> bool:
        task = self._find(task_id)
        if task and task.status == "pending":
            task.status = "in_progress"
            task.assigned_to = agent_name
            return True
        return False

    def complete(self, task_id: int, note: str = "") -> bool:
        task = self._find(task_id)
        if task and task.status == "in_progress":
            task.status = "completed"
            if note:
                task.notes.append(note)
            return True
        return False

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {"pending": 0, "in_progress": 0, "completed": 0}
        for t in self._tasks:
            counts[t.status] += 1
        return counts

    def _find(self, task_id: int) -> SharedTask | None:
        return next((t for t in self._tasks if t.id == task_id), None)


def exercise_3():
    """Demonstrate shared task list operations."""
    tasks = SharedTaskList()
    tasks.add("Set up project structure")
    tasks.add("Implement database models")
    tasks.add("Create API endpoints")
    tasks.add("Write unit tests")

    tasks.claim(1, "Alice")
    tasks.complete(1, "Project scaffolded with FastAPI template")
    tasks.claim(2, "Bob")
    tasks.claim(3, "Carol")
    tasks.complete(2, "SQLAlchemy models created")

    print(f"  Status: {tasks.summary()}")
    for t in tasks._tasks:
        assigned = f" ({t.assigned_to})" if t.assigned_to else ""
        notes = f" — {t.notes[0]}" if t.notes else ""
        print(f"  [{t.status:<12}] #{t.id} {t.description}{assigned}{notes}")


if __name__ == "__main__":
    print("=== Exercise 1: Team Role Assignment ===")
    exercise_1()

    print("\n=== Exercise 2: Workflow Orchestrator ===")
    exercise_2()

    print("\n=== Exercise 3: Shared Task List ===")
    exercise_3()

    print("\nAll exercises completed!")

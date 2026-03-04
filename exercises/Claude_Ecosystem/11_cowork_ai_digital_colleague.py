"""
Exercises for Lesson 11: Cowork — AI Digital Colleague
Topic: Claude_Ecosystem

Solutions to practice problems from the lesson.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


# === Exercise 1: Cowork Session Simulator ===
# Problem: Simulate a Cowork session where Claude autonomously executes
#   a multi-step task with checkpoints and status updates.

class TaskStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING = "waiting_for_input"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CoworkStep:
    """A single step in a Cowork task execution."""
    description: str
    status: TaskStatus = TaskStatus.QUEUED
    output: str = ""
    duration_seconds: int = 0


@dataclass
class CoworkSession:
    """A Cowork autonomous work session."""
    task: str
    steps: list[CoworkStep] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    notifications: list[str] = field(default_factory=list)

    def add_step(self, description: str) -> int:
        self.steps.append(CoworkStep(description))
        return len(self.steps) - 1

    def execute_step(self, index: int, output: str,
                     duration: int = 5) -> None:
        step = self.steps[index]
        step.status = TaskStatus.RUNNING
        step.output = output
        step.duration_seconds = duration
        step.status = TaskStatus.COMPLETED

    def fail_step(self, index: int, error: str) -> None:
        step = self.steps[index]
        step.status = TaskStatus.FAILED
        step.output = error
        self.notifications.append(f"Step failed: {step.description} — {error}")

    def request_input(self, index: int, question: str) -> None:
        step = self.steps[index]
        step.status = TaskStatus.WAITING
        self.notifications.append(f"Input needed: {question}")

    def progress(self) -> dict[str, Any]:
        completed = sum(1 for s in self.steps
                        if s.status == TaskStatus.COMPLETED)
        total = len(self.steps)
        total_time = sum(s.duration_seconds for s in self.steps)
        return {
            "completed": completed,
            "total": total,
            "pct": round(completed / max(total, 1) * 100),
            "elapsed_seconds": total_time,
            "current_status": (
                self.steps[-1].status.value if self.steps else "no steps"
            ),
        }


def exercise_1():
    """Simulate a Cowork session."""
    session = CoworkSession(task="Refactor auth module and add tests")

    # Plan steps
    steps = [
        "Analyze current auth module structure",
        "Identify code smells and refactoring targets",
        "Refactor session management",
        "Refactor token validation",
        "Write unit tests for refactored code",
        "Run test suite and verify coverage",
    ]
    for desc in steps:
        session.add_step(desc)

    # Simulate execution
    session.execute_step(0, "Found 3 files: auth.py, session.py, tokens.py", 10)
    session.execute_step(1, "Issues: God class, no DI, mixed concerns", 8)
    session.execute_step(2, "Extracted SessionManager class", 30)
    session.execute_step(3, "Extracted TokenValidator class", 25)
    session.fail_step(4, "Import error: missing conftest.py fixture")
    session.request_input(5, "Should I create conftest.py with shared fixtures?")

    progress = session.progress()
    print(f"  Task: {session.task}")
    print(f"  Progress: {progress['completed']}/{progress['total']} "
          f"({progress['pct']}%)")
    print(f"  Elapsed: {progress['elapsed_seconds']}s")
    print(f"  Notifications:")
    for n in session.notifications:
        print(f"    - {n}")


# === Exercise 2: Task Delegation Planner ===
# Problem: Determine which tasks are suitable for Cowork (autonomous)
#   vs interactive Claude Code sessions.

@dataclass
class TaskProfile:
    """Profile of a development task for delegation analysis."""
    description: str
    estimated_steps: int
    needs_human_judgment: bool
    is_well_defined: bool
    affects_production: bool
    has_test_coverage: bool


def assess_cowork_suitability(task: TaskProfile) -> dict[str, Any]:
    """Assess whether a task is suitable for Cowork delegation."""
    score = 0
    reasons: list[str] = []

    if task.is_well_defined:
        score += 3
        reasons.append("Well-defined requirements")
    else:
        score -= 2
        reasons.append("Ambiguous requirements — needs clarification")

    if not task.needs_human_judgment:
        score += 2
        reasons.append("Can proceed without human decisions")
    else:
        score -= 1
        reasons.append("May need human input during execution")

    if task.has_test_coverage:
        score += 2
        reasons.append("Existing tests provide safety net")

    if task.affects_production:
        score -= 2
        reasons.append("Production impact — review recommended")

    if task.estimated_steps <= 10:
        score += 1
    else:
        reasons.append("Many steps — consider breaking down")

    recommendation = (
        "Cowork" if score >= 4
        else "Interactive" if score <= 1
        else "Cowork with checkpoints"
    )
    return {
        "score": score,
        "recommendation": recommendation,
        "reasons": reasons,
    }


def exercise_2():
    """Assess task suitability for Cowork."""
    tasks = [
        TaskProfile("Add pagination to API list endpoints", 5,
                     False, True, False, True),
        TaskProfile("Redesign the database schema", 15,
                     True, False, True, False),
        TaskProfile("Update dependencies and fix deprecations", 8,
                     False, True, False, True),
        TaskProfile("Implement new payment gateway", 12,
                     True, False, True, False),
    ]
    for t in tasks:
        result = assess_cowork_suitability(t)
        print(f"  {t.description}")
        print(f"    → {result['recommendation']} (score: {result['score']})")
        for r in result["reasons"][:2]:
            print(f"      - {r}")
        print()


# === Exercise 3: Notification System ===
# Problem: Build a notification system for Cowork session events.

class NotificationPriority(Enum):
    INFO = "info"
    WARNING = "warning"
    ACTION_REQUIRED = "action_required"
    ERROR = "error"


@dataclass
class Notification:
    """A notification from a Cowork session."""
    priority: NotificationPriority
    message: str
    session_task: str
    timestamp: datetime = field(default_factory=datetime.now)


class NotificationManager:
    """Manages notifications from Cowork sessions."""

    def __init__(self) -> None:
        self._notifications: list[Notification] = []

    def notify(self, priority: NotificationPriority,
               message: str, session_task: str) -> None:
        self._notifications.append(
            Notification(priority, message, session_task))

    def get_unread(self,
                   min_priority: NotificationPriority | None = None,
                   ) -> list[Notification]:
        if min_priority is None:
            return list(self._notifications)
        priority_order = list(NotificationPriority)
        min_idx = priority_order.index(min_priority)
        return [n for n in self._notifications
                if priority_order.index(n.priority) >= min_idx]

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {p.value: 0 for p in NotificationPriority}
        for n in self._notifications:
            counts[n.priority.value] += 1
        return counts


def exercise_3():
    """Demonstrate Cowork notification system."""
    mgr = NotificationManager()
    mgr.notify(NotificationPriority.INFO, "Started refactoring auth module",
               "Refactor auth")
    mgr.notify(NotificationPriority.INFO, "Completed step 1/6: analysis",
               "Refactor auth")
    mgr.notify(NotificationPriority.WARNING,
               "Test coverage dropped below 80%", "Refactor auth")
    mgr.notify(NotificationPriority.ACTION_REQUIRED,
               "Need approval to modify production config", "Refactor auth")
    mgr.notify(NotificationPriority.ERROR,
               "Build failed after dependency update", "Update deps")

    print(f"  Summary: {mgr.summary()}")
    print(f"\n  Action-required and above:")
    for n in mgr.get_unread(NotificationPriority.ACTION_REQUIRED):
        print(f"    [{n.priority.value}] {n.message} ({n.session_task})")


if __name__ == "__main__":
    print("=== Exercise 1: Cowork Session ===")
    exercise_1()

    print("\n=== Exercise 2: Delegation Planner ===")
    exercise_2()

    print("\n=== Exercise 3: Notification System ===")
    exercise_3()

    print("\nAll exercises completed!")

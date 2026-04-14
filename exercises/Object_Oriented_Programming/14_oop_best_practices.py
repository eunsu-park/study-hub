"""
Exercise 14: OOP Best Practices
Topic: Object-Oriented Programming

Practice applying OOP best practices: SRP, DI, composition, and clean design.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod


# =============================================================================
# Challenge: Build a Clean Task Management System
#
# Requirements:
# 1. Task (dataclass): title, priority (1-5), done status, created timestamp
# 2. TaskRepository: stores tasks, find/filter operations (SRP: only storage)
# 3. TaskService: business logic using injected repository (DIP)
# 4. PriorityFilter (Strategy): filter tasks by priority criteria
# =============================================================================


@dataclass
class Task:
    """A task with title, priority, and completion status.

    Fields:
        title (str): Task title.
        priority (int): Priority 1 (highest) to 5 (lowest). Default 3.
        done (bool): Whether task is completed. Default False.

    Post-init validation:
        - title must not be empty (raise ValueError)
        - priority must be 1-5 (raise ValueError)

    Methods:
        complete(): Mark task as done.
        __str__(): Return "[X] title (P1)" or "[ ] title (P1)"
    """

    # TODO: Implement
    pass


class TaskRepository:
    """Repository for storing and retrieving tasks (SRP: only persistence).

    Methods:
        add(task): Add a task. Raise ValueError if title already exists.
        remove(title): Remove task by title. Raise KeyError if not found.
        get(title): Get task by title. Return None if not found.
        all(): Return list of all tasks.
        find(predicate): Return list of tasks matching predicate function.
        __len__(): Return number of tasks.
    """

    # TODO: Implement
    pass


class TaskFilter(ABC):
    """Abstract filter strategy."""

    @abstractmethod
    def matches(self, task: Task) -> bool:
        pass


class HighPriorityFilter(TaskFilter):
    """Matches tasks with priority <= 2."""

    # TODO: Implement
    pass


class PendingFilter(TaskFilter):
    """Matches tasks that are not done."""

    # TODO: Implement
    pass


class CombinedFilter(TaskFilter):
    """Combines multiple filters with AND logic.

    Args:
        *filters: Variable number of TaskFilter instances.

    matches(): Returns True only if ALL filters match.
    """

    # TODO: Implement
    pass


class TaskService:
    """Business logic for task management (uses injected repository).

    Args:
        repository (TaskRepository): Injected dependency.

    Methods:
        create_task(title, priority=3): Create and add task, return it.
        complete_task(title): Mark task as done. Raise KeyError if not found.
        pending_count(): Return number of incomplete tasks.
        filter_tasks(task_filter): Return tasks matching the filter strategy.
        summary(): Return dict with keys:
            "total", "done", "pending", "high_priority_pending"
    """

    # TODO: Implement
    pass


if __name__ == "__main__":
    # Set up
    repo = TaskRepository()
    service = TaskService(repo)

    # Create tasks
    service.create_task("Write tests", 1)
    service.create_task("Fix bug", 2)
    service.create_task("Update docs", 3)
    service.create_task("Refactor code", 2)
    service.create_task("Add logging", 4)

    assert len(repo) == 5

    # Complete some tasks
    service.complete_task("Fix bug")
    service.complete_task("Add logging")

    # Summary
    summary = service.summary()
    assert summary["total"] == 5
    assert summary["done"] == 2
    assert summary["pending"] == 3
    assert summary["high_priority_pending"] == 2  # Write tests (P1), Refactor (P2)

    print("Summary:", summary)

    # Filter with Strategy pattern
    high_priority = service.filter_tasks(HighPriorityFilter())
    assert len(high_priority) == 3  # P1 + P2 + P2 (including done)
    print(f"\nHigh priority tasks: {[t.title for t in high_priority]}")

    pending = service.filter_tasks(PendingFilter())
    assert len(pending) == 3
    print(f"Pending tasks: {[t.title for t in pending]}")

    # Combined filter
    urgent = service.filter_tasks(CombinedFilter(HighPriorityFilter(), PendingFilter()))
    assert len(urgent) == 2  # Write tests, Refactor code
    print(f"Urgent (high priority + pending): {[t.title for t in urgent]}")

    # Duplicate prevention
    try:
        service.create_task("Write tests", 1)
        assert False, "Should raise ValueError for duplicate"
    except ValueError:
        pass

    # Task validation
    try:
        Task("", 3)
        assert False, "Should reject empty title"
    except ValueError:
        pass

    try:
        Task("Test", 6)
        assert False, "Should reject invalid priority"
    except ValueError:
        pass

    # Print all tasks
    print("\nAll tasks:")
    for task in repo.all():
        print(f"  {task}")

    print("\nAll tests passed!")

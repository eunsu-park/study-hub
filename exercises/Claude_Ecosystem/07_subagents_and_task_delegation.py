"""
Exercises for Lesson 07: Subagents and Task Delegation
Topic: Claude_Ecosystem

Solutions to practice problems from the lesson.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# === Exercise 1: Subagent Type Selector ===
# Problem: Given a task description, select the optimal subagent type
#   (Explore, Plan, General-Purpose).

class SubagentType(Enum):
    EXPLORE = "Explore"
    PLAN = "Plan"
    GENERAL_PURPOSE = "general-purpose"


@dataclass
class SubagentSpec:
    """Specification for a Claude Code subagent type."""
    agent_type: SubagentType
    description: str
    read_only: bool
    tools: list[str]
    best_for: list[str]


SUBAGENT_SPECS = [
    SubagentSpec(
        SubagentType.EXPLORE,
        "Fast codebase exploration and search",
        True,
        ["Read", "Glob", "Grep", "Bash(read-only)"],
        ["find files", "search code", "understand architecture",
         "locate definitions", "explore"],
    ),
    SubagentSpec(
        SubagentType.PLAN,
        "Design implementation strategies",
        True,
        ["Read", "Glob", "Grep", "Bash(read-only)"],
        ["plan", "design", "architect", "strategy", "approach",
         "implement plan"],
    ),
    SubagentSpec(
        SubagentType.GENERAL_PURPOSE,
        "Execute complex multi-step tasks",
        False,
        ["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        ["fix", "implement", "build", "create", "modify",
         "test", "debug", "refactor"],
    ),
]


def select_subagent(task: str) -> SubagentSpec:
    """Select the best subagent type for a given task."""
    task_lower = task.lower()
    scored: list[tuple[SubagentSpec, int]] = []

    for spec in SUBAGENT_SPECS:
        score = sum(1 for kw in spec.best_for if kw in task_lower)
        scored.append((spec, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0] if scored[0][1] > 0 else SUBAGENT_SPECS[2]


def exercise_1():
    """Demonstrate subagent selection."""
    tasks = [
        "Find all files that import the auth module",
        "Plan the implementation for a new caching layer",
        "Fix the failing unit test in test_parser.py",
        "Explore how the routing system works",
        "Create a new API endpoint for user profiles",
    ]
    for task in tasks:
        spec = select_subagent(task)
        print(f"  '{task}'")
        print(f"    → {spec.agent_type.value} "
              f"(read-only: {spec.read_only})\n")


# === Exercise 2: Task Decomposer ===
# Problem: Break a complex task into subtasks and assign each
#   to the appropriate subagent type.

@dataclass
class Subtask:
    """A decomposed subtask with agent assignment."""
    description: str
    agent_type: SubagentType
    dependencies: list[int]  # indices of prerequisite subtasks
    parallel: bool = False


def decompose_task(task: str) -> list[Subtask]:
    """Decompose a high-level task into ordered subtasks.

    This is a rule-based simulation of how Claude Code breaks tasks
    into subagent-appropriate chunks.
    """
    subtasks: list[Subtask] = []

    # Phase 1: Exploration (always first)
    subtasks.append(Subtask(
        f"Explore codebase to understand context for: {task}",
        SubagentType.EXPLORE, dependencies=[], parallel=False,
    ))

    # Phase 2: Planning
    subtasks.append(Subtask(
        f"Design implementation approach for: {task}",
        SubagentType.PLAN, dependencies=[0], parallel=False,
    ))

    # Phase 3: Implementation (can be parallelized)
    subtasks.append(Subtask(
        f"Implement changes for: {task}",
        SubagentType.GENERAL_PURPOSE, dependencies=[1], parallel=True,
    ))

    # Phase 4: Verification
    subtasks.append(Subtask(
        f"Run tests and verify: {task}",
        SubagentType.GENERAL_PURPOSE, dependencies=[2], parallel=False,
    ))

    return subtasks


def exercise_2():
    """Demonstrate task decomposition."""
    task = "Add user authentication with JWT to the FastAPI backend"
    subtasks = decompose_task(task)
    for i, st in enumerate(subtasks):
        deps = f" (after: {st.dependencies})" if st.dependencies else ""
        par = " [parallelizable]" if st.parallel else ""
        print(f"  {i}. [{st.agent_type.value}] {st.description}"
              f"{deps}{par}")


# === Exercise 3: Result Aggregator ===
# Problem: Collect and merge results from multiple subagent runs.

@dataclass
class SubagentResult:
    """Result from a subagent execution."""
    agent_type: SubagentType
    task: str
    files_found: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    success: bool = True


def aggregate_results(results: list[SubagentResult]) -> dict[str, Any]:
    """Aggregate results from multiple subagent runs."""
    all_files_found: set[str] = set()
    all_files_modified: set[str] = set()
    all_insights: list[str] = []
    all_succeeded = True

    for r in results:
        all_files_found.update(r.files_found)
        all_files_modified.update(r.files_modified)
        all_insights.extend(r.insights)
        if not r.success:
            all_succeeded = False

    return {
        "total_agents": len(results),
        "all_succeeded": all_succeeded,
        "unique_files_found": sorted(all_files_found),
        "files_modified": sorted(all_files_modified),
        "insights": all_insights,
        "agents_by_type": {
            t.value: sum(1 for r in results if r.agent_type == t)
            for t in SubagentType
        },
    }


def exercise_3():
    """Demonstrate result aggregation from parallel subagents."""
    results = [
        SubagentResult(
            SubagentType.EXPLORE, "Find auth-related files",
            files_found=["src/auth.py", "src/middleware.py", "tests/test_auth.py"],
            insights=["Auth uses session-based authentication",
                       "No JWT implementation found"]),
        SubagentResult(
            SubagentType.EXPLORE, "Find API route files",
            files_found=["src/routes/users.py", "src/routes/auth.py",
                          "src/middleware.py"],
            insights=["Routes follow RESTful conventions"]),
        SubagentResult(
            SubagentType.GENERAL_PURPOSE, "Implement JWT auth",
            files_modified=["src/auth.py", "src/middleware.py"],
            insights=["Added PyJWT dependency"]),
    ]
    summary = aggregate_results(results)
    print(f"  Agents: {summary['total_agents']}, "
          f"all succeeded: {summary['all_succeeded']}")
    print(f"  Files found:    {summary['unique_files_found']}")
    print(f"  Files modified: {summary['files_modified']}")
    print(f"  By type: {summary['agents_by_type']}")
    print(f"  Insights:")
    for insight in summary["insights"]:
        print(f"    - {insight}")


if __name__ == "__main__":
    print("=== Exercise 1: Subagent Selection ===")
    exercise_1()

    print("\n=== Exercise 2: Task Decomposition ===")
    exercise_2()

    print("\n=== Exercise 3: Result Aggregation ===")
    exercise_3()

    print("\nAll exercises completed!")

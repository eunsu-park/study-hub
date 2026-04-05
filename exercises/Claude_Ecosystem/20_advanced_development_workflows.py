"""
Exercises for Lesson 20: Advanced Development Workflows
Topic: Claude_Ecosystem

Solutions to practice problems from the lesson.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# === Exercise 1: TDD Workflow Engine ===
# Problem: Model a Test-Driven Development workflow where Claude
#   writes tests first, then implements code to pass them.

class TDDPhase(Enum):
    RED = "red"       # Write failing test
    GREEN = "green"   # Write minimal code to pass
    REFACTOR = "refactor"  # Clean up code


@dataclass
class TDDCycle:
    """A single Red-Green-Refactor TDD cycle."""
    test_name: str
    phase: TDDPhase = TDDPhase.RED
    test_code: str = ""
    impl_code: str = ""
    refactored: bool = False
    test_passes: bool = False


class TDDWorkflow:
    """Manages TDD workflow cycles."""

    def __init__(self) -> None:
        self.cycles: list[TDDCycle] = []

    def write_test(self, test_name: str, test_code: str) -> int:
        """RED phase: write a failing test."""
        cycle = TDDCycle(test_name=test_name, test_code=test_code,
                         phase=TDDPhase.RED)
        self.cycles.append(cycle)
        return len(self.cycles) - 1

    def implement(self, cycle_idx: int, impl_code: str) -> None:
        """GREEN phase: write minimal implementation."""
        cycle = self.cycles[cycle_idx]
        cycle.impl_code = impl_code
        cycle.phase = TDDPhase.GREEN
        cycle.test_passes = True  # simulate passing

    def refactor(self, cycle_idx: int, impl_code: str) -> None:
        """REFACTOR phase: improve code while keeping tests green."""
        cycle = self.cycles[cycle_idx]
        cycle.impl_code = impl_code
        cycle.phase = TDDPhase.REFACTOR
        cycle.refactored = True

    def summary(self) -> dict[str, Any]:
        return {
            "total_cycles": len(self.cycles),
            "completed": sum(1 for c in self.cycles
                             if c.phase == TDDPhase.REFACTOR),
            "passing": sum(1 for c in self.cycles if c.test_passes),
            "phases": {p.value: sum(1 for c in self.cycles if c.phase == p)
                       for p in TDDPhase},
        }


def exercise_1():
    """Demonstrate TDD workflow."""
    tdd = TDDWorkflow()

    # Cycle 1: Add function
    idx = tdd.write_test("test_add",
                         "def test_add(): assert add(1, 2) == 3")
    tdd.implement(idx, "def add(a, b): return a + b")
    tdd.refactor(idx, "def add(a: int, b: int) -> int:\n    return a + b")

    # Cycle 2: Subtract function
    idx = tdd.write_test("test_subtract",
                         "def test_subtract(): assert subtract(5, 3) == 2")
    tdd.implement(idx, "def subtract(a, b): return a - b")

    # Cycle 3: Still in RED
    tdd.write_test("test_multiply",
                   "def test_multiply(): assert multiply(3, 4) == 12")

    stats = tdd.summary()
    print(f"  Cycles: {stats['total_cycles']}, "
          f"completed: {stats['completed']}, "
          f"passing: {stats['passing']}")
    print(f"  Phases: {stats['phases']}")
    for c in tdd.cycles:
        print(f"  [{c.phase.value:<9}] {c.test_name}: "
              f"pass={c.test_passes}, refactored={c.refactored}")


# === Exercise 2: Multi-File Refactoring Tracker ===
# Problem: Track changes across multiple files during a refactoring
#   operation, with dependency analysis.

@dataclass
class FileChange:
    """A change made to a file during refactoring."""
    file_path: str
    change_type: str  # "modify", "create", "delete", "rename"
    description: str
    depends_on: list[str] = field(default_factory=list)


class RefactoringPlan:
    """Plan and track a multi-file refactoring operation."""

    def __init__(self, description: str) -> None:
        self.description = description
        self.changes: list[FileChange] = []
        self._completed: set[str] = set()

    def add_change(self, change: FileChange) -> None:
        self.changes.append(change)

    def get_safe_order(self) -> list[FileChange]:
        """Topological sort: dependencies first."""
        ordered: list[FileChange] = []
        visited: set[str] = set()

        def visit(path: str) -> None:
            if path in visited:
                return
            visited.add(path)
            change = next((c for c in self.changes if c.file_path == path), None)
            if change:
                for dep in change.depends_on:
                    visit(dep)
                ordered.append(change)

        for change in self.changes:
            visit(change.file_path)
        return ordered

    def complete(self, file_path: str) -> bool:
        change = next((c for c in self.changes if c.file_path == file_path), None)
        if not change:
            return False
        unmet = [d for d in change.depends_on if d not in self._completed]
        if unmet:
            return False
        self._completed.add(file_path)
        return True

    def progress(self) -> dict[str, Any]:
        return {
            "total": len(self.changes),
            "completed": len(self._completed),
            "remaining": len(self.changes) - len(self._completed),
            "pct": round(len(self._completed) / max(len(self.changes), 1) * 100),
        }


def exercise_2():
    """Demonstrate multi-file refactoring tracking."""
    plan = RefactoringPlan("Extract auth module from monolith")
    plan.add_change(FileChange("src/auth/types.py", "create",
                               "Define auth type interfaces"))
    plan.add_change(FileChange("src/auth/session.py", "create",
                               "Extract session management",
                               depends_on=["src/auth/types.py"]))
    plan.add_change(FileChange("src/auth/token.py", "create",
                               "Extract token validation",
                               depends_on=["src/auth/types.py"]))
    plan.add_change(FileChange("src/app.py", "modify",
                               "Update imports to use new auth module",
                               depends_on=["src/auth/session.py",
                                           "src/auth/token.py"]))
    plan.add_change(FileChange("tests/test_auth.py", "modify",
                               "Update test imports",
                               depends_on=["src/auth/session.py"]))

    print(f"  Plan: {plan.description}")
    print(f"  Safe execution order:")
    for i, change in enumerate(plan.get_safe_order()):
        deps = f" (after: {change.depends_on})" if change.depends_on else ""
        print(f"    {i + 1}. [{change.change_type}] {change.file_path}"
              f"{deps}")

    # Execute in order
    for change in plan.get_safe_order():
        success = plan.complete(change.file_path)
        status = "done" if success else "BLOCKED"
        print(f"  {change.file_path}: {status}")

    print(f"  Progress: {plan.progress()}")


# === Exercise 3: CI/CD Integration Checker ===
# Problem: Validate that Claude Code changes are compatible with
#   the project's CI/CD pipeline.

@dataclass
class CICheck:
    """A CI/CD pipeline check."""
    name: str
    command: str
    required: bool = True
    timeout_seconds: int = 120


@dataclass
class CIResult:
    name: str
    passed: bool
    output: str
    duration_seconds: int


class CIPipeline:
    """Simulates a CI/CD pipeline for validating changes."""

    def __init__(self) -> None:
        self.checks: list[CICheck] = []

    def add_check(self, check: CICheck) -> None:
        self.checks.append(check)

    def simulate_run(self, changed_files: list[str]) -> dict[str, Any]:
        """Simulate running the CI pipeline."""
        results: list[CIResult] = []
        all_passed = True

        for check in self.checks:
            # Simulate: lint fails if .py files changed, tests pass
            passed = True
            output = "OK"
            if "lint" in check.name.lower():
                if any(f.endswith(".py") for f in changed_files):
                    passed = True
                    output = f"Checked {len(changed_files)} files"
            elif "test" in check.name.lower():
                passed = True
                output = "All tests passed"
            elif "build" in check.name.lower():
                passed = True
                output = "Build successful"

            results.append(CIResult(check.name, passed, output, 30))
            if not passed and check.required:
                all_passed = False

        return {
            "passed": all_passed,
            "checks": len(results),
            "results": [
                {"name": r.name, "passed": r.passed, "output": r.output}
                for r in results
            ],
        }


def exercise_3():
    """Demonstrate CI/CD pipeline simulation."""
    pipeline = CIPipeline()
    pipeline.add_check(CICheck("lint", "ruff check src/"))
    pipeline.add_check(CICheck("type-check", "mypy src/"))
    pipeline.add_check(CICheck("unit-tests", "pytest tests/unit/"))
    pipeline.add_check(CICheck("integration-tests", "pytest tests/integration/",
                               required=False))
    pipeline.add_check(CICheck("build", "python -m build"))

    changed = ["src/auth/session.py", "src/auth/token.py", "tests/test_auth.py"]
    result = pipeline.simulate_run(changed)

    print(f"  Pipeline: {'PASSED' if result['passed'] else 'FAILED'}")
    print(f"  Changed files: {changed}")
    for r in result["results"]:
        icon = "pass" if r["passed"] else "FAIL"
        print(f"    [{icon}] {r['name']}: {r['output']}")


if __name__ == "__main__":
    print("=== Exercise 1: TDD Workflow ===")
    exercise_1()

    print("\n=== Exercise 2: Multi-File Refactoring ===")
    exercise_2()

    print("\n=== Exercise 3: CI/CD Pipeline ===")
    exercise_3()

    print("\nAll exercises completed!")

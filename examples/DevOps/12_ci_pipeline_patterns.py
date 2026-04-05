#!/usr/bin/env python3
"""Example: CI Pipeline Patterns — Build DAGs, Caching, and Parallelism

Demonstrates CI pipeline modeling beyond a single provider: directed acyclic
graph (DAG) scheduling, build artifact caching, parallel stage execution,
and pipeline-as-code validation.
Related lesson: 03_CI_Fundamentals.md
"""

# =============================================================================
# WHY MODEL CI PIPELINES IN CODE?
# Understanding pipeline topology (DAGs, fan-out/fan-in, matrix builds) helps
# you design faster, more reliable CI regardless of the provider (Jenkins,
# GitLab CI, CircleCI, Buildkite, etc.). This example builds a provider-
# agnostic pipeline engine that schedules stages respecting dependencies.
# =============================================================================

import time
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any


# =============================================================================
# 1. PIPELINE DATA MODEL
# =============================================================================

class StageStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Stage:
    """A single CI stage (e.g., lint, test, build, deploy)."""
    name: str
    commands: list[str]
    depends_on: list[str] = field(default_factory=list)
    timeout_seconds: int = 300
    retry_count: int = 0
    cache_key: str = ""
    status: StageStatus = StageStatus.PENDING
    duration_ms: float = 0.0


@dataclass
class Pipeline:
    """A CI pipeline composed of stages with dependency edges."""
    name: str
    stages: dict[str, Stage] = field(default_factory=dict)
    trigger: str = "push"

    def add_stage(self, stage: Stage) -> None:
        self.stages[stage.name] = stage

    def get_execution_order(self) -> list[list[str]]:
        """Topological sort returning stages grouped by execution wave."""
        in_degree: dict[str, int] = {s: 0 for s in self.stages}
        graph: dict[str, list[str]] = defaultdict(list)
        for name, stage in self.stages.items():
            for dep in stage.depends_on:
                graph[dep].append(name)
                in_degree[name] += 1

        waves: list[list[str]] = []
        queue = deque(n for n, d in in_degree.items() if d == 0)
        while queue:
            wave = list(queue)
            waves.append(wave)
            next_queue: deque[str] = deque()
            for node in wave:
                for child in graph[node]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_queue.append(child)
            queue = next_queue

        # Detect cycles
        scheduled = sum(len(w) for w in waves)
        if scheduled != len(self.stages):
            unscheduled = set(self.stages) - {s for w in waves for s in w}
            raise ValueError(f"Cycle detected involving: {unscheduled}")
        return waves


# =============================================================================
# 2. PIPELINE EXECUTOR (SIMULATED)
# =============================================================================

@dataclass
class BuildCache:
    """Simple content-addressable build cache."""
    store: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def compute_key(inputs: list[str]) -> str:
        h = hashlib.sha256()
        for item in sorted(inputs):
            h.update(item.encode())
        return h.hexdigest()[:12]

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def put(self, key: str, artifact: str) -> None:
        self.store[key] = artifact


def run_pipeline(pipeline: Pipeline, cache: BuildCache | None = None) -> dict[str, StageStatus]:
    """Execute a pipeline respecting DAG order with optional caching."""
    results: dict[str, StageStatus] = {}
    waves = pipeline.get_execution_order()

    print(f"Pipeline '{pipeline.name}' — {len(pipeline.stages)} stages, "
          f"{len(waves)} waves")
    print("-" * 50)

    for wave_idx, wave in enumerate(waves):
        print(f"\nWave {wave_idx + 1} (parallel): {wave}")
        for stage_name in wave:
            stage = pipeline.stages[stage_name]

            # Check if all dependencies passed
            deps_ok = all(results.get(d) == StageStatus.PASSED for d in stage.depends_on)
            if not deps_ok:
                stage.status = StageStatus.SKIPPED
                results[stage_name] = StageStatus.SKIPPED
                print(f"  [{stage_name}] SKIPPED (dependency failed)")
                continue

            # Check cache
            if cache and stage.cache_key:
                cached = cache.get(stage.cache_key)
                if cached:
                    stage.status = StageStatus.PASSED
                    results[stage_name] = StageStatus.PASSED
                    print(f"  [{stage_name}] CACHE HIT ({stage.cache_key})")
                    continue

            # Simulate execution
            stage.status = StageStatus.RUNNING
            start = time.monotonic()
            # In real CI, commands would be executed here
            success = "FAIL" not in " ".join(stage.commands).upper()
            stage.duration_ms = (time.monotonic() - start) * 1000

            if success:
                stage.status = StageStatus.PASSED
                if cache and stage.cache_key:
                    cache.put(stage.cache_key, f"artifact-{stage_name}")
            else:
                stage.status = StageStatus.FAILED

            results[stage_name] = stage.status
            print(f"  [{stage_name}] {stage.status.value.upper()} "
                  f"({stage.duration_ms:.1f}ms)")

    return results


# =============================================================================
# 3. PIPELINE VALIDATION
# =============================================================================

def validate_pipeline(pipeline: Pipeline) -> list[str]:
    """Validate pipeline structure for common mistakes."""
    errors: list[str] = []

    # Check for undefined dependencies
    for name, stage in pipeline.stages.items():
        for dep in stage.depends_on:
            if dep not in pipeline.stages:
                errors.append(f"Stage '{name}' depends on undefined stage '{dep}'")

    # Check for cycles (via execution order)
    try:
        pipeline.get_execution_order()
    except ValueError as e:
        errors.append(str(e))

    # Check for stages with no commands
    for name, stage in pipeline.stages.items():
        if not stage.commands:
            errors.append(f"Stage '{name}' has no commands")

    return errors


# =============================================================================
# 4. MATRIX BUILD EXPANDER
# =============================================================================

def expand_matrix(stage_template: dict, matrix: dict[str, list]) -> list[Stage]:
    """Expand a stage template across a build matrix (e.g., OS x Python version)."""
    import itertools
    keys = list(matrix.keys())
    combos = list(itertools.product(*matrix.values()))
    stages = []
    for combo in combos:
        params = dict(zip(keys, combo))
        suffix = "-".join(str(v) for v in combo)
        name = f"{stage_template['name']}-{suffix}"
        commands = [
            cmd.format(**params) for cmd in stage_template["commands"]
        ]
        stages.append(Stage(
            name=name,
            commands=commands,
            depends_on=stage_template.get("depends_on", []),
        ))
    return stages


# =============================================================================
# 5. DEMO
# =============================================================================

if __name__ == "__main__":
    # --- Build a realistic CI pipeline ---
    p = Pipeline(name="backend-ci", trigger="pull_request")

    p.add_stage(Stage(name="checkout", commands=["git checkout $SHA"]))
    p.add_stage(Stage(name="install", commands=["pip install -r requirements.txt"],
                      depends_on=["checkout"], cache_key="deps-abc123"))
    p.add_stage(Stage(name="lint", commands=["ruff check ."],
                      depends_on=["install"]))
    p.add_stage(Stage(name="typecheck", commands=["mypy src/"],
                      depends_on=["install"]))
    p.add_stage(Stage(name="unit-test", commands=["pytest tests/unit/"],
                      depends_on=["lint", "typecheck"]))
    p.add_stage(Stage(name="integration-test", commands=["pytest tests/integration/"],
                      depends_on=["lint", "typecheck"]))
    p.add_stage(Stage(name="build", commands=["python -m build"],
                      depends_on=["unit-test", "integration-test"]))

    # Validate and run
    errors = validate_pipeline(p)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
    else:
        cache = BuildCache()
        results = run_pipeline(p, cache)
        print(f"\nFinal results: { {k: v.value for k, v in results.items()} }")

    # --- Matrix expansion ---
    print(f"\n{'=' * 50}")
    print("Matrix Expansion")
    print("=" * 50)
    matrix_stages = expand_matrix(
        {"name": "test", "commands": ["python{python} -m pytest"], "depends_on": ["install"]},
        {"os": ["ubuntu", "macos"], "python": ["3.11", "3.12"]},
    )
    for s in matrix_stages:
        print(f"  {s.name}: {s.commands}")

#!/usr/bin/env python3
"""Example: GitHub Actions Workflow Generator & Validator

Demonstrates programmatic generation and validation of GitHub Actions
workflow YAML files. Covers job dependencies, matrix builds, caching,
artifact uploads, and schema validation.
Related lesson: 03_CI_CD_Fundamentals.md
"""

# =============================================================================
# WHY GENERATE WORKFLOWS PROGRAMMATICALLY?
# As projects scale, maintaining dozens of hand-written YAML workflows becomes
# error-prone. Generating them from a Python model lets you enforce standards,
# share common steps, and validate structure before committing.
# =============================================================================

import yaml
import json
import copy
from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# 1. WORKFLOW DATA MODEL
# =============================================================================
# Define reusable building blocks for GitHub Actions workflows.

@dataclass
class Step:
    """A single step within a GitHub Actions job."""
    name: str
    uses: str | None = None
    run: str | None = None
    with_: dict[str, Any] | None = None
    env: dict[str, str] | None = None
    id_: str | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"name": self.name}
        if self.id_:
            d["id"] = self.id_
        if self.uses:
            d["uses"] = self.uses
        if self.run:
            d["run"] = self.run
        if self.with_:
            d["with"] = self.with_
        if self.env:
            d["env"] = self.env
        return d


@dataclass
class Job:
    """A job within a workflow."""
    name: str
    runs_on: str = "ubuntu-latest"
    needs: list[str] | None = None
    steps: list[Step] = field(default_factory=list)
    strategy: dict | None = None
    env: dict[str, str] | None = None
    timeout_minutes: int = 30

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "name": self.name,
            "runs-on": self.runs_on,
            "timeout-minutes": self.timeout_minutes,
            "steps": [s.to_dict() for s in self.steps],
        }
        if self.needs:
            d["needs"] = self.needs
        if self.strategy:
            d["strategy"] = self.strategy
        if self.env:
            d["env"] = self.env
        return d


@dataclass
class Workflow:
    """A complete GitHub Actions workflow."""
    name: str
    on: dict[str, Any]
    jobs: dict[str, Job] = field(default_factory=dict)
    env: dict[str, str] | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "name": self.name,
            "on": self.on,
            "jobs": {k: v.to_dict() for k, v in self.jobs.items()},
        }
        if self.env:
            d["env"] = self.env
        return d

    def to_yaml(self) -> str:
        return yaml.dump(
            self.to_dict(),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


# =============================================================================
# 2. COMMON STEP TEMPLATES
# =============================================================================
# Reusable steps that appear in almost every CI workflow.

def checkout_step(fetch_depth: int = 1) -> Step:
    return Step(
        name="Checkout code",
        uses="actions/checkout@v4",
        with_={"fetch-depth": fetch_depth},
    )


def setup_python_step(version: str = "3.12") -> Step:
    return Step(
        name=f"Set up Python {version}",
        uses="actions/setup-python@v5",
        with_={"python-version": version},
    )


def pip_cache_step() -> Step:
    return Step(
        name="Cache pip dependencies",
        uses="actions/cache@v4",
        with_={
            "path": "~/.cache/pip",
            "key": "${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}",
            "restore-keys": "${{ runner.os }}-pip-",
        },
    )


def install_deps_step(requirements: str = "requirements.txt") -> Step:
    return Step(
        name="Install dependencies",
        run=f"pip install -r {requirements}",
    )


# =============================================================================
# 3. WORKFLOW GENERATORS
# =============================================================================

def generate_ci_workflow() -> Workflow:
    """Generate a standard Python CI workflow with lint, test, and build."""
    lint_job = Job(
        name="Lint",
        steps=[
            checkout_step(),
            setup_python_step(),
            pip_cache_step(),
            Step(name="Install linters", run="pip install ruff mypy"),
            Step(name="Run ruff", run="ruff check ."),
            Step(name="Run mypy", run="mypy src/"),
        ],
    )

    test_job = Job(
        name="Test",
        needs=["lint"],
        strategy={
            "matrix": {
                "python-version": ["3.10", "3.11", "3.12"],
                "os": ["ubuntu-latest", "macos-latest"],
            },
            "fail-fast": False,
        },
        steps=[
            checkout_step(),
            Step(
                name="Set up Python ${{ matrix.python-version }}",
                uses="actions/setup-python@v5",
                with_={"python-version": "${{ matrix.python-version }}"},
            ),
            pip_cache_step(),
            install_deps_step(),
            Step(
                name="Run tests with coverage",
                run="pytest --cov=src --cov-report=xml -v",
            ),
            Step(
                name="Upload coverage",
                uses="actions/upload-artifact@v4",
                with_={"name": "coverage-${{ matrix.python-version }}-${{ matrix.os }}", "path": "coverage.xml"},
            ),
        ],
    )

    build_job = Job(
        name="Build",
        needs=["test"],
        steps=[
            checkout_step(),
            setup_python_step(),
            Step(name="Build package", run="python -m build"),
            Step(
                name="Upload dist",
                uses="actions/upload-artifact@v4",
                with_={"name": "dist", "path": "dist/"},
            ),
        ],
    )

    return Workflow(
        name="CI Pipeline",
        on={
            "push": {"branches": ["main", "develop"]},
            "pull_request": {"branches": ["main"]},
        },
        jobs={"lint": lint_job, "test": test_job, "build": build_job},
    )


def generate_docker_publish_workflow(image_name: str = "myapp") -> Workflow:
    """Generate a Docker build + push workflow for GHCR."""
    build_job = Job(
        name="Build and Push",
        steps=[
            checkout_step(),
            Step(
                name="Set up Docker Buildx",
                uses="docker/setup-buildx-action@v3",
            ),
            Step(
                name="Log in to GHCR",
                uses="docker/login-action@v3",
                with_={
                    "registry": "ghcr.io",
                    "username": "${{ github.actor }}",
                    "password": "${{ secrets.GITHUB_TOKEN }}",
                },
            ),
            Step(
                name="Extract metadata",
                id_="meta",
                uses="docker/metadata-action@v5",
                with_={
                    "images": f"ghcr.io/${{{{ github.repository_owner }}}}/{image_name}",
                    "tags": (
                        "type=sha\n"
                        "type=ref,event=branch\n"
                        "type=semver,pattern={{version}}\n"
                        "type=semver,pattern={{major}}.{{minor}}"
                    ),
                },
            ),
            Step(
                name="Build and push",
                uses="docker/build-push-action@v5",
                with_={
                    "context": ".",
                    "push": True,
                    "tags": "${{ steps.meta.outputs.tags }}",
                    "labels": "${{ steps.meta.outputs.labels }}",
                    "cache-from": "type=gha",
                    "cache-to": "type=gha,mode=max",
                    "platforms": "linux/amd64,linux/arm64",
                },
            ),
        ],
    )

    return Workflow(
        name="Docker Publish",
        on={
            "push": {"branches": ["main"], "tags": ["v*"]},
        },
        jobs={"build-and-push": build_job},
    )


# =============================================================================
# 4. WORKFLOW VALIDATION
# =============================================================================

REQUIRED_FIELDS = {"name", "on", "jobs"}
VALID_EVENTS = {
    "push", "pull_request", "workflow_dispatch", "schedule",
    "release", "issues", "issue_comment", "workflow_call",
}


def validate_workflow(workflow_dict: dict) -> list[str]:
    """Validate a workflow dictionary against common rules.

    Returns a list of error messages (empty = valid).
    """
    errors: list[str] = []

    # Check required top-level fields
    for f in REQUIRED_FIELDS:
        if f not in workflow_dict:
            errors.append(f"Missing required field: '{f}'")

    # Validate trigger events
    on_config = workflow_dict.get("on", {})
    if isinstance(on_config, dict):
        for event in on_config:
            if event not in VALID_EVENTS:
                errors.append(f"Unknown trigger event: '{event}'")

    # Validate jobs
    jobs = workflow_dict.get("jobs", {})
    if not jobs:
        errors.append("Workflow must have at least one job")

    job_names = set(jobs.keys())
    for job_id, job_config in jobs.items():
        # Every job must have steps
        if "steps" not in job_config:
            errors.append(f"Job '{job_id}' has no steps")

        # Validate 'needs' references
        needs = job_config.get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        for dep in needs:
            if dep not in job_names:
                errors.append(
                    f"Job '{job_id}' depends on unknown job '{dep}'"
                )

        # Check for steps without 'uses' or 'run'
        for i, step in enumerate(job_config.get("steps", [])):
            if "uses" not in step and "run" not in step:
                errors.append(
                    f"Job '{job_id}', step {i} ('{step.get('name', 'unnamed')}')"
                    " must have either 'uses' or 'run'"
                )

    return errors


def detect_circular_dependencies(jobs: dict) -> list[str]:
    """Detect circular job dependencies using DFS."""
    errors: list[str] = []
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {j: WHITE for j in jobs}

    def dfs(node: str, path: list[str]) -> None:
        color[node] = GRAY
        path.append(node)
        needs = jobs[node].get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        for dep in needs:
            if dep not in color:
                continue
            if color[dep] == GRAY:
                cycle = path[path.index(dep):]
                errors.append(
                    f"Circular dependency: {' -> '.join(cycle + [dep])}"
                )
            elif color[dep] == WHITE:
                dfs(dep, path)
        path.pop()
        color[node] = BLACK

    for job_id in jobs:
        if color[job_id] == WHITE:
            dfs(job_id, [])

    return errors


# =============================================================================
# 5. DEMO
# =============================================================================

if __name__ == "__main__":
    # --- Generate CI workflow ---
    ci = generate_ci_workflow()
    ci_yaml = ci.to_yaml()
    print("=" * 70)
    print("Generated CI Workflow (.github/workflows/ci.yml)")
    print("=" * 70)
    print(ci_yaml)

    # --- Validate ---
    ci_dict = ci.to_dict()
    errors = validate_workflow(ci_dict)
    errors.extend(detect_circular_dependencies(ci_dict["jobs"]))
    if errors:
        print("Validation errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("Validation: PASSED (no errors)\n")

    # --- Generate Docker publish workflow ---
    docker_wf = generate_docker_publish_workflow("my-service")
    print("=" * 70)
    print("Generated Docker Publish Workflow")
    print("=" * 70)
    print(docker_wf.to_yaml())

    # --- Validate a BAD workflow to show error detection ---
    bad_workflow = {
        "name": "Bad Workflow",
        "on": {"unknown_event": {}},
        "jobs": {
            "build": {
                "name": "Build",
                "steps": [{"name": "Empty step"}],  # no uses or run
                "needs": ["nonexistent"],
            }
        },
    }
    print("=" * 70)
    print("Validating intentionally bad workflow")
    print("=" * 70)
    bad_errors = validate_workflow(bad_workflow)
    for e in bad_errors:
        print(f"  ERROR: {e}")

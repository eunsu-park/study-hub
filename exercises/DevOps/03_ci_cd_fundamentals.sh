#!/bin/bash
# Exercises for Lesson 03: CI/CD Fundamentals
# Topic: DevOps
# Solutions to practice problems from the lesson.

# === Exercise 1: Design a CI Pipeline ===
# Problem: Write a GitHub Actions workflow for a Python project that runs
# lint, test (matrix), and build stages with proper caching and artifacts.
exercise_1() {
    echo "=== Exercise 1: Design a CI Pipeline ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_DEFAULT: "3.12"

jobs:
  lint:
    name: Lint & Type Check
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_DEFAULT }}
      - uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-lint-${{ hashFiles('requirements-dev.txt') }}
      - run: pip install ruff mypy
      - run: ruff check --output-format=github .
      - run: mypy src/ --ignore-missing-imports

  test:
    name: Test (Python ${{ matrix.python-version }})
    needs: lint
    runs-on: ubuntu-latest
    timeout-minutes: 15
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
      fail-fast: false
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ matrix.python-version }}-${{ hashFiles('requirements*.txt') }}
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: pytest --cov=src --cov-report=xml --junitxml=results.xml -v
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-results-${{ matrix.python-version }}
          path: |
            coverage.xml
            results.xml

  build:
    name: Build Package
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_DEFAULT }}
      - run: pip install build
      - run: python -m build
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

# Key design decisions:
# 1. lint runs first (fast fail) — no point running tests if code doesn't lint
# 2. test uses matrix strategy — catches version-specific regressions
# 3. fail-fast: false — see ALL failures, not just the first
# 4. Caching — pip cache saves ~30s per job on repeated runs
# 5. Artifacts — coverage and test results available for review
SOLUTION
}

# === Exercise 2: Pipeline Anti-Patterns ===
# Problem: Identify and fix the anti-patterns in a given CI pipeline.
exercise_2() {
    echo "=== Exercise 2: Pipeline Anti-Patterns ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# ANTI-PATTERN 1: Monolithic pipeline (everything in one job)
# BAD:
#   jobs:
#     ci:
#       steps:
#         - run: ruff check .
#         - run: pytest
#         - run: docker build .
#         - run: docker push
#         - run: kubectl apply -f deploy.yaml
# Problem: If tests fail at minute 20, you wasted 10 minutes building Docker
# Fix: Split into stages with dependencies

# ANTI-PATTERN 2: No caching
# BAD:
#   - run: pip install -r requirements.txt  # Downloads 500MB every time
# Fix: Cache dependency directories
#   - uses: actions/cache@v4
#     with:
#       path: ~/.cache/pip
#       key: pip-${{ hashFiles('requirements.txt') }}

# ANTI-PATTERN 3: Secrets in logs
# BAD:
#   - run: echo "Deploying with key ${{ secrets.API_KEY }}"
# Fix: Never echo secrets; use environment variables
#   - run: deploy.sh
#     env:
#       API_KEY: ${{ secrets.API_KEY }}

# ANTI-PATTERN 4: No timeout
# BAD: A hung test blocks the runner indefinitely
# Fix: Always set timeout-minutes at job level

# ANTI-PATTERN 5: Testing only on one OS/version
# BAD: Works on ubuntu + python 3.12, breaks on macos + python 3.10
# Fix: Use matrix strategy for critical combinations

# ANTI-PATTERN 6: No artifact retention
# BAD: Test results disappear after the run
# Fix: Upload test results, coverage, and build artifacts

# Checklist for a healthy CI pipeline:
pipeline_checklist = [
    "Runs in under 10 minutes for fast feedback",
    "Fails fast (lint before test before build)",
    "Uses caching for dependencies",
    "Matrix tests across supported versions",
    "Uploads test results and coverage as artifacts",
    "Has timeout limits on all jobs",
    "Never exposes secrets in logs",
    "Runs on PR and push to main",
]
for item in pipeline_checklist:
    print(f"  [x] {item}")
SOLUTION
}

# === Exercise 3: Artifact Management ===
# Problem: Design an artifact versioning and retention scheme
# for a microservices project.
exercise_3() {
    echo "=== Exercise 3: Artifact Management ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class Artifact:
    name: str
    version: str
    artifact_type: str       # docker, pypi, npm, binary
    size_mb: float
    created_at: datetime
    git_sha: str
    branch: str
    metadata: dict

    @property
    def tag(self) -> str:
        """Generate artifact tag following the naming convention."""
        # Pattern: <service>:<semver>-<short-sha>
        return f"{self.name}:{self.version}-{self.git_sha[:7]}"

    @property
    def is_release(self) -> bool:
        return self.branch == "main" and not self.version.endswith("-dev")

# Retention policy
RETENTION_POLICY = {
    "release":  {"keep_count": None, "max_age_days": 365},  # Keep all releases for 1 year
    "staging":  {"keep_count": 10, "max_age_days": 30},     # Last 10 staging builds
    "develop":  {"keep_count": 5, "max_age_days": 14},      # Last 5 dev builds
    "feature":  {"keep_count": 2, "max_age_days": 7},       # Last 2 feature builds
}

def should_retain(artifact: Artifact, policy: dict, existing_count: int) -> bool:
    """Determine if an artifact should be retained based on policy."""
    age = datetime.now() - artifact.created_at
    max_age = timedelta(days=policy["max_age_days"])
    keep_count = policy.get("keep_count")

    if age > max_age:
        return False
    if keep_count is not None and existing_count > keep_count:
        return False
    return True

# Tagging convention examples:
# Production: order-api:1.2.3           (semver only)
# Staging:    order-api:1.2.3-rc.1      (release candidate)
# Develop:    order-api:1.2.3-dev.abc1234 (dev + short SHA)
# Feature:    order-api:0.0.0-feat-search.abc1234

print("Artifact Naming Convention:")
print("  Production:  <service>:<major>.<minor>.<patch>")
print("  Staging:     <service>:<version>-rc.<n>")
print("  Development: <service>:<version>-dev.<sha7>")
print("  Feature:     <service>:0.0.0-<branch>.<sha7>")
print()
print("Retention Policy:")
for env, policy in RETENTION_POLICY.items():
    keep = policy["keep_count"] or "unlimited"
    print(f"  {env:10s}: keep={keep}, max_age={policy['max_age_days']}d")
SOLUTION
}

# === Exercise 4: Build Optimization ===
# Problem: A CI pipeline takes 25 minutes. Identify optimization
# strategies and estimate the time savings.
exercise_4() {
    echo "=== Exercise 4: Build Optimization ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
optimizations = [
    {
        "technique": "Dependency caching",
        "before_min": 4.0,
        "after_min": 0.5,
        "description": "Cache pip/npm/maven downloads between runs",
    },
    {
        "technique": "Parallel test execution",
        "before_min": 8.0,
        "after_min": 3.0,
        "description": "Split tests across 3 parallel runners (pytest-xdist)",
    },
    {
        "technique": "Docker layer caching",
        "before_min": 5.0,
        "after_min": 1.5,
        "description": "Cache Docker build layers (BuildKit, GitHub cache)",
    },
    {
        "technique": "Incremental builds",
        "before_min": 3.0,
        "after_min": 1.0,
        "description": "Only rebuild changed modules (Turborepo, Nx, Bazel)",
    },
    {
        "technique": "Skip unchanged paths",
        "before_min": 5.0,
        "after_min": 0.0,
        "description": "Use path filters to skip unrelated CI jobs",
    },
]

total_before = sum(o["before_min"] for o in optimizations)
total_after = sum(o["after_min"] for o in optimizations)

print(f"{'Technique':<28} {'Before':>7} {'After':>7} {'Saved':>7}")
print("-" * 55)
for o in optimizations:
    saved = o["before_min"] - o["after_min"]
    print(f"{o['technique']:<28} {o['before_min']:>5.1f}m {o['after_min']:>5.1f}m {saved:>5.1f}m")
print("-" * 55)
print(f"{'Total':<28} {total_before:>5.1f}m {total_after:>5.1f}m "
      f"{total_before - total_after:>5.1f}m")
print(f"\nSpeedup: {total_before:.0f}min -> {total_after:.0f}min "
      f"({(1 - total_after/total_before):.0%} faster)")

# Additional tips:
# - Use smaller base images (alpine, slim) for faster pulls
# - Run slow integration tests only on main, not on every PR
# - Use test impact analysis to run only tests affected by changes
# - Pre-build CI images with common dependencies baked in
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 03: CI/CD Fundamentals"
echo "====================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4

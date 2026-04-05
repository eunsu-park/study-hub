#!/usr/bin/env python3
"""Example: CI/CD Testing

Demonstrates test pipeline design patterns, parallel test execution,
test splitting strategies, and CI-aware test configuration.
Related lesson: 13_CI_CD_Testing.md
"""

# =============================================================================
# WHY CI/CD TESTING MATTERS
#
# Tests that work locally but fail in CI — or that take 45 minutes in a
# pipeline — undermine the feedback loop that makes CI valuable.
#
# Key principles:
#   1. Fast feedback — split and parallelize tests
#   2. Deterministic — no flaky tests; same input = same result
#   3. Isolated — no shared state between test runs
#   4. Observable — clear reporting of what failed and why
# =============================================================================

import pytest
import os
import hashlib
import time
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# CI ENVIRONMENT DETECTION
# =============================================================================

def is_ci() -> bool:
    """Detect if running in a CI environment.
    Most CI systems set the CI environment variable."""
    return os.environ.get("CI", "").lower() in ("true", "1", "yes")


def get_ci_provider() -> Optional[str]:
    """Identify the CI provider for provider-specific behavior."""
    if os.environ.get("GITHUB_ACTIONS"):
        return "github_actions"
    if os.environ.get("GITLAB_CI"):
        return "gitlab_ci"
    if os.environ.get("JENKINS_URL"):
        return "jenkins"
    if os.environ.get("CIRCLECI"):
        return "circleci"
    return None


# =============================================================================
# TEST SPLITTING — DISTRIBUTE TESTS ACROSS PARALLEL WORKERS
# =============================================================================

@dataclass
class TestItem:
    """Represents a single test for splitting purposes."""
    name: str
    file_path: str
    estimated_duration_ms: float = 0.0


class TestSplitter:
    """Split tests across N parallel workers for CI pipelines.

    Strategies:
    - Round-robin: simple, even count distribution
    - Duration-based: balance by estimated runtime (requires timing data)
    - File-hash: deterministic split by file path hash
    """

    @staticmethod
    def split_round_robin(tests: list[TestItem], num_workers: int) -> list[list[TestItem]]:
        """Distribute tests evenly by count. Simple but ignores duration."""
        buckets: list[list[TestItem]] = [[] for _ in range(num_workers)]
        for i, test in enumerate(tests):
            buckets[i % num_workers].append(test)
        return buckets

    @staticmethod
    def split_by_duration(tests: list[TestItem], num_workers: int) -> list[list[TestItem]]:
        """Greedy algorithm: assign each test to the worker with the
        least total estimated duration. Minimizes wall-clock time."""
        buckets: list[list[TestItem]] = [[] for _ in range(num_workers)]
        totals = [0.0] * num_workers

        # Sort by duration descending — assign slowest tests first
        sorted_tests = sorted(tests, key=lambda t: t.estimated_duration_ms, reverse=True)

        for test in sorted_tests:
            # Find the worker with the least total time
            min_idx = totals.index(min(totals))
            buckets[min_idx].append(test)
            totals[min_idx] += test.estimated_duration_ms

        return buckets

    @staticmethod
    def split_by_file_hash(tests: list[TestItem], num_workers: int, worker_id: int) -> list[TestItem]:
        """Deterministic split using file path hash.
        Each worker computes its own subset — no coordination needed.
        Used by pytest-split and similar tools."""
        return [
            t for t in tests
            if int(hashlib.md5(t.file_path.encode()).hexdigest(), 16) % num_workers == worker_id
        ]


# =============================================================================
# RETRY LOGIC FOR FLAKY TESTS
# =============================================================================

class FlakyTestRetrier:
    """Retry mechanism for tests that occasionally fail due to external factors.

    Use sparingly — retries mask real bugs. Only use for genuinely
    non-deterministic scenarios (network timeouts, race conditions)."""

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries
        self.results: list[dict] = []

    def run_with_retry(self, test_func, *args) -> bool:
        """Run test up to max_retries+1 times. Return True if any attempt passes."""
        for attempt in range(self.max_retries + 1):
            try:
                test_func(*args)
                self.results.append({"attempt": attempt + 1, "passed": True})
                return True
            except AssertionError:
                self.results.append({"attempt": attempt + 1, "passed": False})
        return False


# =============================================================================
# TESTS
# =============================================================================

class TestCIDetection:
    """Verify CI environment detection works correctly."""

    def test_is_ci_returns_bool(self):
        result = is_ci()
        assert isinstance(result, bool)

    def test_get_ci_provider_returns_known_or_none(self):
        result = get_ci_provider()
        valid = {None, "github_actions", "gitlab_ci", "jenkins", "circleci"}
        assert result in valid


class TestTestSplitter:
    """Verify test splitting strategies."""

    @pytest.fixture
    def sample_tests(self):
        return [
            TestItem(f"test_{i}", f"test_file_{i}.py", estimated_duration_ms=d)
            for i, d in enumerate([100, 500, 200, 800, 150, 300, 50, 600, 250, 400])
        ]

    def test_round_robin_even_distribution(self, sample_tests):
        """Round-robin should distribute tests as evenly as possible."""
        buckets = TestSplitter.split_round_robin(sample_tests, num_workers=3)
        counts = [len(b) for b in buckets]
        # 10 tests / 3 workers: expect [4, 3, 3] distribution
        assert max(counts) - min(counts) <= 1
        # All tests must be assigned
        assert sum(counts) == len(sample_tests)

    def test_duration_based_balancing(self, sample_tests):
        """Duration-based split should minimize the max worker time."""
        buckets = TestSplitter.split_by_duration(sample_tests, num_workers=3)
        totals = [sum(t.estimated_duration_ms for t in b) for b in buckets]

        # The difference between fastest and slowest worker should be small
        # relative to the total work
        total_duration = sum(t.estimated_duration_ms for t in sample_tests)
        imbalance = max(totals) - min(totals)
        assert imbalance < total_duration * 0.3, f"Imbalance too high: {imbalance}"

    def test_file_hash_deterministic(self, sample_tests):
        """Same inputs must produce the same split every time."""
        split1 = TestSplitter.split_by_file_hash(sample_tests, num_workers=3, worker_id=0)
        split2 = TestSplitter.split_by_file_hash(sample_tests, num_workers=3, worker_id=0)
        assert [t.name for t in split1] == [t.name for t in split2]

    def test_file_hash_covers_all_tests(self, sample_tests):
        """All workers combined must cover every test exactly once."""
        all_names = set()
        for worker_id in range(3):
            split = TestSplitter.split_by_file_hash(sample_tests, num_workers=3, worker_id=worker_id)
            names = {t.name for t in split}
            assert not names & all_names, "Duplicate assignment"
            all_names |= names
        assert all_names == {t.name for t in sample_tests}


class TestRetryMechanism:
    """Verify the retry mechanism for flaky tests."""

    def test_passes_on_first_try(self):
        retrier = FlakyTestRetrier(max_retries=2)
        result = retrier.run_with_retry(lambda: None)  # always passes
        assert result is True
        assert len(retrier.results) == 1

    def test_fails_after_all_retries(self):
        retrier = FlakyTestRetrier(max_retries=2)

        def always_fails():
            raise AssertionError("flaky failure")

        result = retrier.run_with_retry(always_fails)
        assert result is False
        assert len(retrier.results) == 3  # 1 attempt + 2 retries


# =============================================================================
# CI PIPELINE CONFIGURATION EXAMPLES
# =============================================================================

GITHUB_ACTIONS_EXAMPLE = """
# === .github/workflows/test.yml ===
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
        shard: [0, 1, 2, 3]  # 4 parallel test shards
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -r requirements-test.txt
      - run: |
          pytest tests/ \\
            --splits 4 \\
            --group ${{ matrix.shard }} \\
            --splitting-algorithm least_duration \\
            -v --junitxml=results-${{ matrix.shard }}.xml
      - uses: actions/upload-artifact@v4
        with:
          name: test-results-${{ matrix.shard }}
          path: results-*.xml
"""

# =============================================================================
# RUNNING THIS FILE
# =============================================================================
# Basic run:
#   pytest 13_ci_cd_testing.py -v
#
# Parallel execution (install: pip install pytest-xdist):
#   pytest 13_ci_cd_testing.py -v -n auto
#
# Test splitting (install: pip install pytest-split):
#   pytest --splits 3 --group 0 tests/

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

#!/bin/bash
# Exercises for Lesson 11: Performance Testing
# Topic: Testing_and_QA
# Solutions to practice problems from the lesson.

# === Exercise 1: Basic Load Test with Locust ===
# Problem: Write a Locust test for a REST API with three endpoints
# (GET /users, GET /users/{id}, POST /users). Use appropriate task
# weights to simulate realistic traffic (80% reads, 20% writes).
exercise_1() {
    echo "=== Exercise 1: Basic Load Test with Locust ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from locust import HttpUser, task, between


class APIUser(HttpUser):
    """Simulates a user interacting with a REST API.

    Traffic distribution: 80% reads, 20% writes.
    - GET /users        -> weight 4 (40% of total)
    - GET /users/{id}   -> weight 4 (40% of total)
    - POST /users       -> weight 2 (20% of total)
    """

    wait_time = between(1, 3)

    def on_start(self):
        """Seed a user so GET by ID has a valid target."""
        response = self.client.post("/users", json={
            "name": "seed_user",
            "email": "seed@example.com"
        })
        if response.status_code == 201:
            self.user_id = response.json().get("id", 1)
        else:
            self.user_id = 1

    @task(4)
    def list_users(self):
        """GET /users — list all users (40% of traffic)."""
        self.client.get("/users")

    @task(4)
    def get_user(self):
        """GET /users/{id} — fetch a single user (40% of traffic)."""
        self.client.get(f"/users/{self.user_id}")

    @task(2)
    def create_user(self):
        """POST /users — create a new user (20% of traffic)."""
        import random
        suffix = random.randint(1, 100_000)
        with self.client.post(
            "/users",
            json={
                "name": f"user_{suffix}",
                "email": f"user_{suffix}@example.com"
            },
            catch_response=True
        ) as response:
            if response.status_code == 201:
                data = response.json()
                if "id" not in data:
                    response.failure("Response missing 'id' field")
            elif response.status_code == 409:
                response.success()  # Duplicate is acceptable under load
            else:
                response.failure(f"Unexpected status: {response.status_code}")


# Run headless for CI:
# locust -f locustfile.py --host=http://localhost:8000 \
#     --headless --users 50 --spawn-rate 5 --run-time 2m --csv=results
SOLUTION
}

# === Exercise 2: Custom Load Shape ===
# Problem: Create a LoadTestShape that simulates a workday pattern —
# low traffic at night, ramp up in the morning, peak at noon,
# gradual decline in the evening.
exercise_2() {
    echo "=== Exercise 2: Custom Load Shape ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from locust import HttpUser, LoadTestShape, task, between


class WorkdayShape(LoadTestShape):
    """Simulates a workday traffic pattern over a compressed timeline.

    Real pattern (24h) compressed into 12 minutes:
    - Night   (0-2 min):   10 users, slow ramp
    - Morning (2-4 min):   ramp to 80 users
    - Peak    (4-6 min):   100 users (noon rush)
    - Afternoon (6-8 min): 70 users (gradual decline)
    - Evening (8-10 min):  40 users
    - Night   (10-12 min): back to 10 users
    """

    stages = [
        {"duration": 120,  "users": 10,  "spawn_rate": 2},   # Night
        {"duration": 240,  "users": 80,  "spawn_rate": 10},  # Morning ramp
        {"duration": 360,  "users": 100, "spawn_rate": 5},   # Noon peak
        {"duration": 480,  "users": 70,  "spawn_rate": 5},   # Afternoon
        {"duration": 600,  "users": 40,  "spawn_rate": 5},   # Evening
        {"duration": 720,  "users": 10,  "spawn_rate": 5},   # Night again
    ]

    def tick(self):
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["duration"]:
                return (stage["users"], stage["spawn_rate"])

        return None  # Stop the test


class WorkdayUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def browse(self):
        self.client.get("/")

    @task(2)
    def view_item(self):
        self.client.get("/items/1")

    @task(1)
    def search(self):
        self.client.get("/search?q=test")


# Run:
# locust -f locustfile.py --host=http://localhost:8000 --headless
SOLUTION
}

# === Exercise 3: Benchmark Comparison ===
# Problem: Use pytest-benchmark to compare three different approaches
# to finding duplicates in a list (set-based, sort-based, brute-force).
exercise_3() {
    echo "=== Exercise 3: Benchmark Comparison ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import random

import pytest


def find_duplicates_set(items):
    """Set-based approach: O(n) time, O(n) space."""
    seen = set()
    duplicates = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return list(duplicates)


def find_duplicates_sort(items):
    """Sort-based approach: O(n log n) time, O(1) extra space."""
    sorted_items = sorted(items)
    duplicates = set()
    for i in range(1, len(sorted_items)):
        if sorted_items[i] == sorted_items[i - 1]:
            duplicates.add(sorted_items[i])
    return list(duplicates)


def find_duplicates_brute(items):
    """Brute-force approach: O(n^2) time, O(1) extra space."""
    duplicates = set()
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j]:
                duplicates.add(items[i])
    return list(duplicates)


@pytest.fixture
def sample_data():
    """Generate a list with known duplicates."""
    random.seed(42)
    data = list(range(500)) + random.choices(range(500), k=100)
    random.shuffle(data)
    return data


def test_find_duplicates_set(benchmark, sample_data):
    benchmark.group = "find-duplicates"
    result = benchmark(find_duplicates_set, sample_data)
    assert len(result) > 0  # Verify correctness


def test_find_duplicates_sort(benchmark, sample_data):
    benchmark.group = "find-duplicates"
    result = benchmark(find_duplicates_sort, sample_data)
    assert len(result) > 0


def test_find_duplicates_brute(benchmark, sample_data):
    benchmark.group = "find-duplicates"
    result = benchmark(find_duplicates_brute, sample_data)
    assert len(result) > 0


# All three should find the same duplicates
def test_all_approaches_agree(sample_data):
    set_result = set(find_duplicates_set(sample_data))
    sort_result = set(find_duplicates_sort(sample_data))
    brute_result = set(find_duplicates_brute(sample_data))
    assert set_result == sort_result == brute_result


# Run with comparison:
# pytest test_benchmark.py --benchmark-compare
# pytest test_benchmark.py --benchmark-group-by=group
SOLUTION
}

# === Exercise 4: Performance Regression Gate ===
# Problem: Set up a pytest-benchmark baseline and write a CI script
# that fails if any benchmark regresses by more than 15%.
exercise_4() {
    echo "=== Exercise 4: Performance Regression Gate ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# conftest.py — standardize machine info for consistent CI comparisons
def pytest_benchmark_generate_machine_info():
    """Return fixed machine info so CI comparisons are stable."""
    return {"cpu": "ci-runner", "machine": "github-actions"}


# tests/bench/test_critical_paths.py
import json
import hashlib


def compute_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def serialize_payload(records: list[dict]) -> str:
    return json.dumps(records, separators=(",", ":"))


def process_batch(items: list[int]) -> list[int]:
    return sorted(set(x * x for x in items if x % 2 == 0))


def test_hash_performance(benchmark):
    data = b"x" * 10_000
    result = benchmark(compute_hash, data)
    assert len(result) == 64


def test_serialize_performance(benchmark):
    records = [{"id": i, "name": f"item_{i}", "value": i * 1.5}
               for i in range(200)]
    result = benchmark(serialize_payload, records)
    assert isinstance(result, str)


def test_batch_processing(benchmark):
    items = list(range(5000))
    result = benchmark(process_batch, items)
    assert all(isinstance(x, int) for x in result)


# CI workflow steps:
#
# 1. Save baseline (run once on main branch):
#    pytest tests/bench/ --benchmark-save=baseline
#
# 2. Compare on every PR (fail if >15% regression):
#    pytest tests/bench/ \
#        --benchmark-compare=0001_baseline \
#        --benchmark-compare-fail=mean:15% \
#        --benchmark-compare-fail=stddev:25%
#
# GitHub Actions example:
#
# - name: Run benchmarks against baseline
#   run: |
#     pytest tests/bench/ \
#       --benchmark-compare=0001_baseline \
#       --benchmark-compare-fail=mean:15%
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 11: Performance Testing"
echo "======================================================"
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4

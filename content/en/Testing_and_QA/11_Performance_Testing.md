# Lesson 11: Performance Testing

**Previous**: [Property-Based Testing](./10_Property_Based_Testing.md) | **Next**: [Security Testing](./12_Security_Testing.md)

---

Performance bugs are invisible until they are catastrophic. A system can pass every functional test and still collapse under real-world load. Performance testing is the discipline of measuring how a system behaves under stress — verifying not just that it produces correct results, but that it produces them fast enough, often enough, and for enough users simultaneously. This lesson covers both macro-level load testing (can your service handle 10,000 concurrent users?) and micro-level benchmarking (is this function fast enough?).

**Difficulty**: ⭐⭐⭐

**Prerequisites**:
- Comfortable with Python and pytest (Lessons 02–03)
- Basic understanding of HTTP and web applications
- Familiarity with statistical concepts (mean, median, percentiles)

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between load testing, stress testing, soak testing, and spike testing
2. Write and run load tests using Locust with custom user scenarios
3. Use pytest-benchmark for precise micro-benchmarks of Python functions
4. Interpret performance metrics including throughput, latency percentiles, and error rates
5. Design load test shapes that simulate realistic traffic patterns

---

## 1. Types of Performance Testing

Performance testing is an umbrella term covering several distinct practices:

| Type | Question It Answers | How It Works |
|---|---|---|
| **Load testing** | Can we handle expected traffic? | Simulate normal to peak user load |
| **Stress testing** | Where is the breaking point? | Increase load beyond capacity |
| **Soak testing** | Are there memory leaks or degradation? | Sustained load over hours/days |
| **Spike testing** | Can we handle sudden bursts? | Abrupt load increase and decrease |
| **Scalability testing** | Does adding resources help? | Measure throughput vs. resources |

Each type answers a different question about your system. Most teams start with load testing and add others as maturity grows.

---

## 2. Load Testing with Locust

[Locust](https://locust.io/) is a Python-based load testing framework where you define user behavior as Python code. It is intuitive for Python developers because test scenarios are just classes and methods.

### 2.1 Installation

```bash
pip install locust
```

### 2.2 Your First Locustfile

Create `locustfile.py`:

```python
from locust import HttpUser, task, between


class WebsiteUser(HttpUser):
    """Simulates a user browsing a website."""

    # Wait 1-3 seconds between tasks (simulates think time)
    wait_time = between(1, 3)

    @task(3)  # Weight: 3x more likely than other tasks
    def view_homepage(self):
        self.client.get("/")

    @task(2)
    def view_article(self):
        self.client.get("/articles/1")

    @task(1)
    def search(self):
        self.client.get("/search?q=python")

    def on_start(self):
        """Called when a simulated user starts. Good for login."""
        self.client.post("/login", json={
            "username": "testuser",
            "password": "testpass"
        })
```

### 2.3 Running Locust

```bash
# Web UI mode (recommended for exploration)
locust -f locustfile.py --host=http://localhost:8000

# Headless mode (for CI/CD)
locust -f locustfile.py \
    --host=http://localhost:8000 \
    --headless \
    --users 100 \
    --spawn-rate 10 \
    --run-time 5m \
    --csv=results
```

The web UI (default port 8089) provides real-time charts of requests per second, response times, and failure rates.

### 2.4 Task Sequences

For multi-step workflows, use `SequentialTaskSet`:

```python
from locust import HttpUser, SequentialTaskSet, task, between


class CheckoutFlow(SequentialTaskSet):
    """Simulates a complete purchase flow — steps execute in order."""

    @task
    def browse_catalog(self):
        self.client.get("/catalog")

    @task
    def add_to_cart(self):
        self.client.post("/cart", json={"product_id": 42, "quantity": 1})

    @task
    def view_cart(self):
        self.client.get("/cart")

    @task
    def checkout(self):
        with self.client.post(
            "/checkout",
            json={"payment_method": "credit_card"},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "order_id" not in data:
                    response.failure("No order_id in response")
            else:
                response.failure(f"Checkout failed: {response.status_code}")


class EcommerceUser(HttpUser):
    wait_time = between(2, 5)
    tasks = [CheckoutFlow]
```

### 2.5 Custom Load Shapes

Default Locust ramps users linearly. For realistic scenarios, define custom shapes:

```python
from locust import LoadTestShape


class StepLoadShape(LoadTestShape):
    """
    Step load pattern:
    - 5 min at 50 users
    - 5 min at 100 users
    - 5 min at 200 users
    - 5 min cooldown at 50 users
    """

    stages = [
        {"duration": 300, "users": 50, "spawn_rate": 10},
        {"duration": 600, "users": 100, "spawn_rate": 10},
        {"duration": 900, "users": 200, "spawn_rate": 20},
        {"duration": 1200, "users": 50, "spawn_rate": 10},
    ]

    def tick(self):
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["duration"]:
                return (stage["users"], stage["spawn_rate"])

        return None  # Stop the test


class SpikeShape(LoadTestShape):
    """Simulate a traffic spike — e.g., flash sale or news mention."""

    def tick(self):
        run_time = self.get_run_time()

        if run_time < 60:
            return (10, 2)       # Normal: 10 users
        elif run_time < 120:
            return (500, 100)    # Spike: 500 users, fast ramp
        elif run_time < 300:
            return (10, 50)      # Recovery: back to 10
        return None
```

### 2.6 Handling Authentication and State

```python
class AuthenticatedUser(HttpUser):
    wait_time = between(1, 3)
    token = None

    def on_start(self):
        """Login and store JWT token."""
        response = self.client.post("/auth/login", json={
            "email": "loadtest@example.com",
            "password": "testpassword"
        })
        self.token = response.json()["access_token"]
        self.client.headers.update({
            "Authorization": f"Bearer {self.token}"
        })

    @task
    def get_profile(self):
        self.client.get("/api/profile")

    @task
    def update_settings(self):
        self.client.put("/api/settings", json={
            "notifications": True
        })
```

---

## 3. Micro-Benchmarking with pytest-benchmark

While Locust tests entire systems, `pytest-benchmark` measures individual function performance with statistical rigor.

### 3.1 Installation

```bash
pip install pytest-benchmark
```

### 3.2 Basic Benchmarks

```python
import json


def serialize_data(data):
    return json.dumps(data)


def test_serialize_benchmark(benchmark):
    """The benchmark fixture handles timing, iterations, and statistics."""
    data = {"users": [{"name": f"user_{i}", "age": i} for i in range(100)]}
    result = benchmark(serialize_data, data)
    assert isinstance(result, str)  # You can still assert correctness
```

### 3.3 Comparing Implementations

```python
import json
import pickle


def test_json_serialize(benchmark):
    data = list(range(1000))
    benchmark(json.dumps, data)


def test_pickle_serialize(benchmark):
    data = list(range(1000))
    benchmark(pickle.dumps, data)
```

Run with comparison:

```bash
pytest tests/bench/ --benchmark-compare
```

### 3.4 Benchmark Groups and Calibration

```python
def test_list_comprehension(benchmark):
    benchmark.group = "list-creation"
    benchmark(lambda: [x**2 for x in range(1000)])


def test_map_function(benchmark):
    benchmark.group = "list-creation"
    benchmark(lambda: list(map(lambda x: x**2, range(1000))))


def test_generator_to_list(benchmark):
    benchmark.group = "list-creation"

    def gen():
        return list(x**2 for x in range(1000))

    benchmark(gen)
```

### 3.5 Pedantic Mode for Precise Measurements

```python
def test_critical_path(benchmark):
    """Pedantic mode gives more control over iterations."""
    benchmark.pedantic(
        my_function,
        args=(large_dataset,),
        iterations=100,     # Exact number of iterations per round
        rounds=10,          # Number of rounds
        warmup_rounds=3     # Warmup rounds (not measured)
    )
```

### 3.6 Preventing Performance Regressions

Save a baseline and compare against it:

```bash
# Save baseline
pytest tests/bench/ --benchmark-save=baseline

# Compare against baseline (fails if >10% slower)
pytest tests/bench/ --benchmark-compare=baseline --benchmark-compare-fail=mean:10%
```

---

## 4. Interpreting Performance Metrics

### 4.1 Key Metrics

| Metric | What It Measures | Why It Matters |
|---|---|---|
| **Throughput** (RPS) | Requests per second | System capacity |
| **p50 (median)** | 50th percentile latency | Typical user experience |
| **p95** | 95th percentile latency | Experience for 1 in 20 users |
| **p99** | 99th percentile latency | Tail latency; worst 1% of users |
| **Error rate** | Percentage of failed requests | Reliability under load |
| **Concurrency** | Simultaneous active requests | Resource utilization |

### 4.2 Why Percentiles Matter More Than Averages

```
Average response time: 200ms   ← Looks fine!
p50: 150ms                     ← Half of users see this
p95: 800ms                     ← 1 in 20 see this
p99: 5,000ms                   ← 1 in 100 wait 5 seconds

The average hides the long tail.
```

A user who makes 10 requests will likely hit the p99 at least once. If your p99 is unacceptable, your average user will experience unacceptable performance eventually.

### 4.3 Reading Locust Results

Locust produces CSV output in headless mode:

```python
import pandas as pd

# Load Locust results
stats = pd.read_csv("results_stats.csv")

# Key columns: Name, Request Count, Failure Count,
#               Median Response Time, 95%ile, 99%ile,
#               Average Response Time, Requests/s

# Check SLO compliance
slo_p99_ms = 500
violations = stats[stats["99%"] > slo_p99_ms]
if not violations.empty:
    print(f"SLO violations:\n{violations[['Name', '99%']]}")
```

---

## 5. A Note on k6

While this lesson focuses on Python-native tools, [k6](https://k6.io/) is worth mentioning as a popular alternative for load testing. k6 uses JavaScript for test scripts and is written in Go for high performance. It excels at:

- Very high concurrency (100,000+ virtual users)
- Built-in protocol support (HTTP, WebSocket, gRPC)
- Cloud execution for distributed testing
- Integration with Grafana for visualization

Choose Locust when your team is Python-centric and values code reuse with your application. Choose k6 when you need extreme concurrency or are already in the Grafana ecosystem.

---

## 6. Performance Testing in CI/CD

### 6.1 Automated Performance Gates

```python
# conftest.py — fail CI if benchmarks regress
import pytest


def pytest_benchmark_compare_machine_info(config, benchmarksession):
    """Custom machine comparison for CI consistency."""
    pass


def pytest_benchmark_generate_machine_info():
    """Standardize machine info for CI."""
    return {"cpu": "ci-runner", "machine": "github-actions"}
```

### 6.2 Lightweight Performance Smoke Tests

Full load tests are expensive. Run lightweight smoke tests in CI and full suites on a schedule:

```python
# tests/perf/test_smoke.py
import time

import pytest


@pytest.mark.performance
def test_homepage_responds_quickly(client):
    """Smoke test: homepage should respond within 200ms."""
    start = time.perf_counter()
    response = client.get("/")
    elapsed = time.perf_counter() - start
    assert response.status_code == 200
    assert elapsed < 0.200, f"Homepage too slow: {elapsed:.3f}s"


@pytest.mark.performance
def test_api_throughput(client):
    """Verify minimum throughput of 50 requests in 1 second."""
    start = time.perf_counter()
    count = 0
    while time.perf_counter() - start < 1.0:
        response = client.get("/api/health")
        assert response.status_code == 200
        count += 1
    assert count >= 50, f"Throughput too low: {count} req/s"
```

---

## 7. Common Pitfalls

1. **Testing in production-unlike environments**: A laptop benchmark tells you nothing about production. Match hardware, data volume, and network topology as closely as possible.

2. **Ignoring warmup**: JIT compilers, connection pools, and caches need warmup. Always include a warmup phase before measuring.

3. **Measuring the wrong thing**: If your load test client is the bottleneck, you are measuring client performance, not server performance. Monitor client CPU and network.

4. **Using averages alone**: Always look at percentiles. An average of 100ms can hide a p99 of 10 seconds.

5. **Not testing with realistic data**: Testing with 10 database rows when production has 10 million will give meaningless results.

---

## Exercises

1. **Basic Load Test**: Write a Locust test for a REST API with three endpoints (`GET /users`, `GET /users/{id}`, `POST /users`). Use appropriate task weights to simulate realistic traffic (80% reads, 20% writes).

2. **Custom Shape**: Create a `LoadTestShape` that simulates a workday pattern — low traffic at night, ramp up in the morning, peak at noon, gradual decline in the evening.

3. **Benchmark Comparison**: Use pytest-benchmark to compare three different approaches to finding duplicates in a list (set-based, sort-based, and brute-force). Generate a comparison report.

4. **Performance Regression Gate**: Set up a pytest-benchmark baseline and write a CI script that fails if any benchmark regresses by more than 15%.

5. **Results Analysis**: Given a set of Locust CSV results, write a Python script that checks whether an SLO of "p99 < 500ms and error rate < 1%" is met for all endpoints.

---

**License**: CC BY-NC 4.0

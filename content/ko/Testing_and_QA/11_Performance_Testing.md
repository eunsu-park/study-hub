# 레슨 11: 성능 테스트 (Performance Testing)

**이전**: [속성 기반 테스트](./10_Property_Based_Testing.md) | **다음**: [보안 테스트](./12_Security_Testing.md)

---

성능 버그는 치명적인 수준에 도달할 때까지 보이지 않습니다. 시스템이 모든 기능 테스트를 통과하고도 실제 부하에서 무너질 수 있습니다. 성능 테스트는 시스템이 스트레스 하에서 어떻게 동작하는지 측정하는 분야입니다 — 올바른 결과를 생산하는지뿐만 아니라, 충분히 빠르게, 충분히 자주, 동시에 충분한 수의 사용자에게 결과를 제공하는지 검증합니다. 이 레슨에서는 매크로 수준의 부하 테스트(서비스가 10,000명의 동시 사용자를 처리할 수 있는가?)와 마이크로 수준의 벤치마킹(이 함수가 충분히 빠른가?) 모두를 다룹니다.

**난이도**: ⭐⭐⭐

**선수 조건**:
- Python과 pytest에 익숙할 것 (레슨 02-03)
- HTTP와 웹 애플리케이션에 대한 기본 이해
- 통계 개념(평균, 중앙값, 백분위수)에 대한 익숙함

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 부하 테스트, 스트레스 테스트, 내구성 테스트(soak testing), 스파이크 테스트를 구분할 수 있다
2. 커스텀 사용자 시나리오로 Locust를 사용하여 부하 테스트를 작성하고 실행할 수 있다
3. pytest-benchmark를 사용하여 Python 함수의 정밀한 마이크로 벤치마크를 수행할 수 있다
4. 처리량, 지연 시간 백분위수, 오류율 등 성능 지표를 해석할 수 있다
5. 현실적인 트래픽 패턴을 시뮬레이션하는 부하 테스트 형태를 설계할 수 있다

---

## 1. 성능 테스트의 유형

성능 테스트는 여러 구분된 실행 방법을 포괄하는 용어입니다:

| 유형 | 답하는 질문 | 수행 방법 |
|---|---|---|
| **부하 테스트** | 예상 트래픽을 처리할 수 있는가? | 정상~최대 사용자 부하를 시뮬레이션 |
| **스트레스 테스트** | 한계점은 어디인가? | 처리 능력을 초과하는 부하 증가 |
| **내구성 테스트** | 메모리 누수나 성능 저하가 있는가? | 수 시간/수 일에 걸친 지속적 부하 |
| **스파이크 테스트** | 갑작스러운 폭증을 처리할 수 있는가? | 급격한 부하 증가 및 감소 |
| **확장성 테스트** | 리소스 추가가 도움이 되는가? | 리소스 대비 처리량 측정 |

각 유형은 시스템에 대해 서로 다른 질문에 답합니다. 대부분의 팀은 부하 테스트부터 시작하고 성숙도가 높아지면 다른 유형을 추가합니다.

---

## 2. Locust를 사용한 부하 테스트

[Locust](https://locust.io/)는 Python 기반의 부하 테스트 프레임워크로, 사용자 행동을 Python 코드로 정의합니다. 테스트 시나리오가 클래스와 메서드로 이루어져 있어 Python 개발자에게 직관적입니다.

### 2.1 설치

```bash
pip install locust
```

### 2.2 첫 번째 Locustfile

`locustfile.py`를 생성합니다:

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

### 2.3 Locust 실행하기

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

웹 UI(기본 포트 8089)는 초당 요청 수, 응답 시간, 실패율에 대한 실시간 차트를 제공합니다.

### 2.4 작업 시퀀스

다단계 워크플로우의 경우 `SequentialTaskSet`을 사용합니다:

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

### 2.5 커스텀 부하 형태

기본 Locust는 사용자를 선형적으로 증가시킵니다. 현실적인 시나리오를 위해 커스텀 형태를 정의합니다:

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

### 2.6 인증과 상태 처리

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

## 3. pytest-benchmark를 사용한 마이크로 벤치마킹

Locust가 전체 시스템을 테스트하는 반면, `pytest-benchmark`는 통계적 엄밀성을 갖춘 개별 함수의 성능을 측정합니다.

### 3.1 설치

```bash
pip install pytest-benchmark
```

### 3.2 기본 벤치마크

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

### 3.3 구현 비교하기

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

비교 실행:

```bash
pytest tests/bench/ --benchmark-compare
```

### 3.4 벤치마크 그룹과 보정

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

### 3.5 정밀 측정을 위한 Pedantic 모드

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

### 3.6 성능 회귀 방지

베이스라인을 저장하고 이에 대해 비교합니다:

```bash
# Save baseline
pytest tests/bench/ --benchmark-save=baseline

# Compare against baseline (fails if >10% slower)
pytest tests/bench/ --benchmark-compare=baseline --benchmark-compare-fail=mean:10%
```

---

## 4. 성능 지표 해석하기

### 4.1 핵심 지표

| 지표 | 측정 대상 | 중요한 이유 |
|---|---|---|
| **처리량** (RPS) | 초당 요청 수 | 시스템 용량 |
| **p50 (중앙값)** | 50번째 백분위수 지연 시간 | 일반적인 사용자 경험 |
| **p95** | 95번째 백분위수 지연 시간 | 20명 중 1명의 경험 |
| **p99** | 99번째 백분위수 지연 시간 | 꼬리 지연 시간; 최악 1%의 사용자 |
| **오류율** | 실패한 요청의 비율 | 부하 하 신뢰성 |
| **동시성** | 동시 활성 요청 수 | 리소스 활용도 |

### 4.2 평균보다 백분위수가 중요한 이유

```
Average response time: 200ms   ← Looks fine!
p50: 150ms                     ← Half of users see this
p95: 800ms                     ← 1 in 20 see this
p99: 5,000ms                   ← 1 in 100 wait 5 seconds

The average hides the long tail.
```

10번의 요청을 보내는 사용자는 p99에 최소 한 번은 걸릴 가능성이 높습니다. p99이 수용 불가하다면, 평균 사용자도 결국 수용 불가한 성능을 경험하게 됩니다.

### 4.3 Locust 결과 읽기

Locust는 헤드리스 모드에서 CSV 출력을 생성합니다:

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

## 5. k6에 대한 참고 사항

이 레슨은 Python 네이티브 도구에 집중하지만, [k6](https://k6.io/)도 부하 테스트의 인기 있는 대안으로 언급할 가치가 있습니다. k6는 테스트 스크립트에 JavaScript를 사용하고 높은 성능을 위해 Go로 작성되었습니다. 다음과 같은 분야에서 뛰어납니다:

- 매우 높은 동시성 (100,000명 이상의 가상 사용자)
- 내장 프로토콜 지원 (HTTP, WebSocket, gRPC)
- 분산 테스트를 위한 클라우드 실행
- 시각화를 위한 Grafana 통합

팀이 Python 중심이고 애플리케이션과의 코드 재사용을 중시한다면 Locust를 선택합니다. 극도의 동시성이 필요하거나 이미 Grafana 생태계에 있다면 k6를 선택합니다.

---

## 6. CI/CD에서의 성능 테스트

### 6.1 자동화된 성능 게이트

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

### 6.2 경량 성능 스모크 테스트

전체 부하 테스트는 비용이 높습니다. CI에서는 경량 스모크 테스트를 실행하고 전체 스위트는 스케줄에 따라 실행합니다:

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

## 7. 일반적인 함정

1. **프로덕션과 다른 환경에서 테스트하기**: 노트북의 벤치마크는 프로덕션에 대해 아무것도 알려주지 않습니다. 하드웨어, 데이터 규모, 네트워크 토폴로지를 가능한 한 유사하게 맞춥니다.

2. **워밍업 무시하기**: JIT 컴파일러, 커넥션 풀, 캐시는 워밍업이 필요합니다. 측정 전에 항상 워밍업 단계를 포함합니다.

3. **잘못된 것 측정하기**: 부하 테스트 클라이언트가 병목이면 서버 성능이 아닌 클라이언트 성능을 측정하고 있는 것입니다. 클라이언트의 CPU와 네트워크를 모니터링합니다.

4. **평균만 사용하기**: 항상 백분위수를 확인합니다. 100ms의 평균이 10초의 p99를 숨길 수 있습니다.

5. **현실적인 데이터로 테스트하지 않기**: 프로덕션에 1,000만 행이 있는데 10행으로 테스트하면 의미 없는 결과를 얻게 됩니다.

---

## 연습 문제

1. **기본 부하 테스트**: 세 개의 엔드포인트(`GET /users`, `GET /users/{id}`, `POST /users`)가 있는 REST API에 대한 Locust 테스트를 작성합니다. 현실적인 트래픽을 시뮬레이션하기 위해 적절한 작업 가중치를 사용합니다 (80% 읽기, 20% 쓰기).

2. **커스텀 형태**: 근무일 패턴을 시뮬레이션하는 `LoadTestShape`를 생성합니다 — 밤에는 낮은 트래픽, 오전에 증가, 정오에 최대, 저녁에 점진적 감소.

3. **벤치마크 비교**: pytest-benchmark를 사용하여 리스트에서 중복을 찾는 세 가지 접근 방식(set 기반, 정렬 기반, 브루트포스)을 비교합니다. 비교 리포트를 생성합니다.

4. **성능 회귀 게이트**: pytest-benchmark 베이스라인을 설정하고 벤치마크가 15% 이상 회귀하면 실패하는 CI 스크립트를 작성합니다.

5. **결과 분석**: Locust CSV 결과가 주어졌을 때, 모든 엔드포인트에 대해 "p99 < 500ms 및 오류율 < 1%"의 SLO가 충족되는지 확인하는 Python 스크립트를 작성합니다.

---

**License**: CC BY-NC 4.0

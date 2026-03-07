# 16. API 생명주기 관리(API Lifecycle Management)

**이전**: [API 보안](./15_API_Security.md) | **다음**: 없음 (최종 레슨)

**난이도**: ⭐⭐⭐

---

## 학습 목표

- 설계부터 폐기(deprecation)와 일몰(sunset)까지 완전한 API 생명주기를 설명할 수 있다
- 소비자에게 변경 사항을 명확하게 전달하는 변경 로그(changelog) 관리 관행을 구현할 수 있다
- Sunset 및 Deprecation 헤더를 사용하여 클라이언트에게 API 폐기를 알릴 수 있다
- 소비자가 API 버전 간 전환할 수 있도록 돕는 마이그레이션 가이드를 작성할 수 있다
- 생명주기 결정에 정보를 제공하는 API 분석 및 모니터링 시스템을 설계할 수 있다
- API 소비자를 위한 중앙 허브 역할을 하는 개발자 포털을 구축할 수 있다

---

## 목차

1. [API 생명주기](#1-api-생명주기)
2. [설계 단계](#2-설계-단계)
3. [개발과 테스트](#3-개발과-테스트)
4. [배포와 버전 관리](#4-배포와-버전-관리)
5. [모니터링과 분석](#5-모니터링과-분석)
6. [변경 로그 관리](#6-변경-로그-관리)
7. [폐기와 일몰](#7-폐기와-일몰)
8. [마이그레이션 가이드](#8-마이그레이션-가이드)
9. [개발자 포털](#9-개발자-포털)
10. [연습 문제](#10-연습-문제)
11. [참고 자료](#11-참고-자료)

---

## 1. API 생명주기

모든 API는 예측 가능한 생명주기를 거칩니다. 각 단계를 이해하면 API를 언제 발전시키고, 폐기하고, 퇴역시킬지에 대해 더 나은 결정을 내릴 수 있습니다.

```
┌──────────────────────────────────────────────────────────┐
│                    API Lifecycle                          │
│                                                          │
│  ┌─────────┐   ┌──────────┐   ┌─────────┐              │
│  │ Design   │──→│ Develop  │──→│  Test    │              │
│  │          │   │          │   │          │              │
│  └─────────┘   └──────────┘   └────┬─────┘              │
│                                     │                    │
│                                     ▼                    │
│  ┌─────────┐   ┌──────────┐   ┌─────────┐              │
│  │ Sunset   │←──│ Deprecate│←──│ Deploy   │              │
│  │ & Retire │   │          │   │ & Run    │              │
│  └─────────┘   └──────────┘   └────┬─────┘              │
│                                     │                    │
│                                     ▼                    │
│                               ┌──────────┐              │
│                               │ Monitor  │              │
│                               │ & Evolve │──→ (next version)
│                               └──────────┘              │
└──────────────────────────────────────────────────────────┘
```

### 단계 요약

| 단계 | 주요 활동 | 산출물 |
|------|----------|--------|
| 설계 | 요구 사항, API 명세, 리뷰 | OpenAPI 명세, 설계 문서 |
| 개발 | 구현, 코드 리뷰 | 소스 코드, 유닛 테스트 |
| 테스트 | 계약 테스트, 통합 테스트, 보안 스캔 | 테스트 보고서 |
| 배포 | 스테이징, 카나리, 프로덕션 롤아웃 | 실행 중인 서비스 |
| 모니터링 | 메트릭, 알림, 사용 분석 | 대시보드, SLA 보고서 |
| 진화 | 버그 수정, 새 기능, 새 버전 | 변경 로그 항목 |
| 폐기 | 퇴역 공지, 마이그레이션 가이드 | Sunset 헤더, 이메일 |
| 일몰 | 엔드포인트 비활성화, 코드 제거 | 아카이브된 문서 |

---

## 2. 설계 단계

### API 설계 리뷰

코드를 작성하기 전에 API 설계 리뷰를 수행합니다. 이를 통해 출시 후 수정 비용이 많이 드는 문제를 사전에 발견할 수 있습니다.

```python
# Design review checklist (as a Python data structure for tooling)

API_DESIGN_CHECKLIST = {
    "naming": [
        "Resources use plural nouns (e.g., /users, /orders)",
        "URL segments use kebab-case or snake_case consistently",
        "No verbs in URLs (use HTTP methods instead)",
        "Nested resources are max 2 levels deep",
    ],
    "consistency": [
        "All endpoints follow the same naming convention",
        "Error response format is consistent (RFC 7807)",
        "Pagination format is consistent across all list endpoints",
        "Date/time format is ISO 8601 everywhere",
    ],
    "security": [
        "All endpoints specify authentication requirements",
        "Sensitive data is never in URL parameters",
        "Rate limits are defined per endpoint category",
        "Input validation rules are specified for all fields",
    ],
    "versioning": [
        "Version strategy is documented (URL path, header, etc.)",
        "Breaking vs. non-breaking change criteria are defined",
        "Backward compatibility policy is stated",
    ],
    "documentation": [
        "Every endpoint has a summary and description",
        "All parameters have types, descriptions, and examples",
        "Error responses are documented with examples",
        "Authentication flow is documented with code samples",
    ],
}
```

### 설계 우선 워크플로우(Design-First Workflow)

```python
# scripts/validate_api_design.py
"""Validate an OpenAPI spec against design standards before implementation."""

import yaml
import sys


def validate_design(spec_path: str) -> list[str]:
    """Check an OpenAPI spec for design standard compliance."""
    with open(spec_path) as f:
        spec = yaml.safe_load(f)

    issues = []

    for path, methods in spec.get("paths", {}).items():
        # Check: no verbs in paths
        verbs = {"get", "create", "update", "delete", "fetch", "list"}
        path_words = set(path.lower().strip("/").split("/"))
        if path_words & verbs:
            issues.append(
                f"[NAMING] Path '{path}' contains a verb. "
                f"Use HTTP methods instead."
            )

        # Check: consistent plural nouns
        segments = [s for s in path.split("/") if s and not s.startswith("{")]
        for segment in segments:
            if segment not in ("api", "v1", "v2", "admin") and not segment.endswith("s"):
                issues.append(
                    f"[NAMING] Path segment '{segment}' in '{path}' "
                    f"should be plural."
                )

        for method, details in methods.items():
            if method not in ("get", "post", "put", "patch", "delete"):
                continue

            # Check: summary exists
            if "summary" not in details:
                issues.append(
                    f"[DOCS] {method.upper()} {path} is missing a summary"
                )

            # Check: responses documented
            responses = details.get("responses", {})
            if not responses:
                issues.append(
                    f"[DOCS] {method.upper()} {path} has no responses defined"
                )

            # Check: error responses
            if method in ("post", "put", "patch"):
                if "422" not in responses and "400" not in responses:
                    issues.append(
                        f"[DOCS] {method.upper()} {path} should document "
                        f"validation error responses (400 or 422)"
                    )

    return issues


if __name__ == "__main__":
    issues = validate_design(sys.argv[1])
    if issues:
        print(f"Found {len(issues)} design issues:\n")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("API design validation passed.")
```

---

## 3. 개발과 테스트

### API를 위한 CI/CD 파이프라인

```yaml
# .github/workflows/api-ci.yml
name: API CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt

      - name: Lint
        run: |
          ruff check .
          mypy app/

      - name: Unit tests
        run: pytest tests/unit/ -v --cov=app --cov-report=xml

      - name: Integration tests
        run: pytest tests/integration/ -v

      - name: Contract tests (Schemathesis)
        run: |
          uvicorn app.main:app --port 8000 &
          sleep 3
          schemathesis run http://localhost:8000/openapi.json --checks all

      - name: OpenAPI spec validation
        run: python scripts/validate_api_design.py docs/openapi.yaml

      - name: Check for breaking changes
        run: |
          # Compare current spec with the published spec
          python scripts/check_breaking_changes.py \
            --old docs/openapi-published.yaml \
            --new docs/openapi.yaml

  deploy-staging:
    needs: lint-and-test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: echo "Deploy to staging environment"

      - name: Run smoke tests
        run: pytest tests/smoke/ --base-url https://staging-api.example.com
```

### 파괴적 변경 감지

```python
# scripts/check_breaking_changes.py
"""Detect breaking changes between two OpenAPI specs."""

import yaml
import sys


def detect_breaking_changes(old_spec: dict, new_spec: dict) -> list[str]:
    """Compare two OpenAPI specs and identify breaking changes.

    Breaking changes include:
    - Removed endpoints
    - Removed required request parameters
    - Changed parameter types
    - Removed response fields
    - Changed response status codes
    """
    breaking = []

    old_paths = old_spec.get("paths", {})
    new_paths = new_spec.get("paths", {})

    # Check for removed endpoints
    for path in old_paths:
        if path not in new_paths:
            breaking.append(f"REMOVED endpoint: {path}")
            continue

        for method in old_paths[path]:
            if method not in new_paths[path]:
                breaking.append(
                    f"REMOVED method: {method.upper()} {path}"
                )

    # Check for removed or changed parameters
    for path in old_paths:
        if path not in new_paths:
            continue

        for method in old_paths[path]:
            if method not in new_paths.get(path, {}):
                continue

            old_params = {
                p["name"]: p
                for p in old_paths[path][method].get("parameters", [])
            }
            new_params = {
                p["name"]: p
                for p in new_paths[path][method].get("parameters", [])
            }

            for name, old_param in old_params.items():
                if name not in new_params:
                    if old_param.get("required", False):
                        breaking.append(
                            f"REMOVED required parameter '{name}' from "
                            f"{method.upper()} {path}"
                        )

    # Check for newly required parameters (existing clients won't send them)
    for path in new_paths:
        if path not in old_paths:
            continue

        for method in new_paths[path]:
            if method not in old_paths.get(path, {}):
                continue

            old_params = {
                p["name"]
                for p in old_paths[path][method].get("parameters", [])
            }

            for param in new_paths[path][method].get("parameters", []):
                if param["name"] not in old_params and param.get("required"):
                    breaking.append(
                        f"ADDED required parameter '{param['name']}' to "
                        f"{method.upper()} {path}"
                    )

    return breaking


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        old = yaml.safe_load(f)
    with open(sys.argv[2]) as f:
        new = yaml.safe_load(f)

    changes = detect_breaking_changes(old, new)
    if changes:
        print(f"BREAKING CHANGES DETECTED ({len(changes)}):\n")
        for change in changes:
            print(f"  - {change}")
        sys.exit(1)
    else:
        print("No breaking changes detected.")
```

---

## 4. 배포와 버전 관리

### 카나리 배포(Canary Deployment)

전체 배포 전에 문제를 감지하기 위해 API 변경 사항을 점진적으로 롤아웃합니다:

```python
import random
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx

app = FastAPI()

# Canary configuration
CANARY_PERCENTAGE = 10  # Send 10% of traffic to the new version
CANARY_UPSTREAM = "http://api-v2:8001"
STABLE_UPSTREAM = "http://api-v1:8000"


@app.middleware("http")
async def canary_router(request: Request, call_next):
    """Route a percentage of traffic to the canary (new version).

    If the canary returns an error, fall back to the stable version
    to minimize user impact.
    """
    # Determine if this request goes to canary
    is_canary = random.randint(1, 100) <= CANARY_PERCENTAGE

    # Opt-in header for testing
    if request.headers.get("X-Use-Canary") == "true":
        is_canary = True

    upstream = CANARY_UPSTREAM if is_canary else STABLE_UPSTREAM

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.request(
                method=request.method,
                url=f"{upstream}{request.url.path}",
                headers=dict(request.headers),
                content=await request.body(),
            )

            # If canary fails with 5xx, fall back to stable
            if is_canary and response.status_code >= 500:
                response = await client.request(
                    method=request.method,
                    url=f"{STABLE_UPSTREAM}{request.url.path}",
                    headers=dict(request.headers),
                    content=await request.body(),
                )

        return JSONResponse(
            content=response.json(),
            status_code=response.status_code,
            headers={
                "X-Served-By": "canary" if is_canary else "stable",
            },
        )
    except Exception:
        return JSONResponse(
            status_code=502,
            content={"detail": "Upstream service unavailable"},
        )
```

### 버전 공존

여러 API 버전을 동시에 실행합니다:

```python
from fastapi import FastAPI, APIRouter

app = FastAPI()

# Version 1 router
v1 = APIRouter(prefix="/api/v1", tags=["v1"])


@v1.get("/users/{user_id}")
async def get_user_v1(user_id: int):
    """V1 response: flat structure."""
    user = await find_user(user_id)
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
    }


# Version 2 router (with enhanced response)
v2 = APIRouter(prefix="/api/v2", tags=["v2"])


@v2.get("/users/{user_id}")
async def get_user_v2(user_id: int):
    """V2 response: nested structure with metadata."""
    user = await find_user(user_id)
    return {
        "data": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "avatar_url": user.avatar_url,
            "created_at": user.created_at.isoformat(),
        },
        "meta": {
            "api_version": "v2",
            "deprecated_fields": [],
        },
    }


app.include_router(v1)
app.include_router(v2)
```

---

## 5. 모니터링과 분석

### API 메트릭

모든 API 엔드포인트에 대해 네 가지 범주의 메트릭을 추적합니다:

```python
import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("api.metrics")


@dataclass
class EndpointMetrics:
    """Metrics for a single API endpoint."""
    request_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    status_codes: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    latencies: list[float] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count

    @property
    def avg_latency_ms(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count


# In-memory metrics store (use Prometheus/Datadog in production)
metrics_store: dict[str, EndpointMetrics] = defaultdict(EndpointMetrics)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Collect request metrics for every endpoint."""
    start = time.perf_counter()

    response = await call_next(request)

    elapsed_ms = (time.perf_counter() - start) * 1000
    endpoint_key = f"{request.method} {request.url.path}"

    metrics = metrics_store[endpoint_key]
    metrics.request_count += 1
    metrics.total_latency_ms += elapsed_ms
    metrics.status_codes[response.status_code] += 1
    metrics.latencies.append(elapsed_ms)

    if response.status_code >= 400:
        metrics.error_count += 1

    # Log slow requests
    if elapsed_ms > 1000:
        logger.warning(
            f"Slow request: {endpoint_key} took {elapsed_ms:.0f}ms "
            f"(status={response.status_code})"
        )

    return response


@app.get("/admin/metrics")
async def get_metrics():
    """Return current API metrics for all endpoints."""
    result = {}
    for endpoint, metrics in metrics_store.items():
        sorted_latencies = sorted(metrics.latencies)
        n = len(sorted_latencies)

        result[endpoint] = {
            "request_count": metrics.request_count,
            "error_count": metrics.error_count,
            "error_rate": f"{metrics.error_rate:.2%}",
            "avg_latency_ms": round(metrics.avg_latency_ms, 2),
            "p50_latency_ms": round(sorted_latencies[n // 2], 2) if n else 0,
            "p95_latency_ms": round(sorted_latencies[int(n * 0.95)], 2) if n else 0,
            "p99_latency_ms": round(sorted_latencies[int(n * 0.99)], 2) if n else 0,
            "status_codes": dict(metrics.status_codes),
        }

    return result
```

### 사용 분석

어떤 소비자가 어떤 엔드포인트를 사용하는지 추적하여 폐기 결정에 정보를 제공합니다:

```python
from collections import defaultdict
from datetime import datetime, timezone


class UsageTracker:
    """Track API usage per consumer for lifecycle decisions.

    This data answers critical questions:
    - Which consumers still use deprecated endpoints?
    - Which endpoints have zero traffic and can be removed?
    - Which consumers will be affected by a breaking change?
    """

    def __init__(self, redis_client):
        self.redis = redis_client

    async def record(self, consumer_id: str, endpoint: str, method: str):
        """Record a single API usage event."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Per-consumer daily usage
        key = f"usage:{today}:{consumer_id}"
        await self.redis.hincrby(key, f"{method} {endpoint}", 1)
        await self.redis.expire(key, 90 * 86400)  # Keep 90 days

        # Per-endpoint daily unique consumers
        endpoint_key = f"consumers:{today}:{method}:{endpoint}"
        await self.redis.sadd(endpoint_key, consumer_id)
        await self.redis.expire(endpoint_key, 90 * 86400)

    async def get_endpoint_consumers(
        self, endpoint: str, method: str, days: int = 30
    ) -> set[str]:
        """Get all consumers who used an endpoint in the last N days."""
        consumers = set()
        for i in range(days):
            date = (
                datetime.now(timezone.utc) - timedelta(days=i)
            ).strftime("%Y-%m-%d")
            key = f"consumers:{date}:{method}:{endpoint}"
            members = await self.redis.smembers(key)
            consumers.update(m.decode() for m in members)
        return consumers

    async def get_dead_endpoints(self, days: int = 30) -> list[str]:
        """Find endpoints with zero traffic in the last N days."""
        all_endpoints = await self.get_all_registered_endpoints()
        dead = []
        for endpoint in all_endpoints:
            consumers = await self.get_endpoint_consumers(
                endpoint["path"], endpoint["method"], days
            )
            if not consumers:
                dead.append(f"{endpoint['method']} {endpoint['path']}")
        return dead
```

### 헬스 체크

```python
from datetime import datetime, timezone
import asyncio


@app.get("/health")
async def health_check():
    """Basic health check for load balancers and monitoring."""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check including dependency status.

    Used by operations teams to diagnose service issues.
    Protected by API key (not exposed to public traffic).
    """
    checks = {}

    # Database check
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = {"status": "healthy", "latency_ms": 1}
    except Exception as e:
        checks["database"] = {"status": "unhealthy", "error": str(e)}

    # Redis check
    try:
        await redis_client.ping()
        checks["redis"] = {"status": "healthy"}
    except Exception as e:
        checks["redis"] = {"status": "unhealthy", "error": str(e)}

    # External API check
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get("https://external-api.example.com/health")
            checks["external_api"] = {
                "status": "healthy" if resp.status_code == 200 else "degraded",
            }
    except Exception as e:
        checks["external_api"] = {"status": "unhealthy", "error": str(e)}

    overall = "healthy" if all(
        c["status"] == "healthy" for c in checks.values()
    ) else "degraded"

    return {
        "status": overall,
        "checks": checks,
        "version": "2.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
```

---

## 6. 변경 로그 관리

### 변경 로그 형식

소비자가 버전 간 변경 사항을 이해할 수 있도록 구조화된 변경 로그를 유지합니다:

```python
# changelog.py
"""API Changelog management."""

from dataclasses import dataclass
from datetime import date
from enum import Enum


class ChangeType(str, Enum):
    ADDED = "added"
    CHANGED = "changed"
    DEPRECATED = "deprecated"
    REMOVED = "removed"
    FIXED = "fixed"
    SECURITY = "security"


@dataclass
class ChangelogEntry:
    type: ChangeType
    description: str
    endpoint: str | None = None
    migration_guide: str | None = None


@dataclass
class ChangelogVersion:
    version: str
    date: date
    entries: list[ChangelogEntry]
    breaking: bool = False


# Example changelog
CHANGELOG = [
    ChangelogVersion(
        version="2.1.0",
        date=date(2025, 6, 1),
        entries=[
            ChangelogEntry(
                type=ChangeType.ADDED,
                description="Added `GET /books/{id}/reviews` endpoint for retrieving book reviews",
                endpoint="GET /books/{id}/reviews",
            ),
            ChangelogEntry(
                type=ChangeType.ADDED,
                description="Added `rating` field to Book response",
                endpoint="GET /books/{id}",
            ),
            ChangelogEntry(
                type=ChangeType.FIXED,
                description="Fixed pagination returning incorrect `total` count when filters are applied",
                endpoint="GET /books",
            ),
            ChangelogEntry(
                type=ChangeType.DEPRECATED,
                description="Deprecated `GET /books/{id}/comments` in favor of `/books/{id}/reviews`",
                endpoint="GET /books/{id}/comments",
                migration_guide="Replace `/comments` with `/reviews`. Response format is identical.",
            ),
        ],
        breaking=False,
    ),
    ChangelogVersion(
        version="2.0.0",
        date=date(2025, 3, 1),
        entries=[
            ChangelogEntry(
                type=ChangeType.CHANGED,
                description="Price field changed from flat `price: 49.99` to nested `price: {amount: 49.99, currency: 'USD'}`",
                endpoint="GET /books/{id}",
                migration_guide="Update client code to read `response.price.amount` instead of `response.price`.",
            ),
            ChangelogEntry(
                type=ChangeType.REMOVED,
                description="Removed `GET /api/v1/books` (sunset completed)",
                endpoint="GET /api/v1/books",
            ),
        ],
        breaking=True,
    ),
]


# Serve changelog via API
@app.get("/changelog")
async def get_changelog(since: str | None = None):
    """Return the API changelog, optionally filtered by version."""
    result = []
    for version in CHANGELOG:
        if since and version.version <= since:
            continue
        result.append({
            "version": version.version,
            "date": version.date.isoformat(),
            "breaking": version.breaking,
            "changes": [
                {
                    "type": entry.type.value,
                    "description": entry.description,
                    "endpoint": entry.endpoint,
                    "migration_guide": entry.migration_guide,
                }
                for entry in version.entries
            ],
        })
    return {"changelog": result}
```

---

## 7. 폐기와 일몰

### 폐기 헤더(Deprecation Headers)

`Deprecation` 헤더(RFC 초안)는 엔드포인트가 폐기되었음을 알립니다. `Sunset` 헤더(RFC 8594)는 제거 시기를 공지합니다.

```python
from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()


# Deprecated endpoint registry
DEPRECATED_ENDPOINTS = {
    ("GET", "/api/v1/books"): {
        "deprecated_at": "2025-01-01",
        "sunset_at": "2025-07-01",
        "replacement": "GET /api/v2/books",
        "migration_guide": "https://docs.example.com/migrate-v1-to-v2",
    },
    ("GET", "/api/v1/books/{book_id}/comments"): {
        "deprecated_at": "2025-03-01",
        "sunset_at": "2025-09-01",
        "replacement": "GET /api/v2/books/{book_id}/reviews",
        "migration_guide": "https://docs.example.com/comments-to-reviews",
    },
}


@app.middleware("http")
async def deprecation_headers_middleware(request: Request, call_next):
    """Add Deprecation and Sunset headers to deprecated endpoints."""
    response = await call_next(request)

    endpoint_key = (request.method, request.url.path)

    # Check if this endpoint is deprecated
    deprecation_info = None
    for (method, path_pattern), info in DEPRECATED_ENDPOINTS.items():
        if method == request.method and _path_matches(request.url.path, path_pattern):
            deprecation_info = info
            break

    if deprecation_info:
        # RFC draft: Deprecation header
        response.headers["Deprecation"] = deprecation_info["deprecated_at"]

        # RFC 8594: Sunset header
        response.headers["Sunset"] = deprecation_info["sunset_at"]

        # Link to the replacement and migration guide
        response.headers["Link"] = (
            f'<{deprecation_info["replacement"]}>; rel="successor-version", '
            f'<{deprecation_info["migration_guide"]}>; rel="deprecation"'
        )

        # Custom header for programmatic detection
        response.headers["X-API-Warn"] = (
            f"This endpoint is deprecated and will be removed on "
            f"{deprecation_info['sunset_at']}. "
            f"Use {deprecation_info['replacement']} instead."
        )

    return response


def _path_matches(actual: str, pattern: str) -> bool:
    """Simple path matching with {param} placeholders."""
    actual_parts = actual.strip("/").split("/")
    pattern_parts = pattern.strip("/").split("/")
    if len(actual_parts) != len(pattern_parts):
        return False
    return all(
        a == p or p.startswith("{")
        for a, p in zip(actual_parts, pattern_parts)
    )
```

### 일몰 시행(Sunset Enforcement)

일몰 날짜 이후에는 엔드포인트를 비활성화합니다:

```python
from datetime import datetime, date, timezone


@app.middleware("http")
async def sunset_enforcement_middleware(request: Request, call_next):
    """Return 410 Gone for endpoints past their sunset date."""
    for (method, path_pattern), info in DEPRECATED_ENDPOINTS.items():
        if method != request.method:
            continue
        if not _path_matches(request.url.path, path_pattern):
            continue

        sunset_date = date.fromisoformat(info["sunset_at"])
        if date.today() >= sunset_date:
            return JSONResponse(
                status_code=410,
                content={
                    "type": "https://api.example.com/errors/gone",
                    "title": "API Endpoint Removed",
                    "status": 410,
                    "detail": (
                        f"This endpoint was removed on {info['sunset_at']}. "
                        f"Use {info['replacement']} instead."
                    ),
                    "migration_guide": info["migration_guide"],
                },
                headers={
                    "Link": f'<{info["replacement"]}>; rel="successor-version"',
                },
            )

    return await call_next(request)
```

### 커뮤니케이션 계획

```python
# scripts/notify_deprecation.py
"""Notify affected API consumers about upcoming deprecation."""

import asyncio
from datetime import date, timedelta


async def notify_affected_consumers(
    endpoint: str,
    method: str,
    sunset_date: date,
    usage_tracker: UsageTracker,
    email_service,
):
    """Send deprecation notices to all consumers still using the endpoint.

    Timeline:
    - 6 months before sunset: Initial announcement
    - 3 months before sunset: Reminder with migration guide
    - 1 month before sunset: Final warning
    - 1 week before sunset: Urgent notice
    """
    consumers = await usage_tracker.get_endpoint_consumers(
        endpoint, method, days=90
    )

    days_until_sunset = (sunset_date - date.today()).days

    if days_until_sunset <= 0:
        subject = f"URGENT: {method} {endpoint} has been removed"
        urgency = "critical"
    elif days_until_sunset <= 7:
        subject = f"FINAL WARNING: {method} {endpoint} removal in {days_until_sunset} days"
        urgency = "high"
    elif days_until_sunset <= 30:
        subject = f"Reminder: {method} {endpoint} removal in {days_until_sunset} days"
        urgency = "medium"
    else:
        subject = f"Notice: {method} {endpoint} deprecated, sunset on {sunset_date}"
        urgency = "low"

    for consumer_id in consumers:
        consumer = await get_consumer_contact(consumer_id)
        await email_service.send(
            to=consumer.email,
            subject=subject,
            body=f"""
Your application ({consumer.app_name}) is still using
{method} {endpoint}, which will be removed on {sunset_date}.

Please migrate to the replacement endpoint.
Migration guide: https://docs.example.com/migration

Your recent usage: {await usage_tracker.get_consumer_usage(consumer_id, endpoint, method)}
            """,
        )

    print(f"Notified {len(consumers)} consumers about {method} {endpoint}")
```

---

## 8. 마이그레이션 가이드

### 효과적인 마이그레이션 가이드 작성

```python
# Example migration guide structure served via API

MIGRATION_GUIDES = {
    "v1-to-v2": {
        "title": "Migrating from API v1 to v2",
        "summary": "Guide for updating your integration from v1 to v2",
        "breaking_changes": [
            {
                "endpoint": "GET /api/v1/books/{id}",
                "change": "Price field restructured from float to object",
                "before": {"price": 49.99},
                "after": {"price": {"amount": 49.99, "currency": "USD"}},
                "migration": "Update price references: `book.price` → `book.price.amount`",
                "code_example": {
                    "before": """
# v1 client code
book = api.get_book(42)
total = book["price"] * quantity
""",
                    "after": """
# v2 client code
book = api.get_book(42)
total = book["price"]["amount"] * quantity
currency = book["price"]["currency"]
""",
                },
            },
            {
                "endpoint": "POST /api/v1/books",
                "change": "ISBN validation now requires ISBN-13 format",
                "before": {"isbn": "0132350884"},
                "after": {"isbn": "978-0132350884"},
                "migration": "Prefix ISBNs with '978-' if not already in ISBN-13 format",
            },
        ],
        "new_features": [
            "Book reviews endpoint: GET /api/v2/books/{id}/reviews",
            "Cursor-based pagination on all list endpoints",
            "Field selection with ?fields=id,title,price",
        ],
        "timeline": {
            "v2_available": "2025-01-01",
            "v1_deprecated": "2025-03-01",
            "v1_sunset": "2025-09-01",
        },
    },
}


@app.get("/migration-guides/{guide_id}")
async def get_migration_guide(guide_id: str):
    """Return a migration guide for API version transitions."""
    guide = MIGRATION_GUIDES.get(guide_id)
    if not guide:
        raise HTTPException(status_code=404, detail="Migration guide not found")
    return guide
```

---

## 9. 개발자 포털

개발자 포털은 API 소비자를 위한 중앙 허브입니다. 문서, 시작 가이드, API 키, 사용 분석, 지원을 결합합니다.

### 포털 아키텍처

```python
from fastapi import FastAPI, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

portal_app = FastAPI(title="API Developer Portal")
templates = Jinja2Templates(directory="templates")


@portal_app.get("/", response_class=HTMLResponse)
async def portal_home(request: Request):
    """Developer portal landing page."""
    return templates.TemplateResponse("portal/home.html", {
        "request": request,
        "api_versions": ["v1 (deprecated)", "v2 (current)"],
        "quick_links": {
            "Getting Started": "/getting-started",
            "API Reference": "/docs",
            "Changelog": "/changelog",
            "SDKs": "/sdks",
            "Status": "/status",
        },
    })


@portal_app.get("/getting-started")
async def getting_started():
    """Step-by-step guide for new API consumers."""
    return {
        "steps": [
            {
                "step": 1,
                "title": "Create an account",
                "description": "Sign up at https://portal.example.com/register",
            },
            {
                "step": 2,
                "title": "Get your API key",
                "description": "Generate an API key from your dashboard",
                "code": 'curl -H "X-API-Key: your-key-here" https://api.example.com/v2/books',
            },
            {
                "step": 3,
                "title": "Make your first request",
                "code": """
import httpx

client = httpx.Client(
    base_url="https://api.example.com/v2",
    headers={"X-API-Key": "your-key-here"},
)

# List books
books = client.get("/books").json()
print(f"Found {books['pagination']['total']} books")
                """,
            },
            {
                "step": 4,
                "title": "Explore the API reference",
                "description": "Full interactive documentation at /docs",
            },
        ],
    }


@portal_app.get("/dashboard")
async def consumer_dashboard(current_user=Depends(get_current_user)):
    """Consumer dashboard showing API keys, usage, and alerts."""
    api_keys = await get_user_api_keys(current_user.id)
    usage = await get_usage_summary(current_user.id)
    alerts = await get_deprecation_alerts(current_user.id)

    return {
        "api_keys": [
            {
                "id": key.id,
                "prefix": key.key_prefix,
                "created_at": key.created_at.isoformat(),
                "last_used": key.last_used_at.isoformat() if key.last_used_at else None,
                "expires_at": key.expires_at.isoformat(),
            }
            for key in api_keys
        ],
        "usage_summary": {
            "total_requests_30d": usage.total,
            "top_endpoints": usage.top_endpoints,
            "error_rate": f"{usage.error_rate:.2%}",
            "avg_latency_ms": usage.avg_latency,
        },
        "alerts": [
            {
                "type": alert.type,
                "message": alert.message,
                "action_url": alert.action_url,
            }
            for alert in alerts
        ],
    }


@portal_app.get("/status")
async def api_status():
    """Public API status page."""
    return {
        "status": "operational",
        "services": {
            "api_v2": {"status": "operational", "uptime_30d": "99.98%"},
            "api_v1": {"status": "deprecated", "sunset": "2025-09-01"},
            "webhooks": {"status": "operational"},
            "documentation": {"status": "operational"},
        },
        "incidents": [],
        "scheduled_maintenance": [],
    }
```

---

## 10. 연습 문제

### 문제 1: 파괴적 변경 감지기

완전한 파괴적 변경 감지 도구를 구축하세요:

- 두 개의 OpenAPI 명세(이전과 현재)를 읽기
- 감지: 제거된 엔드포인트, 제거된 파라미터, 타입 변경, 추가된 필수 필드, 변경된 응답 스키마, 변경된 인증 요구 사항
- 각 변경을 "파괴적(breaking)" 또는 "비파괴적(non-breaking)"으로 분류
- Markdown 형식으로 보고서 출력
- CI 파이프라인에 통합 (파괴적 변경이 감지되면 종료 코드 1)

### 문제 2: 폐기 시스템

FastAPI 애플리케이션을 위한 완전한 폐기 관리 시스템을 구현하세요:

- 일몰 날짜와 함께 폐기된 엔드포인트를 나열하는 설정 파일
- `Deprecation`, `Sunset`, `Link` 헤더를 추가하는 미들웨어
- 일몰 날짜 이후 410 Gone을 반환하는 시행 로직
- 모든 폐기된 엔드포인트와 상태를 보여주는 관리자 엔드포인트
- 아직 폐기된 엔드포인트를 사용하는 소비자를 식별하는 사용 추적
- 정의된 간격으로 영향받는 소비자에게 이메일을 보내는 알림 스크립트

### 문제 3: API 분석 대시보드

다음을 추적하는 분석 시스템을 구축하세요:

- 시간당 엔드포인트별 요청 수 (Redis에 저장)
- 엔드포인트별 오류율
- 엔드포인트별 지연 시간 백분위수 (p50, p95, p99)
- 요청 볼륨 기준 상위 소비자
- 비활성 엔드포인트 (30일 동안 트래픽 없음)
- 버전 채택률 (v1 vs. v2 트래픽 비율)

날짜 범위 필터링과 그룹화 옵션이 포함된 `GET /admin/analytics`를 통해 분석을 노출하세요.

### 문제 4: 마이그레이션 가이드 생성기

두 개의 OpenAPI 명세를 비교하여 마이그레이션 가이드를 자동으로 생성하는 도구를 만드세요:

- 변경 전/후 예시를 포함한 모든 파괴적 변경 나열
- 모든 새 기능 나열 (추가된 엔드포인트, 새 파라미터)
- 모든 폐기 사항 나열
- 이전과 새 API 사용법을 보여주는 코드 예시 생성
- Markdown과 HTML로 출력
- 타임라인 섹션 포함 (폐기 날짜, 일몰 날짜)

### 문제 5: 개발자 포털

다음을 포함하는 최소한의 개발자 포털을 FastAPI로 구축하세요:

- `GET /portal` -- 시작 가이드가 포함된 랜딩 페이지
- `POST /portal/register` -- 개발자 등록
- `POST /portal/api-keys` -- API 키 생성 (속도 제한 등급 선택 포함)
- `GET /portal/dashboard` -- 사용 통계, 활성 API 키, 폐기 알림
- `GET /portal/changelog` -- 버전별 필터링이 가능한 형식화된 변경 로그
- `GET /portal/status` -- 서비스 상태 페이지

개발자 포털을 위한 간단한 HTML 프론트엔드(Jinja2 템플릿)를 포함하세요.

---

## 11. 참고 자료

- [RFC 8594: The Sunset HTTP Header Field](https://tools.ietf.org/html/rfc8594)
- [IETF Draft: Deprecation Header](https://datatracker.ietf.org/doc/draft-ietf-httpapi-deprecation-header/)
- [Keep a Changelog](https://keepachangelog.com/)
- [Semantic Versioning 2.0.0](https://semver.org/)
- [Stripe API Versioning](https://stripe.com/docs/api/versioning)
- [Stripe API Changelog](https://stripe.com/docs/changelog)
- [Google API Design Guide: Versioning](https://cloud.google.com/apis/design/versioning)
- [Microsoft REST API Guidelines: Deprecation](https://github.com/microsoft/api-guidelines/blob/vNext/Guidelines.md)
- [Backstage Developer Portal](https://backstage.io/)
- [API Evangelist: API Lifecycle](https://apievangelist.com/)

---

**이전**: [API 보안](./15_API_Security.md) | [개요](./00_Overview.md) | 다음: 없음 (최종 레슨)

**License**: CC BY-NC 4.0

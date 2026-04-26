# 17. 관찰 가능성(Observability)

**이전**: [프로덕션 배포](./16_Production_Deployment.md) | **다음**: [프로젝트: REST API](./18_Project_REST_API.md)

**난이도**: ⭐⭐⭐

## 학습 목표

- 관찰 가능성(Observability)의 세 기둥(로그, 메트릭, 트레이스)과 이들이 서로를 보완하는 방법을 설명한다
- structlog(Python) 또는 pino(Node.js)를 사용하여 상관관계 ID(correlation ID)를 포함한 구조화 로깅(structured logging)을 구현한다
- Prometheus 메트릭(metrics)으로 백엔드 애플리케이션을 계측하고 의미 있는 Grafana 대시보드를 생성한다
- OpenTelemetry를 활용하여 마이크로서비스 전반에 걸친 분산 추적(distributed tracing)을 추가한다
- 노이즈를 최소화하고 실행 가능한 신호를 극대화하는 알림(alerting) 전략을 설계한다

## 목차

도구 참조에 들어가기 전에, [**이론과 원리**](#이론과-원리) 섹션을 먼저 읽어보세요. 세 기둥(metrics/logs/traces)의 형식적 정의, W3C Trace Context와 span이 서비스 경계를 가로질러 어떻게 전파되는지, 그리고 OpenTelemetry의 semantic convention과 함께하는 Prometheus의 pull 기반 메트릭 모델을 다룹니다.

1. [관찰 가능성의 세 기둥](#1-관찰-가능성의-세-기둥)
2. [구조화 로깅](#2-구조화-로깅)
3. [Prometheus를 이용한 메트릭](#3-prometheus를-이용한-메트릭)
4. [OpenTelemetry를 이용한 분산 추적](#4-opentelemetry를-이용한-분산-추적)
5. [Grafana 대시보드](#5-grafana-대시보드)
6. [알림 전략](#6-알림-전략)
7. [Sentry를 이용한 오류 추적](#7-sentry를-이용한-오류-추적)
8. [연습 문제](#8-연습-문제)

---

## 이론과 원리

관찰 가능성은 "로깅 + 메트릭 + 트레이싱"을 쌓아 둔 것이 *아닙니다*. 세 기둥이 존재하는 이유는 각각이 다른 기둥이 답할 수 없는 다른 부류의 질문에 답하기 때문입니다. 형식적 분리를 이해하면 도구가 제 자리를 찾습니다.

- **(A) 세 기둥: metrics, logs, traces** — 각 기둥이 무엇을 잘하고, 어디서 실패하며, 왜 셋 다 필요한가.
- **(B) 분산 트레이싱: W3C Trace Context와 span 전파** — 요청이 서비스 경계를 가로질러 살아남는 고유 trace ID를 어떻게 얻는가.
- **(C) Prometheus: pull 기반 메트릭과 OpenTelemetry semantic convention** — 지배적 메트릭 모델과 표준 명명.

### A. 세 기둥

각 기둥은 시스템 동작의 다른 *인코딩*이며, 다른 질문에 최적화되어 있습니다.

#### A.1 Metrics: 시간에 걸친 집계 숫자

Metrics는 숫자 값의 시계열입니다: counter, gauge, histogram. 저장 단위는 (timestamp, label set) 튜플당 숫자 하나입니다. 카디널리티는 label set 크기에 의해 제한됩니다. 수백만 개의 고유 label 조합을 가진 메트릭은 위험합니다.

Metrics가 답하는 질문: "시스템이 지금 어떻게 하고 있고, 어떤 추세인가?"

```
http_requests_total{method="GET",route="/users",status="200"} = 12847
http_request_duration_seconds_bucket{le="0.05"} = 11823
http_request_duration_seconds_bucket{le="0.1"} = 12340
```

저장 비용 저렴, 쿼리 비용 저렴, 알림 빠름. 카디널리티에 의해 제한됨: 사용자 ID당 메트릭 label을 가질 수 없습니다. 저장 공간이 폭발합니다.

#### A.2 Logs: 컨텍스트가 있는 개별 이벤트

Logs는 타임스탬프가 있는 이벤트 기록입니다 — 보통 논리적 이벤트당 한 기록. 비용은 이벤트 양에 비례합니다.

Logs가 답하는 질문: "*이 특정 요청*에 무슨 일이 있었나?"

```json
{"ts":"2025-01-15T15:14:23Z","level":"info","event":"user.login","user_id":42,"ip":"192.168.1.5"}
```

높은 카디널리티 label은 logs에서 괜찮습니다(어떤 필드든 이벤트당 고유할 수 있습니다). 비용은 양입니다 — 높은 QPS에서 많은 바이트를 로깅합니다. 샘플링, 구조화 로깅(자유 텍스트가 아니라 JSON), 인덱싱된 검색(Elasticsearch, Loki)이 프로덕션 도구체인입니다.

#### A.3 Traces: 서비스를 가로지르는 인과 사슬

Trace는 분산 시스템을 통과하는 단일 요청의 경로를 기록합니다: 어떤 서비스를 거쳤는지, 각 단계가 얼마나 걸렸는지, 어디서 무엇이 실패했는지. 각 단계는 *span*이며, span은 트리로 중첩됩니다.

Traces가 답하는 질문: "왜 이 요청이 느리지 / 왜 실패했지?"

```
trace_id: abc123
├── span: API gateway       (12ms)
├── span: auth service       (3ms)
└── span: order service     (180ms)
    ├── span: db.query       (45ms)
    └── span: cache.get      (1ms)
```

Traces는 서비스를 가로지르는 *인과성*을 포착하는 유일한 기둥입니다. 가장 비싸기도 합니다(요청마다 많은 span을 만듭니다). 그래서 프로덕션 시스템은 샘플합니다 — 1%의 trace를 보관하고 나머지는 버립니다.

#### A.4 기둥들은 서로 보완적이지 중복되지 않는다

흔한 실수: "logs가 있는데 metrics가 필요한가?" 그렇습니다. metrics는 수백만 요청에 걸쳐 저렴하게 집계하기 때문입니다. 그 규모에서 logs는 쿼리 비용을 감당할 수 없습니다. 반대로, "metrics가 있는데 logs가 필요한가?" 그렇습니다. metrics는 무언가가 잘못되었다는 것을 알려주지만 어떤 특정 요청이나 사용자가 영향을 받았는지는 알려주지 않기 때문입니다.

올바른 멘탈 모델:

- **Metrics**는 모니터링과 알림에(연속, 낮은 카디널리티).
- **Traces**는 서비스를 통과하는 단일 요청의 경로를 이해하는 데.
- **Logs**는 이미 범위를 좁힌 후(종종 trace ID를 통해) 특정 이벤트의 전체 세부 사항에.

### B. 분산 트레이싱: W3C Trace Context

Trace가 작동하는 이유는 모든 서비스가 HTTP 경계를 가로질러 trace ID를 *전파*하는 공통 프로토콜을 말하기 때문입니다. W3C Trace Context 표준(W3C Recommendation, 2020)이 이전의 임시 Zipkin / Jaeger 헤더를 대체했습니다.

#### B.1 두 헤더

```
traceparent: 00-{trace_id}-{parent_span_id}-{flags}
tracestate: vendor1=val,vendor2=val
```

`traceparent`는 필수이며 다음을 운반합니다.

- `00` — 명세 버전.
- `trace_id` — trace를 식별하는 16바이트 16진 ID(요청의 모든 서비스에서 동일).
- `parent_span_id` — *부모* span의 8바이트 span ID(upstream 호출자).
- `flags` — sampled / not sampled 비트.

`tracestate`는 벤더 특화 확장 데이터입니다.

#### B.2 전파 작동 방식

서비스 A가 HTTP로 서비스 B를 호출할 때:

1. A의 tracer가 들어오는 요청 헤더를 읽습니다. `traceparent`가 있으면 그 trace에 합류하고, 없으면 새로 시작합니다.
2. A가 새 span을 만듭니다(받은 무엇이든의 자식).
3. B를 호출하기 전, A가 `traceparent: 00-{trace_id}-{A의_span_id}-01`을 나가는 요청 헤더에 주입합니다.
4. B가 그 헤더를 받고 trace를 계속하며, B의 span은 A의 span의 자식입니다.

Trace 트리는 모든 서비스가 협력해서 만듭니다. 각자가 자체 span을 기여하고, ID가 그것들을 연결합니다. 수집된 span은 (out-of-band로, OTLP/gRPC로) 트레이싱 백엔드(Jaeger, Tempo, Honeycomb)로 보내져 완전한 trace로 꿰매집니다.

#### B.3 샘플링

모든 요청을 trace하면 네트워크와 저장 비용이 곱해집니다. 프로덕션 시스템은 샘플합니다: 1%의 trace를 보관하거나, 100%의 오류 trace와 0.1%의 성공 trace를 보관합니다. 두 가지 주된 전략:

- **Head-based 샘플링.** 시작에서(첫 서비스에서) 샘플할지를 결정합니다. 결정은 `traceparent`의 `flags` 바이트에 있습니다. 저렴하지만 나중에 일어나는 일에 기반해 결정을 내릴 수 없습니다(예: "느린 trace는 항상 보관").
- **Tail-based 샘플링.** 모든 span을 버퍼링하고, trace가 완료된 후 결정합니다. 모든 오류 trace, 모든 느린 trace를 보관하고 성공 trace는 샘플할 수 있습니다. 더 비쌉니다(버퍼링하는 OpenTelemetry Collector 같은 사이드카가 필요).

### C. Prometheus와 OpenTelemetry

Prometheus는 지배적 오픈 소스 메트릭 시스템입니다. 그 모델은 의견이 강하며 업계가 메트릭에 대해 생각하는 방식을 형성했습니다.

#### C.1 Pull 기반 스크레이핑

대부분의 메트릭 시스템은 *push*입니다: 앱이 백엔드로 메트릭을 보냅니다. Prometheus는 *pull*입니다: 백엔드가 각 앱이 노출하는 `/metrics` 엔드포인트를 주기적으로 스크레이프합니다.

```
Prometheus 서버                          앱
     │                                   │
     │── GET /metrics ──────────────────►│
     │◄── http_requests_total{...} ─────│
     │   ... (텍스트 형식) ...            │
     │ (15초마다)                         │
```

Pull 기반의 성질:

- **서비스 발견이 중앙집중**됩니다 — 앱이 아니라 Prometheus에.
- **실패한 스크레이프가 보입니다** — `up{job="..."}` 메트릭으로. 앱이 도달 불가능한 것 자체가 신호입니다.
- **앱이 Prometheus를 모릅니다** — 그저 `/metrics`를 노출할 뿐입니다. 같은 엔드포인트가 호환되는 어떤 스크레이퍼와도 작동합니다.

단점: 단명 작업(cron, batch)이 스크레이프되기 전에 끝날 수 있습니다. 해결책은 *Pushgateway*입니다 — Prometheus가 수집할 때까지 메트릭을 보관하는 중간 서비스.

#### C.2 네 가지 메트릭 타입

| 타입 | 측정하는 것 | 예시 |
|------|------------------|---------|
| Counter | 단조 증가하는 총합 | requests_total, errors_total |
| Gauge | 현재 값(올라가거나 내려갈 수 있음) | memory_bytes, queue_depth |
| Histogram | 관측의 버킷 분포 | request_duration_seconds |
| Summary | 슬라이딩 윈도우의 분위수 추정 | response_size_bytes |

Histogram이 SLO 작업에 가장 중요합니다. 버킷 카운트를 읽어 "100ms 미만인 요청의 비율은?"을 물어볼 수 있게 해줍니다. Summary는 클라이언트 측에서 분위수를 계산하므로 인스턴스 간 집계가 더 어려워집니다. Histogram이 보통 선호됩니다.

#### C.3 OpenTelemetry semantic convention

OpenTelemetry(OTel)는 trace와 metric 모두를 위한 떠오르는 표준이며, 모든 주요 언어에 벤더 중립 SDK를 제공합니다. 가장 큰 기여는 **semantic conventions** — 흔한 속성에 대한 표준 명명 체계입니다.

```
http.request.method = GET
http.response.status_code = 200
http.route = /users/{id}
db.system = postgresql
db.statement = SELECT * FROM ...
```

모든 서비스가 같은 속성 이름을 쓰면 대시보드와 알림이 서비스 간에 작동합니다. Semantic convention 이전에는 모든 팀이 자체 명명을 발명했고 대시보드는 팀별이었습니다. 그것과 함께라면 "HTTP 요청의 95번째 백분위 지연시간을 route별로 보여주세요"가 OTel로 계측된 어떤 서비스에서도 작동하는 한 쿼리입니다.

추세는 OTel을 앱 측의 *벤더 중립 SDK*로 두고, 플랫폼이 사용하는 백엔드(Jaeger, Tempo, Datadog, Honeycomb)로 OTLP를 통해 export하는 것입니다. 백엔드는 바뀔 수 있고, 계측은 바뀌지 않습니다.

### 이론에서 아래 도구로

뒤에 나오는 각 절은 이 틀의 한 조각을 구체화합니다.

- §1 (세 기둥)은 §A를 구체적인 예제로.
- §2 (구조화 로깅)은 §A.2 logs 기둥입니다 — JSON 출력, correlation ID, OTel `trace_id` 주입.
- §3 (Prometheus를 이용한 메트릭)은 §C.1(pull 모델)과 §C.2(네 타입)을 구체적인 코드로.
- §4 (OpenTelemetry를 이용한 분산 추적)은 §B(W3C Trace Context)와 §C.3(semantic convention)을 코드로.
- §5 (Grafana 대시보드)는 §C.1의 메트릭을 읽는 시각화 계층입니다.
- §6 (알림 전략)은 §C 메트릭 위에 세워졌습니다 — 원시 원인이 아니라 SLO 위반에 대한 증상 기반 알림.
- §7 (Sentry를 이용한 오류 추적)은 예외에 특화된 §A.2 logs 기둥이며, 스택 트레이스와 그룹화가 있습니다.

---

## 1. 관찰 가능성의 세 기둥

관찰 가능성(Observability)은 시스템의 외부 출력을 검사하여 내부 상태를 이해하는 능력이다. 세 기둥은 시스템 동작에 대한 상호 보완적인 관점을 제공한다.

### 로그(Logs): 무슨 일이 일어났나

로그는 개별 이벤트를 기록한다. "오후 3시 14분에 무슨 일이 있었나?" 또는 "이 요청은 왜 실패했나?"와 같은 질문에 답한다.

```
2025-01-15T15:14:23Z INFO  user.login user_id=42 ip=192.168.1.5 method=password
2025-01-15T15:14:24Z ERROR payment.charge user_id=42 amount=99.99 error="card_declined"
```

### 메트릭(Metrics): 얼마나 / 얼마나 빠르게

메트릭은 시간에 걸쳐 집계된 수치 측정값이다. "초당 요청 수는 몇 개인가?" 또는 "99번째 백분위 지연 시간(latency)은 얼마인가?"와 같은 질문에 답한다.

```
http_requests_total{method="GET", endpoint="/users", status="200"} 15234
http_request_duration_seconds{quantile="0.99"} 0.250
```

### 트레이스(Traces): 요청의 여정

트레이스는 하나의 요청이 여러 서비스를 거치는 과정을 추적한다. "이 느린 요청에서 병목은 어디인가?" 또는 "어떤 다운스트림 서비스가 오류를 유발하는가?"와 같은 질문에 답한다.

```
[Trace: abc123]
  |-- API Gateway (5ms)
  |   |-- Auth Service (12ms)
  |   |-- User Service (45ms)
  |       |-- PostgreSQL query (30ms)
  |       |-- Redis cache set (2ms)
  |-- Response sent (total: 62ms)
```

### 세 기둥이 서로를 보완하는 방법

| 시나리오                  | 시작점        | 다음 단계     |
|---------------------------|---------------|---------------|
| "오류율이 급증했다"       | 메트릭 (감지) | 로그 (근본 원인 분석) |
| "이 요청이 느리다"        | 트레이스 (병목 찾기) | 메트릭 (시스템적 문제인가?) |
| "사용자가 실패를 보고했다" | 로그 (이벤트 찾기) | 트레이스 (전체 경로 확인) |

이 세 기둥을 연결하는 핵심은 로그 항목을 트레이스 스팬(span) 및 메트릭 레이블에 연결하는 **상관관계 ID(correlation ID)** (트레이스 ID 또는 요청 ID라고도 함)다.

---

## 2. 구조화 로깅

구조화 로깅(structured logging)은 자유 형식의 텍스트 대신 키-값 쌍(일반적으로 JSON)으로 로그 이벤트를 출력한다. 이를 통해 로그를 기계가 파싱할 수 있으면서도 사람이 읽을 수 있게 된다.

### Python: structlog

```python
import structlog
import uuid
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

# Configure structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Middleware to inject request context into every log entry
class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Bind context variables that will appear in all log entries
        # within this request's scope
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host,
        )

        logger.info("request.started")

        response = await call_next(request)

        logger.info(
            "request.completed",
            status_code=response.status_code,
        )

        response.headers["X-Request-ID"] = request_id
        return response

app = FastAPI()
app.add_middleware(RequestContextMiddleware)

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    logger.info("user.fetch", user_id=user_id)
    user = await fetch_user(user_id)
    if not user:
        logger.warning("user.not_found", user_id=user_id)
    return user
```

**출력 (JSON, 이벤트당 한 줄):**

```json
{"request_id": "a1b2c3", "method": "GET", "path": "/users/42", "client_ip": "10.0.0.1", "event": "request.started", "level": "info", "timestamp": "2025-01-15T15:14:23.000Z"}
{"request_id": "a1b2c3", "method": "GET", "path": "/users/42", "client_ip": "10.0.0.1", "user_id": 42, "event": "user.fetch", "level": "info", "timestamp": "2025-01-15T15:14:23.005Z"}
{"request_id": "a1b2c3", "method": "GET", "path": "/users/42", "client_ip": "10.0.0.1", "status_code": 200, "event": "request.completed", "level": "info", "timestamp": "2025-01-15T15:14:23.050Z"}
```

### Node.js: pino

```javascript
const pino = require("pino");
const express = require("express");
const { randomUUID } = require("crypto");

const logger = pino({
  level: process.env.LOG_LEVEL || "info",
  // Redact sensitive fields
  redact: ["req.headers.authorization", "req.headers.cookie"],
  serializers: {
    err: pino.stdSerializers.err,
  },
});

const app = express();

// Request logging middleware
app.use((req, res, next) => {
  const requestId = req.headers["x-request-id"] || randomUUID();
  // Create a child logger with request-scoped context
  req.log = logger.child({
    requestId,
    method: req.method,
    path: req.path,
    clientIp: req.ip,
  });

  req.log.info("request.started");

  const startTime = process.hrtime.bigint();
  res.on("finish", () => {
    const duration = Number(process.hrtime.bigint() - startTime) / 1e6;
    req.log.info({
      statusCode: res.statusCode,
      durationMs: duration.toFixed(2),
      event: "request.completed",
    });
  });

  res.setHeader("X-Request-ID", requestId);
  next();
});

app.get("/users/:id", (req, res) => {
  req.log.info({ userId: req.params.id }, "user.fetch");
  // ... handler logic
});
```

### 로그 레벨 가이드라인

| 레벨     | 사용 시점                                         | 예시                                    |
|---------|---------------------------------------------------|-----------------------------------------|
| DEBUG   | 상세한 진단 정보 (프로덕션에서 비활성화)           | SQL 쿼리, 캐시 히트/미스                |
| INFO    | 정상적인 운영 이벤트                               | 요청 시작/종료, 사용자 행동             |
| WARNING | 예상치 못했지만 복구 가능한 상황                   | 사용 중단된 API 사용, 느린 쿼리          |
| ERROR   | 주의가 필요한 실패                                | 처리되지 않은 예외, 서비스 타임아웃      |
| CRITICAL| 즉각적인 조치가 필요한 시스템 수준의 실패          | 데이터베이스 연결 끊김, OOM              |

---

## 3. Prometheus를 이용한 메트릭

Prometheus(프로메테우스)는 HTTP 엔드포인트를 스크래핑(scraping)하여 메트릭을 수집한다. 애플리케이션은 Prometheus 텍스트 형식으로 메트릭 값을 반환하는 `/metrics` 엔드포인트를 노출한다.

### 메트릭 유형

| 유형      | 설명                              | 예시                                     |
|-----------|-----------------------------------|------------------------------------------|
| Counter   | 단조 증가하는 값                   | 총 요청 수, 총 오류 수                   |
| Gauge     | 증가하거나 감소하는 값             | 활성 연결 수, 큐 깊이                    |
| Histogram | 버킷별 값의 분포                   | 요청 지연 시간, 응답 크기                |
| Summary   | Histogram과 유사하며, 클라이언트 측에서 분위수 계산 | 요청 지연 시간 백분위수 |

### FastAPI와 prometheus-fastapi-instrumentator

```python
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import time

app = FastAPI()

# Auto-instrument all endpoints with standard HTTP metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Custom business metrics
USERS_CREATED = Counter(
    "users_created_total",
    "Total number of users created",
    ["registration_method"],  # Labels for breaking down by method
)

PAYMENT_AMOUNT = Histogram(
    "payment_amount_dollars",
    "Payment amounts in dollars",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
)

ACTIVE_WEBSOCKET_CONNECTIONS = Gauge(
    "active_websocket_connections",
    "Number of active WebSocket connections",
)

DB_QUERY_DURATION = Histogram(
    "db_query_duration_seconds",
    "Database query execution time",
    ["query_type", "table"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)


@app.post("/users")
async def create_user(user: UserCreate):
    result = await save_user(user)
    USERS_CREATED.labels(registration_method="email").inc()
    return result


@app.post("/payments")
async def process_payment(payment: PaymentRequest):
    result = await charge(payment)
    PAYMENT_AMOUNT.observe(payment.amount)
    return result


# Database query timing helper
class QueryTimer:
    def __init__(self, query_type: str, table: str):
        self.query_type = query_type
        self.table = table
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        duration = time.perf_counter() - self.start_time
        DB_QUERY_DURATION.labels(
            query_type=self.query_type, table=self.table
        ).observe(duration)


# Usage
async def get_user_by_id(user_id: int):
    with QueryTimer("select", "users"):
        return await db.fetch_one("SELECT * FROM users WHERE id = $1", user_id)
```

### Prometheus 설정

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "fastapi-app"
    static_configs:
      - targets: ["app:8000"]
    metrics_path: /metrics

  - job_name: "node-exporter"
    static_configs:
      - targets: ["node-exporter:9100"]

  - job_name: "postgres-exporter"
    static_configs:
      - targets: ["postgres-exporter:9187"]
```

---

## 4. OpenTelemetry를 이용한 분산 추적

OpenTelemetry(OTel)는 분산 추적의 업계 표준이다. Jaeger, Zipkin, Grafana Tempo 또는 OTLP 호환 백엔드로 내보낼 수 있는 벤더 중립적인 계측 도구를 제공한다.

### 핵심 개념

- **트레이스(Trace)**: 모든 서비스를 거치는 요청의 전체 여정
- **스팬(Span)**: 트레이스 내의 단일 작업 (예: HTTP 요청, 데이터베이스 쿼리)
- **컨텍스트 전파(Context propagation)**: HTTP 헤더를 통해 서비스 간에 트레이스 ID를 전달하는 것

### FastAPI 계측

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource

# Initialize the tracer provider
resource = Resource.create({
    "service.name": "user-service",
    "service.version": "1.2.0",
    "deployment.environment": "production",
})

provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(
    OTLPSpanExporter(endpoint="http://otel-collector:4317")
)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)

# Auto-instrument frameworks
app = FastAPI()
FastAPIInstrumentor.instrument_app(app)
SQLAlchemyInstrumentor().instrument(engine=engine)
HTTPXClientInstrumentor().instrument()

# Custom spans for business logic
@app.post("/orders")
async def create_order(order: OrderCreate):
    with tracer.start_as_current_span("validate_order") as span:
        span.set_attribute("order.item_count", len(order.items))
        validated = validate_order(order)

    with tracer.start_as_current_span("check_inventory") as span:
        span.set_attribute("order.total", order.total)
        available = await check_inventory(order.items)
        if not available:
            span.set_status(trace.Status(trace.StatusCode.ERROR, "Out of stock"))
            raise HTTPException(status_code=409, detail="Item out of stock")

    with tracer.start_as_current_span("process_payment") as span:
        payment = await charge_payment(order)
        span.set_attribute("payment.id", payment.id)
        span.set_attribute("payment.method", payment.method)

    return {"order_id": order.id, "status": "confirmed"}
```

### Express.js 계측

```javascript
// tracing.js — must be imported BEFORE any other modules
const { NodeSDK } = require("@opentelemetry/sdk-node");
const { getNodeAutoInstrumentations } = require("@opentelemetry/auto-instrumentations-node");
const { OTLPTraceExporter } = require("@opentelemetry/exporter-trace-otlp-grpc");
const { Resource } = require("@opentelemetry/resources");

const sdk = new NodeSDK({
  resource: new Resource({
    "service.name": "order-service",
    "service.version": "2.1.0",
  }),
  traceExporter: new OTLPTraceExporter({
    url: "http://otel-collector:4317",
  }),
  instrumentations: [
    getNodeAutoInstrumentations({
      // Auto-instruments express, http, pg, redis, etc.
      "@opentelemetry/instrumentation-fs": { enabled: false },
    }),
  ],
});

sdk.start();
process.on("SIGTERM", () => sdk.shutdown());
```

```bash
# Start the app with tracing enabled
node --require ./tracing.js server.js
```

---

## 5. Grafana 대시보드

Grafana(그라파나)는 Prometheus(및 다른 데이터 소스)의 메트릭을 대시보드로 시각화한다. 잘 설계된 대시보드는 시스템 상태를 한눈에 파악할 수 있게 해준다.

### 서비스를 위한 RED 메서드

RED 메서드는 모든 서비스에 대한 세 가지 핵심 메트릭을 정의한다:

| 메트릭                    | 설명                      | PromQL 예시                                                         |
|--------------------------|---------------------------|---------------------------------------------------------------------|
| **R**ate (요청률)         | 초당 요청 수              | `rate(http_requests_total[5m])`                                    |
| **E**rrors (오류)         | 오류율                    | `rate(http_requests_total{status=~"5.."}[5m])`                     |
| **D**uration (지연 시간)  | 요청 지연 시간 분포        | `histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))` |

### 리소스를 위한 USE 메서드

USE 메서드는 모든 리소스(CPU, 메모리, 디스크, 네트워크)에 대한 세 가지 핵심 메트릭을 정의한다:

| 메트릭                       | 설명                     | PromQL 예시                                    |
|-----------------------------|--------------------------|------------------------------------------------|
| **U**tilization (사용률)     | 용량 사용 비율            | `process_cpu_seconds_total`                    |
| **S**aturation (포화도)      | 큐 깊이, 대기 중인 작업   | `node_load1` (1분 평균 부하)                  |
| **E**rrors (오류)            | 리소스의 오류 이벤트      | `node_disk_io_errors_total`                    |

### 필수 PromQL 쿼리

```promql
# Request rate (requests per second)
rate(http_requests_total[5m])

# Error rate as a percentage
100 * sum(rate(http_requests_total{status=~"5.."}[5m]))
    / sum(rate(http_requests_total[5m]))

# 99th percentile latency
histogram_quantile(0.99,
    sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
)

# Latency by endpoint (top 5 slowest)
topk(5,
    histogram_quantile(0.95,
        sum(rate(http_request_duration_seconds_bucket[5m])) by (le, endpoint)
    )
)

# Active database connections
pg_stat_activity_count{state="active"}

# Memory usage percentage
100 * process_resident_memory_bytes / machine_memory_bytes
```

### 대시보드 레이아웃 권장 사항

```
Row 1: Overview
  [Request Rate] [Error Rate %] [P50/P95/P99 Latency]

Row 2: Endpoints
  [Top endpoints by request volume]  [Slowest endpoints (P99)]

Row 3: Resources
  [CPU Usage] [Memory Usage] [Active DB Connections]

Row 4: Business Metrics
  [Users Created/min] [Orders Processed/min] [Payment Failures]
```

---

## 6. 알림 전략

좋은 알림(alerting)은 사람의 조치가 필요한 문제를 알리되, 노이즈로 뒤덮이지 않게 한다.

### 알림 설계 원칙

- **원인이 아닌 증상에 대해 알림**: "CPU > 90%"가 아닌 "오류율 > 5%"에 대해 알린다. 높은 CPU 사용률은 부하 하에서 정상일 수 있다.
- **실행 가능한 알림만**: 모든 알림에는 명확한 조치가 있어야 한다. 알림을 일상적으로 무시한다면 삭제한다.
- **컨텍스트 포함**: 알림 메시지에는 대시보드를 열지 않고도 디버깅을 시작할 수 있는 충분한 정보가 있어야 한다.
- **단계별 심각도**: 모든 것이 새벽 3시에 누군가를 깨울 필요는 없다.

### 심각도 수준

| 수준     | 응답 시간  | 채널         | 예시                             |
|---------|-----------|--------------|----------------------------------|
| Critical | 즉각       | PagerDuty    | 서비스 다운, 데이터 손실 위험     |
| Warning  | 업무 시간  | Slack        | 오류율 상승, 디스크 80%          |
| Info     | 다음 스프린트 | 이메일/티켓 | 인증서 30일 후 만료              |

### Prometheus 알림 규칙

```yaml
# alerts.yml
groups:
  - name: api-alerts
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          100 * sum(rate(http_requests_total{status=~"5.."}[5m]))
              / sum(rate(http_requests_total[5m]))
          > 5
        for: 5m    # Must persist for 5 minutes to fire
        labels:
          severity: critical
        annotations:
          summary: "High error rate: {{ $value | printf \"%.1f\" }}%"
          description: "Error rate has been above 5% for 5 minutes."
          runbook: "https://wiki.example.com/runbooks/high-error-rate"

      # Slow responses
      - alert: HighLatencyP99
        expr: |
          histogram_quantile(0.99,
              sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
          ) > 1.0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "P99 latency above 1s: {{ $value | printf \"%.2f\" }}s"

      # Service down
      - alert: ServiceDown
        expr: up{job="fastapi-app"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "{{ $labels.instance }} is down"

      # Database connection exhaustion
      - alert: DatabaseConnectionsHigh
        expr: pg_stat_activity_count{state="active"} > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Active DB connections: {{ $value }}/100"
```

---

## 7. Sentry를 이용한 오류 추적

Sentry(센트리)는 전체 컨텍스트와 함께 처리되지 않은 예외를 캡처한다: 스택 트레이스(stack trace), 요청 데이터, 사용자 정보, 그리고 오류로 이어진 이벤트 흔적인 브레드크럼(breadcrumb).

### FastAPI 통합

```python
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

sentry_sdk.init(
    dsn="https://examplePublicKey@o0.ingest.sentry.io/0",
    environment="production",
    release="api-server@1.2.0",
    traces_sample_rate=0.1,      # Sample 10% of transactions for performance
    profiles_sample_rate=0.1,    # Sample 10% for profiling
    integrations=[
        FastApiIntegration(),
        SqlalchemyIntegration(),
    ],
    # Filter out sensitive data
    before_send=filter_sensitive_data,
)

def filter_sensitive_data(event, hint):
    """Sentry로 전송하기 전에 민감한 정보를 제거한다."""
    if "request" in event and "headers" in event["request"]:
        headers = event["request"]["headers"]
        # Remove authorization headers
        event["request"]["headers"] = {
            k: v for k, v in headers.items()
            if k.lower() not in ("authorization", "cookie", "x-api-key")
        }
    return event
```

### Express.js 통합

```javascript
const Sentry = require("@sentry/node");

Sentry.init({
  dsn: "https://examplePublicKey@o0.ingest.sentry.io/0",
  environment: process.env.NODE_ENV,
  release: `api-server@${process.env.npm_package_version}`,
  tracesSampleRate: 0.1,
  integrations: [
    Sentry.expressIntegration(),
    Sentry.httpIntegration(),
  ],
});

const app = express();

// Sentry request handler must be the first middleware
app.use(Sentry.expressRequestHandler());

// ... routes ...

// Sentry error handler must be before any other error middleware
app.use(Sentry.expressErrorHandler());

// Custom error handler
app.use((err, req, res, next) => {
  res.status(500).json({ error: "Internal server error" });
});
```

### 컨텍스트 캡처

```python
# Add user context to all events in a request
@app.middleware("http")
async def sentry_user_context(request: Request, call_next):
    if hasattr(request.state, "user"):
        sentry_sdk.set_user({
            "id": request.state.user.id,
            "username": request.state.user.username,
            "email": request.state.user.email,
        })
    response = await call_next(request)
    return response

# Add breadcrumbs for debugging context
sentry_sdk.add_breadcrumb(
    category="payment",
    message=f"Processing payment for order {order_id}",
    level="info",
    data={"amount": 99.99, "currency": "USD"},
)

# Capture handled exceptions with extra context
try:
    result = await external_api.call()
except ExternalAPIError as e:
    sentry_sdk.capture_exception(e)
    # Continue with fallback logic
    result = get_cached_response()
```

---

## 8. 연습 문제

### 문제 1: 구조화 로깅 파이프라인

다음 조건을 갖춘 FastAPI 애플리케이션을 위한 완전한 로깅 파이프라인을 구현하라:
- JSON 출력으로 structlog 사용
- 모든 로그 항목에 요청 ID, 사용자 ID, 트레이스 ID 포함
- 민감한 필드 마스킹 (비밀번호, 토큰, 신용카드 번호)
- stdout에 기록 (컨테이너 로그 수집용)
- 타이밍이 포함된 요청/응답을 로깅하는 미들웨어 포함

### 문제 2: 커스텀 Prometheus 메트릭

다음 요구 사항을 갖춘 이커머스(e-commerce) API를 위한 Prometheus 메트릭을 설계하고 구현하라:
- 결제 방법별 주문 처리 시간 추적 (히스토그램)
- 제품 카테고리별 재고 수준 추적 (게이지)
- 이유별 결제 실패 시도 추적 (레이블이 있는 카운터)
- 장바구니 이탈률 추적 (카운터에서 유도)
- 각 메트릭을 위한 대시보드를 구동하는 PromQL 쿼리 작성

### 문제 3: 분산 추적 시나리오

세 가지 서비스가 있다: API 게이트웨이, 주문 서비스, 결제 서비스. 사용자가 주문을 생성하면 재고 확인과 결제 처리가 트리거된다. 세 서비스 모두에 대해 OpenTelemetry 계측을 구현하라 (FastAPI 사용):
- HTTP 호출 간에 트레이스 컨텍스트를 전파한다
- 비즈니스 로직에 대한 커스텀 스팬을 생성한다
- 디버깅을 위한 스팬 속성을 기록한다
- 적절한 스팬 상태로 오류를 처리한다

### 문제 4: 알림 규칙셋

프로덕션 API 서비스를 위한 완전한 Prometheus 알림 설정을 설계하라. 다음에 대한 알림을 포함한다:
- SLA 위반 (가용성 99.9%, P99 지연 시간 < 500ms)
- 리소스 고갈 (CPU, 메모리, 디스크, 연결 수)
- 비즈니스 이상 (주문율이 롤링 평균에서 50% 이상 하락)
- 보안 이벤트 (인증 실패율 급증)
각 알림에 대해 심각도, `for` 기간을 명시하고 런북(runbook) 개요를 작성하라.

### 문제 5: Docker Compose 관찰 가능성 스택

완전한 관찰 가능성 스택을 설정하는 Docker Compose 파일을 만들어라:
- Prometheus (메트릭 수집)
- Grafana (RED 대시보드가 미리 프로비저닝된 대시보드)
- Jaeger (트레이스 시각화)
- OpenTelemetry Collector (트레이스 수신, Jaeger로 내보내기)
- 세 기둥 모두로 계측된 샘플 FastAPI 애플리케이션

각 서비스의 설정 파일을 포함하고 전체 파이프라인이 엔드-투-엔드(end-to-end)로 작동하는지 확인하라.

---

## 참고 자료

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [structlog Documentation](https://www.structlog.org/)
- [pino Documentation](https://getpino.io/)
- [Sentry Documentation](https://docs.sentry.io/)
- [Google SRE Book: Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)
- [The RED Method](https://www.weave.works/blog/the-red-method-key-metrics-for-microservices-architecture/)
- [The USE Method](https://www.brendangregg.com/usemethod.html)

---

**이전**: [프로덕션 배포](./16_Production_Deployment.md) | **다음**: [프로젝트: REST API](./18_Project_REST_API.md)

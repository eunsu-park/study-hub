# 레슨 27: 분산 관측 가능성 (Distributed Observability)

[개요](./00_Overview.md) | [이전: 분산 테스트](./26_Distributed_Testing.md) | [다음: 캡스톤 — 분산 KV 스토어](./28_Capstone_Distributed_KV.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. 서비스 경계를 넘는 컨텍스트 전파(context propagation)를 사용한 분산 트레이싱(distributed tracing) 구현
2. 마이크로서비스를 통한 요청 추적을 위한 상관관계 ID(correlation ID) 체계 설계
3. 중앙 집중식 집계를 사용한 구조화된 분산 로깅(structured logging) 구축
4. 분산 시스템을 위한 메트릭 수집(metric collection)과 이상 감지(anomaly detection) 구현
5. 트레이스, 로그, 메트릭을 함께 사용하여 프로덕션 분산 시스템 디버그

---

## 목차

1. [관측 가능성 기초](#1-관측-가능성-기초)
2. [분산 트레이싱](#2-분산-트레이싱)
3. [컨텍스트 전파](#3-컨텍스트-전파)
4. [상관관계 ID](#4-상관관계-id)
5. [분산 로깅](#5-분산-로깅)
6. [메트릭 수집](#6-메트릭-수집)
7. [이상 감지](#7-이상-감지)
8. [분산 시스템 디버깅](#8-분산-시스템-디버깅)
9. [실제 관측 가능성 스택](#9-실제-관측-가능성-스택)
10. [요약 및 핵심 정리](#10-요약-및-핵심-정리)
11. [연습 문제](#11-연습-문제)
12. [참고 문헌](#12-참고-문헌)

---

## 1. 관측 가능성 기초

### 1.1 세 가지 기둥 (Three Pillars)

```python
import time
import json
import uuid
import random
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


def explain_three_pillars():
    """관측 가능성의 세 가지 기둥을 설명한다."""
    print("=== Three Pillars of Observability ===\n")

    pillars = {
        "Traces": {
            "what": "End-to-end request path across services",
            "answers": "Where did the request go? What was slow?",
            "tools": "Jaeger, Zipkin, AWS X-Ray, OpenTelemetry",
        },
        "Logs": {
            "what": "Timestamped text records from each service",
            "answers": "What happened? What was the error message?",
            "tools": "ELK Stack, Loki, CloudWatch Logs",
        },
        "Metrics": {
            "what": "Numeric measurements over time",
            "answers": "How many? How fast? What percentage?",
            "tools": "Prometheus, Datadog, CloudWatch Metrics",
        },
    }

    for name, info in pillars.items():
        print(f"  {name}:")
        for k, v in info.items():
            print(f"    {k}: {v}")
        print()


explain_three_pillars()
```

---

## 2. 분산 트레이싱

### 2.1 트레이스와 스팬 모델 (Trace and Span Model)

```python
@dataclass
class Span:
    """
    분산 트레이스(distributed trace) 내의 단일 작업 단위.

    스팬(span)은 서비스 내의 작업을 나타낸다:
    - HTTP 요청 핸들러
    - 데이터베이스 쿼리
    - 메시지 처리 단계
    """
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    service_name: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    duration_ms: float = 0.0
    status: str = "ok"  # ok, error
    tags: dict = field(default_factory=dict)
    logs: list = field(default_factory=list)

    def finish(self):
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000

    def log(self, message: str, fields: dict = None):
        self.logs.append({
            "timestamp": time.time(),
            "message": message,
            "fields": fields or {},
        })

    def set_tag(self, key: str, value: Any):
        self.tags[key] = value


class Tracer:
    """
    분산 트레이싱 시스템.

    스팬을 생성하고 관리하며, 서비스 경계를 넘어
    트레이스 컨텍스트를 전파하고, 완료된 스팬을 수집한다.
    """

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.active_spans: Dict[str, Span] = {}
        self.completed_spans: list[Span] = []

    def start_span(self, operation_name: str, parent: Optional[Span] = None,
                   trace_id: Optional[str] = None) -> Span:
        """새 스팬을 시작한다."""
        span = Span(
            trace_id=trace_id or parent.trace_id if parent else str(uuid.uuid4())[:16],
            span_id=str(uuid.uuid4())[:16],
            parent_span_id=parent.span_id if parent else None,
            operation_name=operation_name,
            service_name=self.service_name,
        )
        self.active_spans[span.span_id] = span
        return span

    def finish_span(self, span: Span):
        """스팬을 종료하고 기록한다."""
        span.finish()
        self.active_spans.pop(span.span_id, None)
        self.completed_spans.append(span)

    def inject_context(self, span: Span) -> dict:
        """
        트레이스 컨텍스트를 캐리어(예: HTTP 헤더)에 주입한다.

        이를 통해 트레이스가 서비스 경계를 넘어 계속될 수 있다.
        """
        return {
            "x-trace-id": span.trace_id,
            "x-span-id": span.span_id,
            "x-parent-span-id": span.parent_span_id or "",
        }

    def extract_context(self, carrier: dict) -> Tuple[str, str]:
        """캐리어에서 트레이스 컨텍스트를 추출한다."""
        trace_id = carrier.get("x-trace-id", "")
        parent_span_id = carrier.get("x-span-id", "")
        return trace_id, parent_span_id


class TraceCollector:
    """완전한 트레이스를 조합하는 중앙 집중식 트레이스 수집기."""

    def __init__(self):
        self.spans: list[Span] = []
        self.traces: Dict[str, list[Span]] = defaultdict(list)

    def collect(self, span: Span):
        self.spans.append(span)
        self.traces[span.trace_id].append(span)

    def get_trace(self, trace_id: str) -> list[Span]:
        spans = self.traces.get(trace_id, [])
        return sorted(spans, key=lambda s: s.start_time)

    def render_trace(self, trace_id: str) -> str:
        """트레이스를 시각적 트리로 렌더링한다."""
        spans = self.get_trace(trace_id)
        if not spans:
            return "No spans found"

        lines = [f"Trace: {trace_id}"]
        root_spans = [s for s in spans if s.parent_span_id is None]
        children = defaultdict(list)
        for s in spans:
            if s.parent_span_id:
                children[s.parent_span_id].append(s)

        def render_span(span, indent=0):
            prefix = "  " * indent + ("└─ " if indent > 0 else "")
            status = "✓" if span.status == "ok" else "✗"
            lines.append(
                f"{prefix}[{status}] {span.service_name}/{span.operation_name} "
                f"({span.duration_ms:.1f}ms)"
            )
            for child in sorted(children.get(span.span_id, []),
                              key=lambda s: s.start_time):
                render_span(child, indent + 1)

        for root in root_spans:
            render_span(root)

        return "\n".join(lines)


def demonstrate_distributed_tracing():
    """서비스 간 분산 트레이싱을 시연한다."""
    print("=== Distributed Tracing ===\n")

    collector = TraceCollector()

    # 서비스 A: API 게이트웨이
    tracer_a = Tracer("api-gateway")
    root_span = tracer_a.start_span("POST /orders")
    root_span.set_tag("http.method", "POST")
    root_span.set_tag("http.url", "/orders")

    # 서비스 A가 서비스 B를 호출
    headers = tracer_a.inject_context(root_span)
    time.sleep(0.01)

    # 서비스 B: 주문 서비스
    tracer_b = Tracer("order-service")
    trace_id, parent_id = tracer_b.extract_context(headers)
    span_b = tracer_b.start_span("createOrder", trace_id=trace_id)
    span_b.parent_span_id = parent_id

    # 서비스 B가 서비스 C(데이터베이스)를 호출
    span_db = tracer_b.start_span("INSERT orders", parent=span_b)
    span_db.set_tag("db.type", "postgresql")
    time.sleep(0.005)
    tracer_b.finish_span(span_db)

    # 서비스 B가 서비스 D(결제)를 호출
    headers_d = tracer_b.inject_context(span_b)
    tracer_d = Tracer("payment-service")
    trace_id_d, parent_id_d = tracer_d.extract_context(headers_d)
    span_d = tracer_d.start_span("chargeCard", trace_id=trace_id_d)
    span_d.parent_span_id = parent_id_d
    time.sleep(0.015)
    tracer_d.finish_span(span_d)

    time.sleep(0.005)
    tracer_b.finish_span(span_b)

    time.sleep(0.002)
    tracer_a.finish_span(root_span)

    # 모든 스팬 수집
    for tracer in [tracer_a, tracer_b, tracer_d]:
        for span in tracer.completed_spans:
            collector.collect(span)

    # 트레이스 렌더링
    print(collector.render_trace(root_span.trace_id))
    print(f"\nTotal spans: {len(collector.spans)}")
    total_time = root_span.duration_ms
    print(f"Total time: {total_time:.1f}ms")


demonstrate_distributed_tracing()
```

---

## 3. 컨텍스트 전파

### 3.1 W3C Trace Context

```python
class W3CTraceContext:
    """
    W3C Trace Context 전파 형식.

    traceparent: {version}-{trace-id}-{parent-id}-{trace-flags}
    tracestate: 벤더별 키-값 쌍
    """

    VERSION = "00"

    @staticmethod
    def create_traceparent(trace_id: str, span_id: str,
                           sampled: bool = True) -> str:
        flags = "01" if sampled else "00"
        return f"{W3CTraceContext.VERSION}-{trace_id}-{span_id}-{flags}"

    @staticmethod
    def parse_traceparent(header: str) -> Optional[dict]:
        parts = header.split("-")
        if len(parts) != 4:
            return None
        return {
            "version": parts[0],
            "trace_id": parts[1],
            "parent_id": parts[2],
            "sampled": parts[3] == "01",
        }

    @staticmethod
    def create_tracestate(entries: dict) -> str:
        return ",".join(f"{k}={v}" for k, v in entries.items())


def demonstrate_context_propagation():
    """W3C 트레이스 컨텍스트 전파를 시연한다."""
    print("=== Context Propagation ===\n")

    trace_id = uuid.uuid4().hex[:32]
    span_id = uuid.uuid4().hex[:16]

    # traceparent 헤더 생성
    traceparent = W3CTraceContext.create_traceparent(trace_id, span_id)
    print(f"traceparent: {traceparent}")

    # 다시 파싱
    parsed = W3CTraceContext.parse_traceparent(traceparent)
    print(f"Parsed: {json.dumps(parsed, indent=2)}")

    # 벤더별 데이터를 위한 tracestate
    tracestate = W3CTraceContext.create_tracestate({
        "vendor1": "value1",
        "vendor2": "value2",
    })
    print(f"tracestate: {tracestate}")


demonstrate_context_propagation()
```

---

## 4. 상관관계 ID

### 4.1 요청 상관관계 (Request Correlation)

```python
class CorrelationContext:
    """
    분산 시스템을 통한 요청 추적을 위한 상관관계 컨텍스트.

    트레이스 컨텍스트(스팬을 추적)와 달리, 상관관계 컨텍스트는
    요청 ID, 사용자 ID, 세션 ID와 같은 비즈니스 레벨 식별자를
    전체 요청 체인을 통해 전달한다.
    """

    def __init__(self):
        self.request_id: str = str(uuid.uuid4())
        self.correlation_id: str = str(uuid.uuid4())[:8]
        self.user_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.baggage: dict = {}

    def to_headers(self) -> dict:
        headers = {
            "x-request-id": self.request_id,
            "x-correlation-id": self.correlation_id,
        }
        if self.user_id:
            headers["x-user-id"] = self.user_id
        for k, v in self.baggage.items():
            headers[f"x-baggage-{k}"] = v
        return headers

    @classmethod
    def from_headers(cls, headers: dict) -> 'CorrelationContext':
        ctx = cls()
        ctx.request_id = headers.get("x-request-id", ctx.request_id)
        ctx.correlation_id = headers.get("x-correlation-id", ctx.correlation_id)
        ctx.user_id = headers.get("x-user-id")
        for k, v in headers.items():
            if k.startswith("x-baggage-"):
                ctx.baggage[k[10:]] = v
        return ctx


def demonstrate_correlation_ids():
    """상관관계 ID 전파를 시연한다."""
    print("=== Correlation IDs ===\n")

    # 클라이언트가 초기 컨텍스트를 생성
    ctx = CorrelationContext()
    ctx.user_id = "user-123"
    ctx.baggage["tenant"] = "acme-corp"

    headers = ctx.to_headers()
    print("Request headers:")
    for k, v in headers.items():
        print(f"  {k}: {v}")

    # 다운스트림 서비스가 컨텍스트를 추출
    downstream_ctx = CorrelationContext.from_headers(headers)
    print(f"\nDownstream service sees:")
    print(f"  correlation_id: {downstream_ctx.correlation_id}")
    print(f"  user_id: {downstream_ctx.user_id}")
    print(f"  tenant: {downstream_ctx.baggage.get('tenant')}")

    # 이 요청의 모든 로그가 상관관계 ID를 공유
    print(f"\n  All log entries tagged with correlation_id={downstream_ctx.correlation_id}")


demonstrate_correlation_ids()
```

---

## 5. 분산 로깅

### 5.1 구조화된 로깅 (Structured Logging)

```python
class StructuredLogger:
    """
    분산 시스템을 위한 구조화된 로깅.

    모든 로그 항목이 상관관계, 타이밍, 컨텍스트를 위한
    일관된 필드가 포함된 JSON 형식이다.
    """

    def __init__(self, service_name: str, instance_id: str):
        self.service_name = service_name
        self.instance_id = instance_id
        self.logs: list[dict] = []

    def log(self, level: str, message: str, context: Optional[CorrelationContext] = None,
            **extra):
        entry = {
            "timestamp": time.time(),
            "level": level,
            "service": self.service_name,
            "instance": self.instance_id,
            "message": message,
        }
        if context:
            entry["correlation_id"] = context.correlation_id
            entry["request_id"] = context.request_id
            entry["user_id"] = context.user_id

        entry.update(extra)
        self.logs.append(entry)
        return entry

    def info(self, message: str, **kwargs):
        return self.log("INFO", message, **kwargs)

    def error(self, message: str, **kwargs):
        return self.log("ERROR", message, **kwargs)

    def warn(self, message: str, **kwargs):
        return self.log("WARN", message, **kwargs)


class LogAggregator:
    """
    중앙 집중식 로그 집계기(log aggregator).

    모든 서비스에서 로그를 수집하고
    상관관계 기반 검색을 제공한다.
    """

    def __init__(self):
        self.logs: list[dict] = []

    def ingest(self, logs: list[dict]):
        self.logs.extend(logs)

    def search_by_correlation(self, correlation_id: str) -> list[dict]:
        return [
            log for log in self.logs
            if log.get("correlation_id") == correlation_id
        ]

    def search_by_level(self, level: str, since: float = 0) -> list[dict]:
        return [
            log for log in self.logs
            if log.get("level") == level and log.get("timestamp", 0) >= since
        ]

    def timeline(self, correlation_id: str) -> list[dict]:
        logs = self.search_by_correlation(correlation_id)
        return sorted(logs, key=lambda l: l.get("timestamp", 0))


def demonstrate_distributed_logging():
    """분산 구조화된 로깅을 시연한다."""
    print("=== Distributed Logging ===\n")

    aggregator = LogAggregator()
    ctx = CorrelationContext()
    ctx.user_id = "user-456"

    # API 게이트웨이 로그
    gw_logger = StructuredLogger("api-gateway", "gw-01")
    gw_logger.info("Request received", context=ctx, path="/api/orders", method="POST")
    gw_logger.info("Routing to order-service", context=ctx)

    # 주문 서비스 로그
    order_logger = StructuredLogger("order-service", "order-02")
    order_logger.info("Processing order", context=ctx, order_total=99.99)
    order_logger.info("Calling payment service", context=ctx)

    # 결제 서비스 로그
    pay_logger = StructuredLogger("payment-service", "pay-01")
    pay_logger.info("Charging card", context=ctx, amount=99.99)
    pay_logger.error("Payment declined", context=ctx, reason="insufficient_funds")

    # 주문 서비스가 오류를 수신
    order_logger.error("Payment failed", context=ctx, error="payment_declined")

    # 모든 로그 집계
    for logger in [gw_logger, order_logger, pay_logger]:
        aggregator.ingest(logger.logs)

    # 상관관계 ID로 검색
    print(f"Request timeline (correlation_id={ctx.correlation_id}):")
    for log in aggregator.timeline(ctx.correlation_id):
        level = log["level"]
        svc = log["service"]
        msg = log["message"]
        print(f"  [{level:5s}] {svc:20s}: {msg}")

    # 오류 검색
    errors = aggregator.search_by_level("ERROR")
    print(f"\nRecent errors: {len(errors)}")


demonstrate_distributed_logging()
```

---

## 6. 메트릭 수집

### 6.1 메트릭 유형

```python
class Counter:
    """단조 증가 카운터(monotonically increasing counter)."""
    def __init__(self, name: str, labels: dict = None):
        self.name = name
        self.labels = labels or {}
        self.value: float = 0

    def inc(self, amount: float = 1.0):
        self.value += amount


class Gauge:
    """올라가고 내려갈 수 있는 값."""
    def __init__(self, name: str, labels: dict = None):
        self.name = name
        self.labels = labels or {}
        self.value: float = 0

    def set(self, value: float):
        self.value = value

    def inc(self, amount: float = 1.0):
        self.value += amount

    def dec(self, amount: float = 1.0):
        self.value -= amount


class Histogram:
    """구성 가능한 버킷을 사용한 값의 분포."""
    def __init__(self, name: str, buckets: list[float] = None, labels: dict = None):
        self.name = name
        self.labels = labels or {}
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        self.counts: Dict[float, int] = {b: 0 for b in self.buckets}
        self.counts[float('inf')] = 0
        self.sum: float = 0
        self.count: int = 0

    def observe(self, value: float):
        self.sum += value
        self.count += 1
        for bucket in self.buckets:
            if value <= bucket:
                self.counts[bucket] += 1
        self.counts[float('inf')] += 1

    def percentile(self, p: float) -> float:
        """근사 백분위수를 계산한다."""
        target = self.count * p
        cumulative = 0
        for bucket in sorted(self.counts.keys()):
            cumulative += self.counts[bucket]
            if cumulative >= target:
                return bucket
        return self.buckets[-1]


class MetricsRegistry:
    """서비스를 위한 중앙 메트릭 레지스트리."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}

    def counter(self, name: str, **labels) -> Counter:
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        if key not in self.counters:
            self.counters[key] = Counter(name, labels)
        return self.counters[key]

    def gauge(self, name: str, **labels) -> Gauge:
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        if key not in self.gauges:
            self.gauges[key] = Gauge(name, labels)
        return self.gauges[key]

    def histogram(self, name: str, buckets: list[float] = None, **labels) -> Histogram:
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        if key not in self.histograms:
            self.histograms[key] = Histogram(name, buckets, labels)
        return self.histograms[key]

    def snapshot(self) -> dict:
        return {
            "service": self.service_name,
            "counters": {k: c.value for k, c in self.counters.items()},
            "gauges": {k: g.value for k, g in self.gauges.items()},
            "histograms": {
                k: {"count": h.count, "sum": h.sum,
                    "p50": h.percentile(0.5), "p99": h.percentile(0.99)}
                for k, h in self.histograms.items()
            },
        }


def demonstrate_metrics():
    """분산 시스템을 위한 메트릭 수집을 시연한다."""
    print("=== Metrics Collection ===\n")

    registry = MetricsRegistry("order-service")

    # 요청 카운터
    req_counter = registry.counter("http_requests_total", method="POST", path="/orders")
    err_counter = registry.counter("http_errors_total", method="POST", path="/orders")

    # 활성 연결 게이지
    connections = registry.gauge("active_connections")

    # 요청 지속시간 히스토그램
    duration = registry.histogram("http_request_duration_seconds",
                                   buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0])

    # 트래픽 시뮬레이션
    for _ in range(1000):
        req_counter.inc()
        connections.inc()

        # 요청 지속시간 시뮬레이션
        latency = random.expovariate(10)  # ~100ms 평균
        duration.observe(latency)

        if random.random() < 0.02:  # 2% 오류율
            err_counter.inc()

        connections.dec()

    snapshot = registry.snapshot()
    print("Metrics snapshot:")
    print(f"  Requests: {snapshot['counters']}")
    print(f"  Connections: {snapshot['gauges']}")
    print(f"  Latency: {snapshot['histograms']}")


demonstrate_metrics()
```

---

## 7. 이상 감지

### 7.1 통계적 이상 감지

```python
class AnomalyDetector:
    """
    분산 시스템 메트릭을 위한 단순 이상 감지(anomaly detection).

    이동 평균(moving average)과 표준편차(standard deviation)를
    사용하여 시계열 데이터에서 이상을 감지한다.
    """

    def __init__(self, window_size: int = 30, threshold_sigmas: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold_sigmas
        self.values: list[float] = []
        self.anomalies: list[dict] = []

    def observe(self, value: float, timestamp: float = None) -> bool:
        """값을 기록하고 이상 여부를 확인한다."""
        self.values.append(value)
        ts = timestamp or time.time()

        if len(self.values) < self.window_size:
            return False

        window = self.values[-self.window_size:]
        mean = sum(window) / len(window)
        std_dev = (sum((x - mean) ** 2 for x in window) / len(window)) ** 0.5

        if std_dev == 0:
            return False

        z_score = abs(value - mean) / std_dev
        is_anomaly = z_score > self.threshold

        if is_anomaly:
            self.anomalies.append({
                "timestamp": ts,
                "value": value,
                "mean": round(mean, 3),
                "std_dev": round(std_dev, 3),
                "z_score": round(z_score, 2),
            })

        return is_anomaly


def demonstrate_anomaly_detection():
    """분산 시스템 메트릭에 대한 이상 감지를 시연한다."""
    print("=== Anomaly Detection ===\n")

    detector = AnomalyDetector(window_size=20, threshold_sigmas=2.5)

    # 정상 지연시간: ~100ms, 표준편차 10ms
    for i in range(50):
        latency = random.gauss(100, 10)
        is_anomaly = detector.observe(latency)

    # 이상 주입: 갑작스러운 지연시간 급증
    for i in range(5):
        latency = random.gauss(500, 50)  # 정상의 5배
        is_anomaly = detector.observe(latency)
        if is_anomaly:
            print(f"  ANOMALY at t={50+i}: latency={latency:.0f}ms "
                  f"(z={detector.anomalies[-1]['z_score']})")

    # 복구
    for i in range(20):
        latency = random.gauss(100, 10)
        detector.observe(latency)

    print(f"\nTotal anomalies detected: {len(detector.anomalies)}")


demonstrate_anomaly_detection()
```

---

## 8. 분산 시스템 디버깅

### 8.1 디버깅 워크플로우

```python
class DistributedDebugger:
    """
    분산 시스템을 위한 디버깅 툴킷.

    트레이스, 로그, 메트릭을 결합하여 문제를 진단한다.
    """

    def __init__(self, trace_collector: TraceCollector, log_aggregator: LogAggregator):
        self.traces = trace_collector
        self.logs = log_aggregator

    def diagnose_slow_request(self, trace_id: str) -> dict:
        """트레이스 데이터를 사용하여 느린 요청을 진단한다."""
        spans = self.traces.get_trace(trace_id)
        if not spans:
            return {"error": "Trace not found"}

        root = spans[0]
        total_time = root.duration_ms

        # 가장 느린 스팬 찾기
        slowest = max(spans, key=lambda s: s.duration_ms)

        # 오류 스팬 찾기
        errors = [s for s in spans if s.status == "error"]

        # 크리티컬 경로(critical path) 찾기
        critical_path = self._find_critical_path(spans)

        return {
            "trace_id": trace_id,
            "total_time_ms": total_time,
            "num_spans": len(spans),
            "slowest_span": {
                "service": slowest.service_name,
                "operation": slowest.operation_name,
                "duration_ms": slowest.duration_ms,
            },
            "errors": [
                {"service": e.service_name, "operation": e.operation_name}
                for e in errors
            ],
            "critical_path": critical_path,
        }

    def _find_critical_path(self, spans: list[Span]) -> list[dict]:
        """크리티컬 경로(가장 긴 순차 체인)를 찾는다."""
        children = defaultdict(list)
        for s in spans:
            if s.parent_span_id:
                children[s.parent_span_id].append(s)

        def longest_path(span):
            child_spans = children.get(span.span_id, [])
            if not child_spans:
                return [span]
            longest = max(
                (longest_path(c) for c in child_spans),
                key=lambda p: sum(s.duration_ms for s in p),
            )
            return [span] + longest

        root = next((s for s in spans if s.parent_span_id is None), spans[0])
        path = longest_path(root)
        return [{"service": s.service_name, "op": s.operation_name,
                 "ms": round(s.duration_ms, 1)} for s in path]

    def correlate_error(self, correlation_id: str) -> dict:
        """실패한 요청에 대해 모든 서비스의 로그를 상관시킨다."""
        logs = self.logs.search_by_correlation(correlation_id)
        errors = [l for l in logs if l.get("level") == "ERROR"]

        services_involved = list(set(l.get("service") for l in logs))
        error_chain = []
        for error in errors:
            error_chain.append({
                "service": error.get("service"),
                "message": error.get("message"),
                "timestamp": error.get("timestamp"),
            })

        return {
            "correlation_id": correlation_id,
            "services_involved": services_involved,
            "total_log_entries": len(logs),
            "error_chain": sorted(error_chain, key=lambda e: e.get("timestamp", 0)),
        }


def demonstrate_debugging():
    """분산 시스템 디버깅을 시연한다."""
    print("=== Debugging Distributed Systems ===\n")

    # 설정
    collector = TraceCollector()
    aggregator = LogAggregator()
    debugger = DistributedDebugger(collector, aggregator)

    # 느린 요청 트레이스 시뮬레이션
    trace_id = "abc123"
    spans = [
        Span(trace_id=trace_id, span_id="s1", operation_name="POST /api",
             service_name="gateway", duration_ms=250),
        Span(trace_id=trace_id, span_id="s2", parent_span_id="s1",
             operation_name="createOrder", service_name="order-svc", duration_ms=200),
        Span(trace_id=trace_id, span_id="s3", parent_span_id="s2",
             operation_name="INSERT", service_name="postgres", duration_ms=150),
        Span(trace_id=trace_id, span_id="s4", parent_span_id="s2",
             operation_name="charge", service_name="payment-svc", duration_ms=30),
    ]
    for span in spans:
        collector.collect(span)

    diagnosis = debugger.diagnose_slow_request(trace_id)
    print("Slow request diagnosis:")
    print(f"  Total time: {diagnosis['total_time_ms']}ms")
    print(f"  Slowest: {diagnosis['slowest_span']}")
    print(f"  Critical path:")
    for step in diagnosis['critical_path']:
        print(f"    {step['service']}/{step['op']}: {step['ms']}ms")


demonstrate_debugging()
```

---

## 9. 실제 관측 가능성 스택

```python
def compare_observability_stacks():
    """실제 관측 가능성 스택을 비교한다."""
    print("=== Observability Stack Comparison ===\n")

    stacks = [
        {"name": "OpenTelemetry + Jaeger + Prometheus + ELK",
         "type": "Open source",
         "traces": "Jaeger/Tempo", "metrics": "Prometheus", "logs": "Elasticsearch"},
        {"name": "Datadog",
         "type": "SaaS",
         "traces": "Datadog APM", "metrics": "Datadog Metrics", "logs": "Datadog Logs"},
        {"name": "Grafana Stack (LGTM)",
         "type": "Open source",
         "traces": "Tempo", "metrics": "Mimir", "logs": "Loki"},
        {"name": "AWS Native",
         "type": "Cloud",
         "traces": "X-Ray", "metrics": "CloudWatch", "logs": "CloudWatch Logs"},
    ]

    for stack in stacks:
        print(f"  {stack['name']} ({stack['type']}):")
        print(f"    Traces: {stack['traces']}")
        print(f"    Metrics: {stack['metrics']}")
        print(f"    Logs: {stack['logs']}")
        print()


compare_observability_stacks()
```

---

## 10. 요약 및 핵심 정리

### 관측 가능성 체크리스트

> **분산 관측 가능성 체크리스트 (DISTRIBUTED OBSERVABILITY CHECKLIST)**
>
> ☐ W3C 컨텍스트 전파를 사용한 분산 트레이싱
> ☐ 모든 서비스 간 호출에 상관관계 ID
> ☐ 일관된 필드가 포함된 구조화된 JSON 로깅
> ☐ 서비스별 RED 메트릭 (Rate, Errors, Duration)
> ☐ 전문 검색이 가능한 중앙 집중식 로그 집계
> ☐ 디버깅을 위한 트레이스-로그 상관관계
> ☐ 지연시간과 오류율에 대한 이상 감지
> ☐ 각 서비스와 전체 시스템에 대한 대시보드

### 핵심 원칙

1. **상관관계가 핵심이다**: 상관관계 ID 없이 서비스 간 디버깅은 불가능하다.
2. **구조화된 로그 > 비구조화 로그**: JSON 로그로 머신 파싱과 검색이 가능하다.
3. **트레이스가 경로를, 로그가 상세를 보여준다**: 트레이스로 느린 서비스를 찾고, 로그로 이유를 찾는다.
4. **모든 서비스에 RED 메트릭**: Rate(속도), Errors(오류), Duration(지속시간) — 최소 필수 메트릭이다.
5. **OpenTelemetry가 표준이다**: 벤더 중립적이고, 널리 지원되며, 미래를 보장한다.

---

## 11. 연습 문제

### 문제 1: 트레이스 분석

4개 서비스에 걸쳐 12개 스팬이 있는 트레이스가 주어졌을 때, 크리티컬 경로(critical path)를 식별하고 각 서비스에서 소비된 시간의 백분율을 계산한다.

### 문제 2: 로그 상관관계 설계

다음을 포함하는 구조화된 로깅 형식을 설계한다: timestamp, level, service, instance, correlation_id, trace_id, span_id, user_id, 그리고 임의의 키-값 필드. 로그 집계 검색을 구현한다.

### 문제 3: 메트릭 대시보드

3계층 애플리케이션(web → API → DB)을 위한 대시보드를 설계한다. 다음을 포함한다: 요청 속도, 오류율, 지연시간 백분위수, 활성 연결, 큐 깊이. 경고 임계값을 정의한다.

### 문제 4: 구현 도전

다음을 포함하는 관측 가능성 라이브러리를 구축한다: Tracer(스팬 생성과 컨텍스트 전파), StructuredLogger(상관관계가 포함된 JSON 로깅), MetricsRegistry(카운터, 게이지, 히스토그램), 모두 상관관계 ID로 연결됨.

### 문제 5: 디버깅 연습

공통 correlation_id를 가진 5개 서비스의 로그에서 간헐적 500 오류가 나타났다. 체계적인 디버깅 절차를 설계한다. 어떤 정보가 필요한가? 어떤 순서로 조사하는가?

---

## 12. 참고 문헌

1. Sridharan, C. (2018). *Distributed Systems Observability*. O'Reilly Media.
2. OpenTelemetry documentation: https://opentelemetry.io/docs/
3. Sigelman, B. et al. (2010). "Dapper, a Large-Scale Distributed Systems Tracing Infrastructure." Google Technical Report.
4. W3C Trace Context specification: https://www.w3.org/TR/trace-context/
5. Beyer, B. et al. (2016). *Site Reliability Engineering*. O'Reilly Media.
6. Majors, C. et al. (2022). *Observability Engineering*. O'Reilly Media.
7. Kleppmann, M. (2017). *Designing Data-Intensive Applications*, Ch. 4. O'Reilly Media.

---

[다음: 레슨 28 — 캡스톤: 분산 KV 스토어](./28_Capstone_Distributed_KV.md)

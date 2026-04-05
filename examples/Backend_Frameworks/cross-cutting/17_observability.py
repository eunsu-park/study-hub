"""
Observability — Logging, Metrics, Distributed Tracing
Demonstrates: structured JSON logging, Prometheus-style metrics collection,
              trace context propagation, and middleware instrumentation.

Run: pip install fastapi uvicorn && uvicorn 17_observability:app --reload
"""

from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
import json
import time
import uuid
import logging
from collections import defaultdict
from datetime import datetime, timezone
from functools import wraps

app = FastAPI(title="Observability Demo", version="1.0.0")


# --- 1. Structured JSON Logger ---

class StructuredLogger:
    """Logger that outputs machine-parseable JSON lines."""

    def __init__(self, service: str, version: str = "1.0.0"):
        self.service = service
        self.version = version
        self._context: dict = {}

    def bind(self, **kwargs) -> "StructuredLogger":
        """Create a child logger with additional context fields."""
        child = StructuredLogger(self.service, self.version)
        child._context = {**self._context, **kwargs}
        return child

    def _emit(self, level: str, message: str, **extra):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "service": self.service,
            "version": self.version,
            "message": message,
            **self._context,
            **extra,
        }
        print(json.dumps(entry))

    def info(self, msg: str, **kw): self._emit("INFO", msg, **kw)
    def warn(self, msg: str, **kw): self._emit("WARN", msg, **kw)
    def error(self, msg: str, **kw): self._emit("ERROR", msg, **kw)


logger = StructuredLogger("observability-demo", "1.0.0")


# --- 2. Prometheus-Style Metrics ---

class MetricsCollector:
    """Simple in-process metrics collector (counter + histogram)."""

    def __init__(self):
        self.counters: dict[str, float] = defaultdict(float)
        self.histograms: dict[str, list[float]] = defaultdict(list)

    def inc(self, name: str, value: float = 1.0, labels: dict | None = None):
        key = self._key(name, labels)
        self.counters[key] += value

    def observe(self, name: str, value: float, labels: dict | None = None):
        key = self._key(name, labels)
        self.histograms[key].append(value)

    def _key(self, name: str, labels: dict | None) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def exposition(self) -> str:
        """Render metrics in Prometheus text format."""
        lines = []
        for key, val in sorted(self.counters.items()):
            lines.append(f"{key} {val}")
        for key, values in sorted(self.histograms.items()):
            if values:
                lines.append(f"{key}_count {len(values)}")
                lines.append(f"{key}_sum {sum(values):.4f}")
                lines.append(f"{key}_avg {sum(values)/len(values):.4f}")
        return "\n".join(lines) + "\n"


metrics = MetricsCollector()


# --- 3. Trace Context Propagation ---

class TraceContext:
    """W3C-style trace context for distributed tracing."""

    def __init__(self, trace_id: str | None = None, parent_span_id: str | None = None):
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self.span_id = uuid.uuid4().hex[:8]
        self.parent_span_id = parent_span_id

    def child(self) -> "TraceContext":
        return TraceContext(trace_id=self.trace_id, parent_span_id=self.span_id)

    def to_header(self) -> str:
        return f"00-{self.trace_id}-{self.span_id}-01"

    @classmethod
    def from_header(cls, header: str) -> "TraceContext":
        parts = header.split("-")
        if len(parts) == 4:
            return cls(trace_id=parts[1], parent_span_id=parts[2])
        return cls()


# --- 4. Middleware: Request Metrics + Tracing ---

@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    # Extract or create trace context
    traceparent = request.headers.get("traceparent", "")
    ctx = TraceContext.from_header(traceparent) if traceparent else TraceContext()

    # Bind trace IDs to logger
    req_logger = logger.bind(trace_id=ctx.trace_id, span_id=ctx.span_id)
    req_logger.info("request_start", method=request.method, path=str(request.url.path))

    start = time.perf_counter()
    response: Response = await call_next(request)
    duration = time.perf_counter() - start

    # Record metrics
    labels = {"method": request.method, "path": request.url.path, "status": str(response.status_code)}
    metrics.inc("http_requests_total", labels=labels)
    metrics.observe("http_request_duration_seconds", duration, labels=labels)

    # Propagate trace context in response
    response.headers["traceparent"] = ctx.to_header()
    req_logger.info("request_end", status=response.status_code, duration_ms=round(duration * 1000, 2))
    return response


# --- 5. Metrics Endpoint ---

@app.get("/metrics")
async def get_metrics():
    return Response(content=metrics.exposition(), media_type="text/plain")


@app.get("/")
async def root():
    return {"message": "Observability demo"}


@app.get("/slow")
async def slow_endpoint():
    """Intentionally slow endpoint for testing latency metrics."""
    import asyncio
    await asyncio.sleep(0.5)
    return {"message": "done"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

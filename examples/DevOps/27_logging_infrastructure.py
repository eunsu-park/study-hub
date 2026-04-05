#!/usr/bin/env python3
"""Example: Logging Infrastructure — Pipeline, Parsing & Cost Optimization

Demonstrates production logging infrastructure: log pipeline architecture
(collect, parse, route, store), multi-format parser, log sampling and
filtering for cost control, and log-based metric extraction.
Related lesson: 10_Logging_and_Log_Management.md (infrastructure focus)
"""

# =============================================================================
# WHY LOGGING INFRASTRUCTURE?
# At scale, logs are the largest telemetry data source (often 10-100x the
# volume of metrics and traces). Without a proper pipeline — parsing,
# filtering, sampling, and routing — costs explode and signal drowns in
# noise. This example builds the core components of a Fluentd/Vector-style
# log pipeline.
# =============================================================================

import json
import re
import time
import random
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional


# =============================================================================
# 1. LOG RECORD MODEL
# =============================================================================

@dataclass
class LogRecord:
    """A parsed, structured log record."""
    timestamp: float
    level: str
    message: str
    source: str = ""
    service: str = ""
    fields: dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    size_bytes: int = 0

    def to_json(self) -> str:
        d = {
            "timestamp": datetime.fromtimestamp(self.timestamp, timezone.utc).isoformat(),
            "level": self.level,
            "message": self.message,
            "source": self.source,
            "service": self.service,
            **self.fields,
        }
        return json.dumps(d)


# =============================================================================
# 2. MULTI-FORMAT LOG PARSER
# =============================================================================

# Common log format patterns
PATTERNS = {
    "json": None,  # Native JSON parsing
    "nginx": re.compile(
        r'(?P<ip>\S+) \S+ \S+ \[(?P<time>[^\]]+)\] "(?P<method>\S+) '
        r'(?P<path>\S+) \S+" (?P<status>\d+) (?P<bytes>\d+)'
    ),
    "syslog": re.compile(
        r'(?P<timestamp>\w{3}\s+\d+\s+\S+) (?P<host>\S+) '
        r'(?P<program>\S+?)(?:\[(?P<pid>\d+)\])?: (?P<message>.*)'
    ),
    "python": re.compile(
        r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) '
        r'(?P<level>\w+)\s+(?P<logger>\S+)\s+(?P<message>.*)'
    ),
}


def parse_log(raw_line: str) -> Optional[LogRecord]:
    """Auto-detect format and parse a log line."""
    raw_line = raw_line.strip()
    if not raw_line:
        return None

    # Try JSON first
    if raw_line.startswith("{"):
        try:
            data = json.loads(raw_line)
            return LogRecord(
                timestamp=time.time(),
                level=data.get("level", "INFO").upper(),
                message=data.get("message", data.get("msg", "")),
                service=data.get("service", ""),
                fields={k: v for k, v in data.items()
                        if k not in ("level", "message", "msg", "service")},
                raw=raw_line,
                size_bytes=len(raw_line),
            )
        except json.JSONDecodeError:
            pass

    # Try regex patterns
    for fmt_name, pattern in PATTERNS.items():
        if pattern is None:
            continue
        match = pattern.match(raw_line)
        if match:
            groups = match.groupdict()
            return LogRecord(
                timestamp=time.time(),
                level=groups.get("level", groups.get("status", "INFO")).upper(),
                message=groups.get("message", groups.get("path", raw_line)),
                source=fmt_name,
                fields=groups,
                raw=raw_line,
                size_bytes=len(raw_line),
            )

    # Fallback: treat as plain text
    return LogRecord(
        timestamp=time.time(), level="INFO", message=raw_line,
        source="plaintext", raw=raw_line, size_bytes=len(raw_line),
    )


# =============================================================================
# 3. LOG PIPELINE (FILTER, TRANSFORM, ROUTE)
# =============================================================================

@dataclass
class PipelineStats:
    """Statistics for the log pipeline."""
    received: int = 0
    parsed: int = 0
    filtered_out: int = 0
    sampled_out: int = 0
    routed: dict[str, int] = field(default_factory=dict)
    bytes_in: int = 0
    bytes_out: int = 0


@dataclass
class LogPipeline:
    """A configurable log processing pipeline."""
    name: str
    filters: list[Callable[[LogRecord], bool]] = field(default_factory=list)
    transforms: list[Callable[[LogRecord], LogRecord]] = field(default_factory=list)
    routes: dict[str, Callable[[LogRecord], bool]] = field(default_factory=dict)
    sample_rate: float = 1.0  # 1.0 = keep all, 0.1 = keep 10%
    stats: PipelineStats = field(default_factory=PipelineStats)

    def process(self, raw_lines: list[str]) -> dict[str, list[LogRecord]]:
        """Process raw log lines through the pipeline."""
        output: dict[str, list[LogRecord]] = {name: [] for name in self.routes}
        output["_default"] = []

        for line in raw_lines:
            self.stats.received += 1
            self.stats.bytes_in += len(line)

            # Parse
            record = parse_log(line)
            if not record:
                continue
            self.stats.parsed += 1

            # Filter
            if not all(f(record) for f in self.filters):
                self.stats.filtered_out += 1
                continue

            # Sample
            if self.sample_rate < 1.0 and random.random() > self.sample_rate:
                self.stats.sampled_out += 1
                continue

            # Transform
            for transform in self.transforms:
                record = transform(record)

            # Route
            routed = False
            for route_name, matcher in self.routes.items():
                if matcher(record):
                    output[route_name].append(record)
                    self.stats.routed[route_name] = self.stats.routed.get(route_name, 0) + 1
                    self.stats.bytes_out += record.size_bytes
                    routed = True
                    break
            if not routed:
                output["_default"].append(record)
                self.stats.bytes_out += record.size_bytes

        return output


# =============================================================================
# 4. LOG-BASED METRIC EXTRACTION
# =============================================================================

@dataclass
class LogMetricExtractor:
    """Extract metrics from log streams (like Loki's metric queries)."""
    counters: dict[str, int] = field(default_factory=dict)
    rates: dict[str, list[float]] = field(default_factory=dict)

    def count_by(self, records: list[LogRecord], field: str) -> dict[str, int]:
        """Count log records grouped by a field value."""
        counts: dict[str, int] = {}
        for r in records:
            key = r.fields.get(field, getattr(r, field, "unknown"))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def extract_rate(self, records: list[LogRecord],
                     window_seconds: float = 60) -> float:
        """Calculate log rate (records per second)."""
        if len(records) < 2:
            return 0.0
        time_span = records[-1].timestamp - records[0].timestamp
        if time_span <= 0:
            return float(len(records))
        return len(records) / time_span

    def extract_error_ratio(self, records: list[LogRecord]) -> float:
        """Calculate error log ratio."""
        if not records:
            return 0.0
        errors = sum(1 for r in records if r.level in ("ERROR", "FATAL", "CRITICAL"))
        return errors / len(records)


# =============================================================================
# 5. COST ESTIMATOR
# =============================================================================

def estimate_logging_costs(
    daily_gb: float,
    retention_days: int = 30,
    ingestion_cost_per_gb: float = 0.50,
    storage_cost_per_gb: float = 0.03,
) -> dict[str, Any]:
    """Estimate logging infrastructure costs."""
    monthly_ingestion = daily_gb * 30 * ingestion_cost_per_gb
    total_stored = daily_gb * retention_days
    monthly_storage = total_stored * storage_cost_per_gb
    return {
        "daily_volume_gb": daily_gb,
        "monthly_ingestion_cost": round(monthly_ingestion, 2),
        "total_stored_gb": round(total_stored, 1),
        "monthly_storage_cost": round(monthly_storage, 2),
        "total_monthly_cost": round(monthly_ingestion + monthly_storage, 2),
    }


# =============================================================================
# 6. DEMO
# =============================================================================

if __name__ == "__main__":
    random.seed(42)

    # --- Multi-format Parsing ---
    print("=" * 60)
    print("Multi-Format Log Parser")
    print("=" * 60)
    sample_logs = [
        '{"level":"error","message":"connection refused","service":"order-svc","host":"pod-1"}',
        '192.168.1.1 - - [10/Oct/2024:13:55:36 +0000] "GET /api/orders HTTP/1.1" 200 1234',
        '2024-10-10 13:55:36,789 WARNING auth.handler Token expired for user=42',
        'Oct 10 13:55:36 web-01 nginx[1234]: upstream timeout',
    ]
    for line in sample_logs:
        record = parse_log(line)
        if record:
            print(f"  [{record.source or 'json':>10}] {record.level:>7} | {record.message[:50]}")

    # --- Log Pipeline ---
    print(f"\n{'=' * 60}")
    print("Log Processing Pipeline")
    print("=" * 60)
    pipeline = LogPipeline(
        name="production",
        filters=[
            lambda r: r.level != "DEBUG",  # Drop debug logs
            lambda r: "/healthz" not in r.message,  # Drop health checks
        ],
        transforms=[
            lambda r: LogRecord(  # Add environment tag
                **{**r.__dict__, "fields": {**r.fields, "env": "production"}}
            ),
        ],
        routes={
            "errors": lambda r: r.level in ("ERROR", "FATAL", "CRITICAL"),
            "access": lambda r: r.source == "nginx",
            "application": lambda r: True,
        },
        sample_rate=0.8,  # Keep 80% of logs
    )

    # Generate synthetic logs
    log_lines = []
    levels = ["DEBUG", "INFO", "INFO", "INFO", "WARNING", "ERROR"]
    for i in range(500):
        if random.random() < 0.3:
            log_lines.append(
                f'192.168.1.{random.randint(1,10)} - - '
                f'[10/Oct/2024:13:55:{i%60:02d} +0000] '
                f'"GET {random.choice(["/api/orders", "/healthz", "/api/users"])} '
                f'HTTP/1.1" {random.choice([200, 200, 200, 404, 500])} 1234'
            )
        else:
            level = random.choice(levels)
            log_lines.append(json.dumps({
                "level": level.lower(), "message": f"Operation {i} completed",
                "service": random.choice(["order-svc", "payment-svc", "user-svc"]),
            }))

    output = pipeline.process(log_lines)
    stats = pipeline.stats
    print(f"  Received: {stats.received}")
    print(f"  Parsed: {stats.parsed}")
    print(f"  Filtered out: {stats.filtered_out}")
    print(f"  Sampled out: {stats.sampled_out}")
    print(f"  Routed: {stats.routed}")
    print(f"  Bytes in: {stats.bytes_in:,}, Bytes out: {stats.bytes_out:,} "
          f"(reduction: {(1 - stats.bytes_out/max(stats.bytes_in,1))*100:.0f}%)")

    # --- Log-Based Metrics ---
    print(f"\n{'=' * 60}")
    print("Log-Based Metric Extraction")
    print("=" * 60)
    extractor = LogMetricExtractor()
    all_records = [r for recs in output.values() for r in recs]
    error_ratio = extractor.extract_error_ratio(all_records)
    by_level = extractor.count_by(all_records, "level")
    print(f"  Error ratio: {error_ratio:.2%}")
    print(f"  By level: {by_level}")

    # --- Cost Estimation ---
    print(f"\n{'=' * 60}")
    print("Logging Cost Estimation")
    print("=" * 60)
    for daily_gb in [10, 50, 200]:
        cost = estimate_logging_costs(daily_gb)
        print(f"  {daily_gb:>4} GB/day: ${cost['total_monthly_cost']:>8.2f}/month "
              f"(ingestion=${cost['monthly_ingestion_cost']}, "
              f"storage=${cost['monthly_storage_cost']})")
    # Show savings from pipeline filtering
    raw_gb = 50
    after_pipeline = raw_gb * (stats.bytes_out / max(stats.bytes_in, 1))
    before = estimate_logging_costs(raw_gb)
    after = estimate_logging_costs(after_pipeline)
    print(f"\n  Pipeline savings ({raw_gb}GB -> {after_pipeline:.1f}GB/day):")
    print(f"    Before: ${before['total_monthly_cost']}/month")
    print(f"    After:  ${after['total_monthly_cost']}/month")
    print(f"    Saved:  ${before['total_monthly_cost'] - after['total_monthly_cost']:.2f}/month")

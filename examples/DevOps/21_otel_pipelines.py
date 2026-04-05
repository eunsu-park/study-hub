#!/usr/bin/env python3
"""Example: OpenTelemetry Pipelines — Collector Config, Processing & Export

Demonstrates OpenTelemetry Collector pipeline architecture: receivers,
processors (batching, filtering, tail sampling), and exporters, plus
programmatic pipeline configuration and validation.
Related lesson: 23_OpenTelemetry_Pipelines.md
"""

# =============================================================================
# WHY OTEL PIPELINES?
# The OpenTelemetry Collector is the universal telemetry router. It receives
# data from any source, processes it (filter, batch, sample, transform),
# and exports to any backend. Understanding pipeline architecture lets you
# build cost-effective, vendor-neutral observability infrastructure.
# =============================================================================

import json
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


# =============================================================================
# 1. PIPELINE COMPONENT MODELS
# =============================================================================

class ComponentType(Enum):
    RECEIVER = "receiver"
    PROCESSOR = "processor"
    EXPORTER = "exporter"


@dataclass
class PipelineComponent:
    """A component in the OTel Collector pipeline."""
    name: str
    component_type: ComponentType
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class TelemetryPipeline:
    """An OTel Collector pipeline (traces, metrics, or logs)."""
    name: str
    signal_type: str  # traces, metrics, logs
    receivers: list[str] = field(default_factory=list)
    processors: list[str] = field(default_factory=list)
    exporters: list[str] = field(default_factory=list)


@dataclass
class CollectorConfig:
    """Full OTel Collector configuration."""
    components: dict[str, PipelineComponent] = field(default_factory=dict)
    pipelines: dict[str, TelemetryPipeline] = field(default_factory=dict)

    def add_component(self, comp: PipelineComponent) -> None:
        key = f"{comp.component_type.value}/{comp.name}"
        self.components[key] = comp

    def add_pipeline(self, pipeline: TelemetryPipeline) -> None:
        self.pipelines[pipeline.name] = pipeline

    def validate(self) -> list[str]:
        """Validate that all pipeline references resolve to components."""
        errors: list[str] = []
        for pname, pipe in self.pipelines.items():
            for recv in pipe.receivers:
                key = f"receiver/{recv}"
                if key not in self.components:
                    errors.append(f"Pipeline '{pname}': unknown receiver '{recv}'")
            for proc in pipe.processors:
                key = f"processor/{proc}"
                if key not in self.components:
                    errors.append(f"Pipeline '{pname}': unknown processor '{proc}'")
            for exp in pipe.exporters:
                key = f"exporter/{exp}"
                if key not in self.components:
                    errors.append(f"Pipeline '{pname}': unknown exporter '{exp}'")
        return errors

    def to_yaml_dict(self) -> dict:
        """Convert to OTel Collector YAML-compatible dict."""
        receivers = {}
        processors = {}
        exporters = {}
        for key, comp in self.components.items():
            if comp.component_type == ComponentType.RECEIVER:
                receivers[comp.name] = comp.config
            elif comp.component_type == ComponentType.PROCESSOR:
                processors[comp.name] = comp.config
            elif comp.component_type == ComponentType.EXPORTER:
                exporters[comp.name] = comp.config
        pipelines = {}
        for name, pipe in self.pipelines.items():
            pipelines[f"{pipe.signal_type}/{name}"] = {
                "receivers": pipe.receivers,
                "processors": pipe.processors,
                "exporters": pipe.exporters,
            }
        return {
            "receivers": receivers,
            "processors": processors,
            "exporters": exporters,
            "service": {"pipelines": pipelines},
        }


# =============================================================================
# 2. PROCESSOR SIMULATIONS
# =============================================================================

@dataclass
class TelemetryRecord:
    """A generic telemetry record flowing through the pipeline."""
    signal_type: str
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    value: Any = None


def batch_processor(records: list[TelemetryRecord],
                    batch_size: int = 100,
                    timeout_ms: int = 200) -> list[list[TelemetryRecord]]:
    """Simulate the batch processor: group records into batches."""
    batches = []
    for i in range(0, len(records), batch_size):
        batches.append(records[i:i + batch_size])
    return batches


def filter_processor(records: list[TelemetryRecord],
                     include: dict[str, Any] | None = None,
                     exclude: dict[str, Any] | None = None) -> list[TelemetryRecord]:
    """Simulate the filter processor: include/exclude by attributes."""
    result = records
    if include:
        result = [r for r in result
                  if any(r.attributes.get(k) == v for k, v in include.items())]
    if exclude:
        result = [r for r in result
                  if not any(r.attributes.get(k) == v for k, v in exclude.items())]
    return result


def tail_sampling_processor(records: list[TelemetryRecord],
                            error_sample_rate: float = 1.0,
                            slow_threshold_ms: float = 500.0,
                            default_sample_rate: float = 0.1) -> list[TelemetryRecord]:
    """Simulate tail-based sampling: keep errors and slow traces, sample the rest."""
    sampled = []
    for r in records:
        is_error = r.attributes.get("status") == "ERROR"
        is_slow = (r.attributes.get("duration_ms", 0) > slow_threshold_ms)
        if is_error and random.random() < error_sample_rate:
            r.attributes["sampling_reason"] = "error"
            sampled.append(r)
        elif is_slow and random.random() < error_sample_rate:
            r.attributes["sampling_reason"] = "slow"
            sampled.append(r)
        elif random.random() < default_sample_rate:
            r.attributes["sampling_reason"] = "random"
            sampled.append(r)
    return sampled


def attributes_processor(records: list[TelemetryRecord],
                         insert: dict[str, str] | None = None,
                         delete: list[str] | None = None) -> list[TelemetryRecord]:
    """Simulate the attributes processor: add/remove attributes."""
    for r in records:
        if insert:
            r.attributes.update(insert)
        if delete:
            for key in delete:
                r.attributes.pop(key, None)
    return records


# =============================================================================
# 3. PIPELINE EXECUTOR
# =============================================================================

def execute_pipeline(records: list[TelemetryRecord],
                     processors: list[Callable]) -> list[TelemetryRecord]:
    """Execute a chain of processors on telemetry records."""
    current = records
    for proc in processors:
        current = proc(current)
    return current


# =============================================================================
# 4. COST ESTIMATOR
# =============================================================================

def estimate_pipeline_costs(
    ingestion_rate_per_sec: float,
    sampling_rate: float,
    cost_per_million_spans: float = 2.50,
) -> dict[str, Any]:
    """Estimate observability backend costs with and without sampling."""
    daily_raw = ingestion_rate_per_sec * 86400
    daily_sampled = daily_raw * sampling_rate
    monthly_raw = daily_raw * 30 / 1_000_000
    monthly_sampled = daily_sampled * 30 / 1_000_000
    return {
        "daily_raw_spans": int(daily_raw),
        "daily_sampled_spans": int(daily_sampled),
        "monthly_cost_without_sampling": round(monthly_raw * cost_per_million_spans, 2),
        "monthly_cost_with_sampling": round(monthly_sampled * cost_per_million_spans, 2),
        "savings_pct": round((1 - sampling_rate) * 100, 1),
    }


# =============================================================================
# 5. DEMO
# =============================================================================

if __name__ == "__main__":
    random.seed(42)

    # --- Build Collector Config ---
    print("=" * 60)
    print("OTel Collector Configuration")
    print("=" * 60)
    config = CollectorConfig()
    config.add_component(PipelineComponent(
        "otlp", ComponentType.RECEIVER,
        {"protocols": {"grpc": {"endpoint": "0.0.0.0:4317"},
                       "http": {"endpoint": "0.0.0.0:4318"}}},
    ))
    config.add_component(PipelineComponent(
        "batch", ComponentType.PROCESSOR,
        {"send_batch_size": 1024, "timeout": "200ms"},
    ))
    config.add_component(PipelineComponent(
        "tail_sampling", ComponentType.PROCESSOR,
        {"policies": [{"name": "errors", "type": "status_code", "status_code": {"status_codes": ["ERROR"]}}]},
    ))
    config.add_component(PipelineComponent(
        "otlphttp", ComponentType.EXPORTER,
        {"endpoint": "https://tempo.example.com:4318"},
    ))
    config.add_component(PipelineComponent(
        "prometheus", ComponentType.EXPORTER,
        {"endpoint": "0.0.0.0:8889"},
    ))
    config.add_pipeline(TelemetryPipeline(
        name="traces", signal_type="traces",
        receivers=["otlp"], processors=["tail_sampling", "batch"],
        exporters=["otlphttp"],
    ))
    config.add_pipeline(TelemetryPipeline(
        name="metrics", signal_type="metrics",
        receivers=["otlp"], processors=["batch"],
        exporters=["prometheus"],
    ))

    errors = config.validate()
    print(f"  Validation: {'PASS' if not errors else 'FAIL'}")
    for e in errors:
        print(f"    ERROR: {e}")
    yaml_dict = config.to_yaml_dict()
    print(f"  Pipelines: {list(yaml_dict['service']['pipelines'].keys())}")

    # --- Processor Simulation ---
    print(f"\n{'=' * 60}")
    print("Pipeline Processing Simulation")
    print("=" * 60)
    # Generate synthetic spans
    records = []
    for i in range(1000):
        records.append(TelemetryRecord(
            signal_type="trace", name=f"span-{i}",
            attributes={
                "service": random.choice(["api", "order", "payment"]),
                "status": "ERROR" if random.random() < 0.05 else "OK",
                "duration_ms": random.expovariate(1/100),
                "user_id": f"user-{random.randint(1, 100)}",
            },
        ))
    print(f"  Input records: {len(records)}")

    # Filter: only keep api and order services
    filtered = filter_processor(records, include={"service": "api"})
    print(f"  After filter (service=api): {len(filtered)}")

    # Tail sampling on all records
    sampled = tail_sampling_processor(records, default_sample_rate=0.1)
    reasons = {}
    for r in sampled:
        reason = r.attributes.get("sampling_reason", "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1
    print(f"  After tail sampling: {len(sampled)} ({reasons})")

    # Batching
    batches = batch_processor(sampled, batch_size=50)
    print(f"  Batched into: {len(batches)} batches")

    # --- Cost Estimation ---
    print(f"\n{'=' * 60}")
    print("Pipeline Cost Estimation")
    print("=" * 60)
    for rate in [100, 1000, 10000]:
        cost = estimate_pipeline_costs(rate, sampling_rate=0.1)
        print(f"  {rate} spans/s: "
              f"${cost['monthly_cost_without_sampling']}/mo raw -> "
              f"${cost['monthly_cost_with_sampling']}/mo sampled "
              f"(saves {cost['savings_pct']}%)")

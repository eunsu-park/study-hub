# 28. Capstone: Full-Stack Observability

**Previous**: [AIOps and Anomaly Detection](./27_AIOps_Anomaly_Detection.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Design an end-to-end observability platform for a production microservices system
2. Select and integrate tools across the metrics, logs, traces, and profiling stack
3. Build a cost model for observability infrastructure and optimize spend
4. Create an observability maturity roadmap for an organization
5. Evaluate build-vs-buy decisions for observability tooling
6. Demonstrate mastery of all observability concepts from Lessons 19-27

---

This capstone lesson synthesizes everything from the observability track (Lessons 19-27) into a cohesive, production-ready observability platform design. You will design the architecture, select the tools, define the processes, and calculate the costs for a realistic microservices platform.

> **Analogy -- Building a Hospital**: Individual medical skills (cardiology, radiology, surgery) are necessary but insufficient to treat patients well. A hospital needs an integrated system: intake (data collection), triage (alerting), diagnostics (correlation and investigation), treatment (remediation), and quality improvement (postmortems). This capstone is about building the hospital, not just practicing medicine.

## 1. Reference Architecture

### 1.1 The Target System

```
E-commerce Platform:
- 30 microservices (Go, Python, Java, Node.js)
- 3 Kubernetes clusters (us-east, eu-west, ap-southeast)
- PostgreSQL, Redis, Elasticsearch, Kafka
- 10,000 requests/second peak
- 99.95% availability SLO
- Team: 50 engineers across 8 teams
```

### 1.2 Full-Stack Observability Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Visualization & Analysis                  │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │  Grafana   │  │  Grafana     │  │  PagerDuty/  │             │
│  │ Dashboards │  │  Alerting    │  │  Opsgenie    │             │
│  │  + SLOs    │  │  (Unified)   │  │  (Paging)    │             │
│  └─────┬─────┘  └──────┬───────┘  └──────────────┘             │
│        │               │                                         │
│  ┌─────▼───────────────▼─────────────────────────────────────┐  │
│  │              Data Sources (Grafana connects to all)         │  │
│  ├───────────┬──────────────┬─────────────┬──────────────────┤  │
│  │Prometheus │   Tempo      │    Loki     │   Pyroscope      │  │
│  │ + Mimir   │  (Traces)    │   (Logs)    │  (Profiles)      │  │
│  │(Metrics)  │              │             │                   │  │
│  └─────┬─────┴──────┬───────┴──────┬──────┴─────────┬────────┘  │
│        │            │              │                │            │
│  ┌─────▼────────────▼──────────────▼────────────────▼────────┐  │
│  │                OTel Collector Gateway                       │  │
│  │  - Tail sampling    - Span metrics connector               │  │
│  │  - PII scrubbing    - Service graph connector              │  │
│  │  - Routing          - Attribute enrichment                 │  │
│  └──────────────────────────┬────────────────────────────────┘  │
│                             │                                    │
│  ┌──────────────────────────▼────────────────────────────────┐  │
│  │              OTel Collector Agents (DaemonSet)              │  │
│  │  - Batch + memory limit   - Health check filtering         │  │
│  │  - Node metadata enrichment                                │  │
│  └──────────────────────────┬────────────────────────────────┘  │
└─────────────────────────────┼────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────┐
│                    Application Layer                              │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │
│  │ Go svc │ │ Py svc │ │Java svc│ │Node svc│ │ ...×30 │       │
│  │+ OTel  │ │+ OTel  │ │+ OTel  │ │+ OTel  │ │        │       │
│  │ SDK    │ │ SDK    │ │ Agent  │ │ SDK    │ │        │       │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘       │
│  ┌─────────────────────────────────────────────────────┐       │
│  │ Cilium + Hubble (eBPF network observability)        │       │
│  └─────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Tool Selection

### 2.1 Selection Criteria

| Criterion | Weight | Factors |
|-----------|--------|---------|
| **Cost** | 25% | Licensing, infrastructure, operational overhead |
| **Interoperability** | 20% | Open standards (OTel, PromQL), data source linking |
| **Scalability** | 20% | Handles 3 clusters, 30 services, 10K rps |
| **Operational complexity** | 15% | Team expertise, maintenance burden |
| **Feature completeness** | 10% | Covers all telemetry types, correlates signals |
| **Vendor independence** | 10% | Avoid lock-in, use open protocols |

### 2.2 Tool Selection Matrix

| Signal | Tool | Justification |
|--------|------|---------------|
| **Metrics** | Prometheus + Mimir | Industry standard, PromQL ecosystem, Mimir for long-term and multi-tenant |
| **Traces** | Grafana Tempo | Object storage backend (cheap), native Grafana integration |
| **Logs** | Grafana Loki | Label-based indexing (low cost), LogQL, Grafana integration |
| **Profiles** | Grafana Pyroscope | Flame graphs, trace-to-profile linking |
| **Collection** | OTel Collector | Vendor-neutral, supports all signals, extensible |
| **Network** | Cilium Hubble | eBPF-based, zero-instrumentation network observability |
| **Visualization** | Grafana | Unified UI for all signals, cross-signal linking |
| **Alerting** | Grafana Unified Alerting | Single alerting engine across all data sources |
| **Paging** | PagerDuty | On-call scheduling, escalation, incident management |

### 2.3 Build vs Buy Decision

| Factor | Self-Hosted (OSS) | SaaS (Grafana Cloud, Datadog) |
|--------|-------------------|-------------------------------|
| **Cost at scale** | Lower (infrastructure + engineering time) | Higher (per-metric, per-host, per-trace pricing) |
| **Operational burden** | High (upgrades, scaling, HA) | Low (managed by vendor) |
| **Customization** | Full control | Limited to vendor features |
| **Data residency** | Full control | Depends on vendor regions |
| **Time to value** | Weeks to months | Days to weeks |
| **Team expertise** | Requires observability platform team | Minimal (vendor supports) |

**Recommendation**: Start with SaaS for small teams (< 20 engineers). Move to self-hosted when you have a dedicated platform team AND cost savings justify the operational overhead.

---

## 3. SLO Framework

### 3.1 Service SLO Definitions

```yaml
# SLO definitions for the e-commerce platform
services:
  - name: api-gateway
    slos:
      - name: availability
        target: 99.99%
        sli: "Proportion of non-5xx responses"
        window: 30d rolling
      - name: latency
        target: 99%
        threshold: 200ms
        sli: "Proportion of requests < 200ms"

  - name: payment-service
    slos:
      - name: availability
        target: 99.95%
        sli: "Proportion of successful payment attempts"
        window: 30d rolling
      - name: latency
        target: 99%
        threshold: 500ms

  - name: search-service
    slos:
      - name: availability
        target: 99.9%
        sli: "Proportion of non-error search responses"
      - name: latency
        target: 95%
        threshold: 300ms
      - name: freshness
        target: 99%
        threshold: 60s
        sli: "Index updated within 60s of source change"

  - name: order-service
    slos:
      - name: availability
        target: 99.95%
      - name: latency
        target: 99%
        threshold: 1000ms

# User journey SLOs
journeys:
  - name: checkout
    target: 99.9%
    sli: "Proportion of checkout attempts completing successfully within 5s"
    services: [api-gateway, cart-service, payment-service, order-service, inventory-service]

  - name: search-and-browse
    target: 99.5%
    sli: "Proportion of searches returning results within 500ms"
    services: [api-gateway, search-service, recommendation-service]
```

### 3.2 Error Budget Policy

```
Error Budget Policy (organization-wide)
────────────────────────────────────────
Budget > 50%:
  → Ship features at normal velocity
  → Standard deployment practices

Budget 25-50%:
  → Mandatory canary deployments
  → No risky deployments on Fridays
  → Weekly SLO review

Budget 5-25%:
  → Feature freeze for non-critical changes
  → Reliability sprint: next sprint dedicated to reliability
  → Daily SLO review

Budget < 5%:
  → Full deployment freeze (except critical security patches)
  → Incident review and postmortem for budget-consuming events
  → Engineering leadership notified

Budget exhausted:
  → SLO violation declared
  → All teams contributing to this SLO enter reliability mode
  → Postmortem required within 48 hours
  → Action items tracked to completion
```

---

## 4. Instrumentation Strategy

### 4.1 Per-Language Instrumentation Plan

| Language | Auto-Instrumentation | Manual Instrumentation | SDK |
|----------|---------------------|----------------------|-----|
| **Go** | OTel contrib libraries | Custom spans for business logic | `go.opentelemetry.io/otel` |
| **Python** | `opentelemetry-instrument` CLI | Custom spans for business logic | `opentelemetry-sdk` |
| **Java** | `-javaagent:otel-javaagent.jar` | `@WithSpan` annotations | OTel Java Agent |
| **Node.js** | `@opentelemetry/auto-instrumentations-node` | Custom spans | `@opentelemetry/sdk-node` |

### 4.2 Mandatory Telemetry Standards

```yaml
# Organization-wide telemetry standards
resource_attributes:
  required:
    - service.name              # Matches Kubernetes service name
    - service.version           # Semantic version from deployment
    - deployment.environment    # production, staging, development
    - k8s.namespace.name        # Kubernetes namespace
    - k8s.pod.name              # Pod name
    - k8s.node.name             # Node name

span_conventions:
  required:
    - http.method               # GET, POST, etc.
    - http.route                # Template, not instance (/users/:id)
    - http.status_code          # Response status code
  recommended:
    - db.system                 # postgresql, redis, etc.
    - db.name                   # Database name
    - messaging.system          # kafka, rabbitmq, etc.

log_standards:
  format: JSON
  required_fields:
    - timestamp (ISO 8601)
    - level (INFO, WARN, ERROR)
    - message
    - service_name
    - trace_id
    - span_id
  forbidden:
    - Passwords, API keys, tokens
    - Full email addresses (use domain only)
    - Credit card numbers
    - Social security numbers

metric_conventions:
  naming: snake_case with unit suffix (_total, _seconds, _bytes)
  labels:
    max_cardinality: 1000 per metric
    forbidden: user_id, request_id, email, ip_address
```

---

## 5. Pipeline Configuration

### 5.1 OTel Collector Gateway (Production)

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: 2048
    spike_limit_mib: 512

  # PII scrubbing
  transform/pii:
    error_mode: ignore
    trace_statements:
      - context: span
        statements:
          - replace_pattern(attributes["db.statement"], "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}", "***@***.***")
          - replace_pattern(attributes["db.statement"], "\\b\\d{4}[- ]?\\d{4}[- ]?\\d{4}[- ]?\\d{4}\\b", "****")
    log_statements:
      - context: log
        statements:
          - replace_pattern(body, "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}", "***@***.***")

  # Tail sampling
  tail_sampling:
    decision_wait: 30s
    num_traces: 300000
    policies:
      - name: errors
        type: status_code
        status_code: {status_codes: [ERROR]}
      - name: slow
        type: latency
        latency: {threshold_ms: 2000}
      - name: critical-services
        type: string_attribute
        string_attribute:
          key: service.name
          values: [payment-service, auth-service, order-service]
      - name: sample-rest
        type: probabilistic
        probabilistic: {sampling_percentage: 10}

  # Attribute enrichment
  attributes/enrich:
    actions:
      - key: platform.team
        from_attribute: service.name
        action: insert
      - key: cost_center
        value: "engineering"
        action: insert

  batch:
    send_batch_size: 2048
    timeout: 10s

connectors:
  spanmetrics:
    histogram:
      explicit:
        buckets: [5ms, 10ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s]
    dimensions:
      - name: http.method
      - name: http.route
      - name: http.status_code
    exemplars:
      enabled: true
  servicegraph:
    latency_histogram_buckets: [10ms, 50ms, 100ms, 500ms, 1s, 5s]

exporters:
  otlp/tempo:
    endpoint: tempo-distributor:4317
  prometheus:
    endpoint: 0.0.0.0:8889
  loki:
    endpoint: http://loki-gateway:3100/loki/api/v1/push
  otlp/pyroscope:
    endpoint: pyroscope:4317

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, transform/pii, attributes/enrich, tail_sampling, batch]
      exporters: [otlp/tempo, spanmetrics, servicegraph]
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [prometheus]
    metrics/derived:
      receivers: [spanmetrics, servicegraph]
      exporters: [prometheus]
    logs:
      receivers: [otlp]
      processors: [memory_limiter, transform/pii, batch]
      exporters: [loki]
```

---

## 6. Cost Management

### 6.1 Cost Model

```
Component          Monthly Cost Estimate
─────────────────────────────────────────
Metrics (Mimir):
  2M active series × $0.008/1K series = $16,000
  Storage: 500 GB × $0.10/GB             =    $50
                                          --------
                                          $16,050

Traces (Tempo):
  10K spans/sec × 2,592,000 sec/mo = 25.9B spans
  After sampling (10%): 2.59B spans
  Storage: ~200 GB × $0.10/GB            =    $20
  (Tempo is very cheap due to object storage)
                                          --------
                                              $20

Logs (Loki):
  30 services × 5 GB/day × 30 days = 4.5 TB
  After filtering: ~1 TB/mo
  Storage: 1 TB × $0.10/GB              =   $100
  Ingestion: 1 TB × $0.50/GB            =   $500
                                          --------
                                             $600

Profiles (Pyroscope):
  30 services × 100 Hz × 30 days        = minimal storage
                                          --------
                                              $50

Infrastructure (K8s):
  OTel Collectors: 20 pods × 0.5 CPU    = 10 CPU
  Mimir: 12 pods × 2 CPU               = 24 CPU
  Tempo: 6 pods × 1 CPU                = 6 CPU
  Loki: 8 pods × 1 CPU                 = 8 CPU
  Grafana: 3 pods × 0.5 CPU            = 1.5 CPU
  Total: ~50 CPU × $0.04/hr × 720 hr   = $1,440
  Memory: 200 GB × $0.005/hr × 720     = $720
                                          --------
                                          $2,160

Total Monthly Cost:                     ~$18,880
Per service: $18,880 / 30               = ~$630/service/month
```

### 6.2 Cost Optimization Levers

| Lever | Savings | Implementation |
|-------|---------|----------------|
| Drop unused metrics | 20-30% of metric cost | Run `mimirtool analyze` quarterly |
| Reduce log verbosity | 40-60% of log cost | Filter DEBUG/TRACE at Collector |
| Increase trace sampling | 50-90% of trace cost | Tail sampling at 5% instead of 10% |
| Shorten retention | 10-20% overall | 7-day raw, 30-day aggregate, 90-day metrics only |
| Use recording rules | 10-15% of metric cost | Pre-aggregate high-cardinality queries |
| Right-size Collector resources | 10-20% of infra cost | Profile Collector CPU/memory usage |

---

## 7. Operational Processes

### 7.1 Observability Team Responsibilities

```
Platform/Observability Team (2-3 engineers):
  - Maintain Collector pipeline configuration
  - Operate metrics/logs/traces backends
  - Create and maintain SLO dashboards
  - Define telemetry standards
  - Review and optimize costs quarterly
  - Train product teams on instrumentation

Product Teams (each team):
  - Instrument their services (following standards)
  - Define SLOs for their services
  - Create service-specific dashboards
  - Write runbooks for their alerts
  - Participate in on-call rotation
  - Conduct postmortems for their incidents
```

### 7.2 Quarterly Observability Review

```
Quarterly Review Agenda (90 minutes):
  1. (15 min) Cost review: actual vs budget, optimization opportunities
  2. (15 min) SLO compliance: which services met/missed SLOs
  3. (15 min) Incident analysis: MTTD/MTTR trends, postmortem action item completion
  4. (15 min) Tool evaluation: any gaps, new tools to evaluate
  5. (15 min) Telemetry quality: cardinality trends, missing instrumentation
  6. (15 min) Roadmap: next quarter priorities
```

---

## 8. Maturity Roadmap

### 8.1 12-Month Roadmap

```
Quarter 1: Foundation
────────────────────
Week 1-2:   Deploy OTel Collector (agent + gateway)
Week 3-4:   Deploy Mimir, Tempo, Loki, Grafana
Week 5-6:   Auto-instrument top 10 critical services
Week 7-8:   Create core SLO dashboards (availability, latency)
Week 9-10:  Configure burn rate alerts for critical services
Week 11-12: Train teams on basic instrumentation and debugging
Milestone:  Level 1 (Informed) achieved

Quarter 2: Correlation
──────────────────────
Week 1-3:   Enable trace-to-log linking (trace_id in all logs)
Week 4-6:   Enable exemplars on key metrics
Week 7-8:   Deploy spanmetrics and servicegraph connectors
Week 9-10:  Instrument remaining 20 services
Week 11-12: Create unified debugging dashboards
Milestone:  Level 2 (Investigative) achieved

Quarter 3: Proactive
────────────────────
Week 1-3:   Define SLOs for all 30 services
Week 4-6:   Deploy Pyroscope for continuous profiling
Week 7-8:   Implement dynamic alerting (statistical baselines)
Week 9-10:  Build change-impact correlation (deploys → anomalies)
Week 11-12: Run first game day exercise
Milestone:  Level 3 (Proactive) achieved

Quarter 4: Optimization
───────────────────────
Week 1-3:   Cost optimization sprint (remove unused telemetry)
Week 4-6:   Implement L2 auto-remediation for known patterns
Week 7-8:   Deploy Cilium Hubble for network observability
Week 9-10:  Evaluate AIOps capabilities (anomaly detection, RCA)
Week 11-12: Publish observability maturity report
Milestone:  Level 3+ (Proactive, optimized) achieved
```

---

## 9. Summary of the Observability Track

| Lesson | Key Takeaway |
|--------|-------------|
| 19. Observability Engineering | Observability is about asking arbitrary questions, not just monitoring dashboards |
| 20. SLO Engineering | SLOs + error budgets provide a decision framework for reliability vs velocity |
| 21. Signal Correlation | Linking metrics, logs, and traces enables 5x faster debugging |
| 22. Advanced Metrics Architecture | Thanos/Mimir solve Prometheus scaling; cardinality is the key cost driver |
| 23. OpenTelemetry Pipelines | The Collector pipeline design determines observability quality and cost |
| 24. eBPF Observability | Kernel-level observability without code changes complements OTel |
| 25. Continuous Profiling | Profiling tells you WHY at the code level; flame graphs are the primary tool |
| 26. Incident Response | Structured response + blameless postmortems turn incidents into learning |
| 27. AIOps and Anomaly Detection | ML enhances alerting but SLO-based alerting is the foundation |
| 28. Capstone (this lesson) | Integration of all concepts into a production-ready platform |

---

## Exercises

### Exercise 1: Platform Design

Design a complete observability platform for a startup with:
- 10 microservices (all Python/FastAPI)
- Single Kubernetes cluster
- 1,000 requests/second peak
- Team of 15 engineers (no dedicated platform team)
- Budget: $3,000/month for observability

Specify: tool selection, architecture diagram, SLO definitions for 3 critical services, OTel Collector configuration, and cost breakdown.

<details>
<summary>Show Answer</summary>

**Tool selection**: Given the small team and limited budget, use **Grafana Cloud Free/Pro tier** for managed observability. This eliminates the need for a platform team.

| Signal | Tool | Justification |
|--------|------|---------------|
| Metrics | Grafana Cloud (managed Mimir) | Free tier: 10K series. Pro: $8/1K series |
| Traces | Grafana Cloud (managed Tempo) | 50 GB free. Pro: $0.50/GB |
| Logs | Grafana Cloud (managed Loki) | 50 GB free. Pro: $0.50/GB |
| Collection | OTel Collector | Self-managed (simple DaemonSet) |
| Alerting | Grafana Cloud Alerting | Included |
| Paging | PagerDuty (free tier for < 5 users) | 2-3 on-call rotations sufficient |

**Architecture:**
```
10 Python/FastAPI services (auto-instrumented via opentelemetry-instrument)
  → OTel Collector DaemonSet (1 per node, 3 nodes)
    → Grafana Cloud (OTLP endpoint)
      → Grafana dashboards + alerting + PagerDuty
```

**Cost breakdown:**
```
Grafana Cloud Pro:
  Metrics: ~50K series × $8/1K = $400
  Traces: ~20 GB/mo (after 10% sampling) × $0.50 = $10
  Logs: ~100 GB/mo × $0.50 = $50
  Profiles: 10 services × $5 = $50
                                    Subtotal: $510

Infrastructure:
  OTel Collectors: 3 pods × 0.25 CPU = $25
  (Backends are managed by Grafana Cloud)
                                    Subtotal: $25

PagerDuty (free tier):                        $0

Total: ~$535/month (well under $3,000 budget)
```

**SLO definitions:**
```yaml
- service: payment-service
  slos:
    - name: availability
      target: 99.9%
      sli: "Non-5xx responses / total responses"
    - name: latency
      target: 99%
      threshold: 500ms

- service: auth-service
  slos:
    - name: availability
      target: 99.99%
    - name: latency
      target: 99%
      threshold: 200ms

- service: order-service
  slos:
    - name: availability
      target: 99.9%
    - name: latency
      target: 95%
      threshold: 1000ms
```

</details>

### Exercise 2: Incident Simulation

Walk through a complete incident response for the following scenario:

At 03:00 UTC on a Monday, the checkout journey SLO drops below target. The error rate for `order-service` spikes to 15%. Simultaneously, `inventory-service` latency increases from 50ms to 5 seconds. Kafka consumer lag for `inventory-events` is growing rapidly.

Describe: (a) detection and alerting, (b) triage and role assignment, (c) investigation using the correlated observability stack, (d) mitigation, (e) root cause identification, and (f) postmortem action items.

<details>
<summary>Show Answer</summary>

**(a) Detection and alerting:**
- 03:00 -- Checkout journey SLO burn rate alert fires (14.4x) -- pages on-call
- 03:00 -- OrderServiceHighErrorRate alert fires (15% > 1% threshold)
- 03:01 -- InventoryServiceHighLatency alert fires (p99 = 5s > 500ms threshold)
- 03:01 -- KafkaConsumerLagHigh alert fires (inventory-events consumer)
- Alertmanager groups: 4 alerts → 1 incident group (correlated by checkout journey)

**(b) Triage and role assignment:**
- 03:02 -- On-call engineer acknowledges. Declares SEV1 (checkout down at 15% error rate).
- 03:03 -- Creates incident channel `#inc-20250317-checkout-failure`
- 03:04 -- Assigns roles: IC = on-call, Tech Lead = inventory-service owner (paged), Scribe = on-call secondary

**(c) Investigation using observability stack:**

```
Step 1: SLO Dashboard (metrics)
  → Checkout journey at 85% (target 99.9%)
  → Order-service error budget: nearly exhausted

Step 2: Error rate breakdown (metrics)
  → order-service POST /orders: 15% 500 errors
  → All errors are "inventory reservation timeout"

Step 3: Click exemplar on error spike (metrics → traces)
  → Trace: api-gateway → order-service → inventory-service [TIMEOUT 5s]
  → inventory-service span: "reserve_items" took 5.0s then timed out

Step 4: View inventory-service metrics
  → Kafka consumer lag: 50,000 messages and growing
  → inventory-service CPU: 10% (not CPU-bound)
  → inventory-service db connections: 50/50 (EXHAUSTED!)

Step 5: View traces for inventory-service (Tempo)
  → All traces show: postgres SELECT taking 4-5 seconds
  → Attribute: db.statement = "SELECT * FROM inventory WHERE sku IN (...)"

Step 6: View logs for inventory-service (Loki)
  → "WARN: Lock wait timeout exceeded for inventory table"
  → "ERROR: cannot acquire lock on row in relation 'inventory'"
  → "INFO: Kafka consumer commit failed: rebalance in progress"

Step 7: Check PostgreSQL metrics
  → pg_stat_activity: 48 active queries, 40 in "Lock" wait state
  → One query running for 2 hours: ALTER TABLE inventory ADD COLUMN ...

Root cause identified: A schema migration (ALTER TABLE ADD COLUMN) is
holding a lock on the inventory table, blocking all SELECT queries.
The migration was triggered by a cron job at 03:00.
```

**(d) Mitigation:**
- 03:15 -- Kill the ALTER TABLE query: `SELECT pg_terminate_backend(pid)`
- 03:16 -- Inventory-service queries resume, consumer lag starts decreasing
- 03:20 -- Error rate drops to 0%, checkout SLO recovering
- 03:30 -- All metrics back to baseline, incident resolved

**(e) Root cause:**
A database migration cron job (scheduled at 03:00 UTC) ran `ALTER TABLE inventory ADD COLUMN last_restock_date TIMESTAMP` on a large table (10M rows). On PostgreSQL, `ALTER TABLE ADD COLUMN` with a default value on older versions acquires an `ACCESS EXCLUSIVE` lock, blocking all reads. The lock blocked all inventory queries, causing cascading timeouts in order-service.

**(f) Postmortem action items:**

| # | Action | Category | Priority |
|---|--------|----------|----------|
| 1 | Use `ALTER TABLE ADD COLUMN ... DEFAULT NULL` (no lock on PG 11+) or use concurrent migration tools | Prevention | P0 |
| 2 | Schedule schema migrations during maintenance windows, not via cron | Prevention | P1 |
| 3 | Add PostgreSQL lock wait monitoring: alert when any query waits > 30s | Detection | P0 |
| 4 | Add Kafka consumer lag alert with lower threshold (currently too high) | Detection | P1 |
| 5 | Implement query timeout at application level (5s max, currently infinite) | Mitigation | P1 |
| 6 | Add circuit breaker between order-service and inventory-service | Mitigation | P2 |
| 7 | Require migration review checklist (lock analysis) before any schema change | Prevention | P1 |

</details>

### Exercise 3: Cost Optimization

Your observability platform costs $25,000/month. Leadership asks you to reduce it to $15,000/month without significantly impacting debugging capability. Your current usage:

- Metrics: 3M active series ($24,000 on managed platform)
- Traces: 100 GB/month ($50)
- Logs: 2 TB/month ($1,000)
- Infrastructure: $2,000

Propose a cost optimization plan with specific actions and expected savings.

<details>
<summary>Show Answer</summary>

**Current cost: $27,050/month. Target: $15,000/month. Need to cut $12,050.**

The cost is overwhelmingly dominated by metrics ($24,000 = 89% of total). Focus there.

**Optimization plan:**

| # | Action | Signal | Expected Savings | New Cost |
|---|--------|--------|-----------------|----------|
| 1 | Run `mimirtool analyze` to find unused metrics. Expect ~30% are queried by zero dashboards or alerts. Drop them via metric_relabel_configs. | Metrics | $7,200 (30% of $24,000) | $16,800 |
| 2 | Reduce histogram bucket count from 11 to 7 (remove rarely-hit buckets). Histograms are the biggest cardinality multiplier. | Metrics | $2,400 (10%) | $14,400 |
| 3 | Use recording rules to pre-aggregate per-pod metrics to per-deployment. Drop per-pod metrics from long-term storage. | Metrics | $2,400 (10%) | $12,000 |
| 4 | Increase trace sampling from current rate to 5%. Keep 100% of errors and slow traces. | Traces | $25 (50% of $50) | $11,975 |
| 5 | Filter DEBUG and TRACE level logs at the OTel Collector. Typically 40-60% of log volume. | Logs | $500 (50% of $1,000) | $11,475 |
| 6 | Right-size Collector pods (profile actual CPU/memory, reduce requests). | Infra | $400 (20% of $2,000) | $11,075 |

**Total savings: $12,925**
**New monthly cost: ~$14,125** (within $15,000 target)

**Key insight**: Metrics cardinality is the #1 cost driver. The three metric-focused actions (1, 2, 3) account for $12,000 of the $12,925 savings. Traces and logs are already cheap in this architecture.

**Risk mitigation**: Before dropping any metrics, verify they are not used in any dashboard, alert, or recording rule. Use a 2-week "shadow" period where dropped metrics are still collected but not stored, allowing rollback if needed.

</details>

---

## References

- [Observability Engineering (O'Reilly)](https://www.oreilly.com/library/view/observability-engineering/9781492076438/)
- [Google SRE Book](https://sre.google/sre-book/table-of-contents/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Grafana LGTM Stack](https://grafana.com/oss/lgtm-stack/)
- [CNCF Observability Landscape](https://landscape.cncf.io/card-mode?category=observability-and-analysis)
- [Cloud Native Observability (O'Reilly)](https://www.oreilly.com/library/view/cloud-native-observability/9781098145545/)

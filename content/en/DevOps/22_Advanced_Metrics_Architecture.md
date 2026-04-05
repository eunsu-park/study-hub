# 22. Advanced Metrics Architecture

**Previous**: [Signal Correlation](./21_Signal_Correlation.md) | **Next**: [OpenTelemetry Pipelines](./23_OpenTelemetry_Pipelines.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Design a Prometheus federation architecture for multi-cluster and multi-region deployments
2. Compare and select long-term storage solutions (Thanos, Cortex, Mimir) based on operational requirements
3. Implement cardinality management strategies to prevent metric explosion
4. Write recording rules that optimize query performance and enable long-range analysis
5. Configure remote write and remote read for durable metrics storage
6. Apply metric relabeling to control ingestion volume and cost

---

A single Prometheus server works well for small deployments, but as your infrastructure grows beyond a few hundred services, you face fundamental challenges: single-node storage limits, cross-cluster querying, long-term retention, high availability, and cardinality explosion. This lesson covers the architectures and tools that scale Prometheus metrics to enterprise-grade deployments.

> **Analogy -- Library System**: A single library (Prometheus instance) works for a small town. But a university system with dozens of campus libraries needs a catalog system (federation/Thanos) that lets you search across all libraries from one desk, an archive (long-term storage) for books no longer on active shelves, and a deduplication system to handle the same book appearing in multiple branches.

## 1. Prometheus Scaling Challenges

### 1.1 Single-Instance Limits

| Challenge | Symptom | Threshold |
|-----------|---------|-----------|
| **Storage** | Disk fills up, old data dropped | ~2 weeks retention with 1M active series |
| **Memory** | OOM kills, slow queries | ~10M active time series |
| **Query performance** | Dashboard timeouts, slow alerts | Complex queries over >7 days |
| **Availability** | Single point of failure | Any Prometheus restart = gap |
| **Multi-cluster** | Cannot query across clusters | More than one K8s cluster |

### 1.2 Scaling Strategies Overview

```
                    ┌──────────────────┐
                    │   Requirements   │
                    └─────────┬────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
     ┌────────▼────────┐ ┌───▼────────┐ ┌────▼────────┐
     │ Multi-cluster   │ │ Long-term  │ │ High        │
     │ querying        │ │ storage    │ │ availability│
     └────────┬────────┘ └───┬────────┘ └────┬────────┘
              │               │               │
              └───────────────┼───────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
     ┌────────▼────────┐ ┌───▼────────┐ ┌────▼────────┐
     │ Federation      │ │ Thanos     │ │ Mimir       │
     │ (built-in)      │ │ Cortex     │ │ (Grafana)   │
     │                 │ │ Mimir      │ │             │
     │ Simple, limited │ │ Sidecar or │ │ Write-path  │
     │                 │ │ receive    │ │ scaling     │
     └─────────────────┘ └────────────┘ └─────────────┘
```

---

## 2. Prometheus Federation

### 2.1 Hierarchical Federation

Federation allows a global Prometheus to scrape aggregated metrics from lower-level Prometheus instances:

```
                    ┌──────────────────┐
                    │ Global Prometheus│
                    │ (cross-cluster   │
                    │  dashboards,     │
                    │  global alerts)  │
                    └───┬─────────┬────┘
                        │         │
              ┌─────────▼─┐   ┌──▼──────────┐
              │ Prom-US    │   │ Prom-EU     │
              │ (us-east   │   │ (eu-west    │
              │  cluster)  │   │  cluster)   │
              └─────┬──────┘   └──────┬──────┘
                    │                  │
              ┌─────▼──────┐   ┌──────▼──────┐
              │ Targets    │   │ Targets     │
              │ (pods,     │   │ (pods,      │
              │  nodes)    │   │  nodes)     │
              └────────────┘   └─────────────┘
```

### 2.2 Federation Configuration

```yaml
# Global Prometheus configuration
scrape_configs:
  - job_name: "federate-us-east"
    honor_labels: true
    metrics_path: "/federate"
    params:
      'match[]':
        # Only federate recording rules and key aggregates
        - '{__name__=~"job:.*"}'                    # Recording rules
        - '{__name__="up"}'                          # Target health
        - '{__name__=~".*:.*:rate5m"}'              # Pre-aggregated rates
        - '{__name__="kube_pod_status_phase"}'       # K8s metadata
    static_configs:
      - targets: ["prometheus-us-east.internal:9090"]
        labels:
          cluster: "us-east"
          region: "us"

  - job_name: "federate-eu-west"
    honor_labels: true
    metrics_path: "/federate"
    params:
      'match[]':
        - '{__name__=~"job:.*"}'
        - '{__name__="up"}'
        - '{__name__=~".*:.*:rate5m"}'
    static_configs:
      - targets: ["prometheus-eu-west.internal:9090"]
        labels:
          cluster: "eu-west"
          region: "eu"
```

### 2.3 Federation Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| **Pull-based** | Global Prometheus must reach all instances | VPN/mesh networking |
| **Single point of failure** | Global Prometheus down = no cross-cluster view | Deploy HA pair |
| **Data duplication** | Same metrics stored in local + global | Federate only recording rules |
| **Query limitations** | Cannot join raw metrics across clusters | Use Thanos/Mimir instead |
| **Scalability ceiling** | Single global Prometheus has memory limits | Shard by functional area |

---

## 3. Thanos

### 3.1 Architecture Overview

Thanos extends Prometheus with long-term storage, global querying, and high availability without modifying Prometheus itself:

```
┌──────────────────────────────────────────────────────┐
│                    Thanos Query                       │
│            (global PromQL query endpoint)             │
│    Deduplicates data from multiple Prometheus         │
└───────┬──────────────┬──────────────┬────────────────┘
        │              │              │
┌───────▼──────┐ ┌─────▼──────┐ ┌────▼─────────────┐
│ Thanos       │ │ Thanos     │ │ Thanos Store     │
│ Sidecar      │ │ Sidecar    │ │ Gateway          │
│ (Prom US)    │ │ (Prom EU)  │ │ (Object Storage) │
│              │ │            │ │                   │
│ Uploads TSDB │ │ Uploads    │ │ Reads historical  │
│ blocks to    │ │ blocks     │ │ data from S3/GCS  │
│ object store │ │            │ │                   │
└──────┬───────┘ └─────┬──────┘ └────────┬──────────┘
       │               │                  │
       └───────┬───────┘                  │
               ▼                          ▼
       ┌───────────────┐         ┌────────────────┐
       │ Object Storage│         │ Object Storage │
       │ (S3/GCS/Azure)│ ◄──────│ (same bucket)  │
       └───────────────┘         └────────────────┘
```

### 3.2 Thanos Components

| Component | Role | Deployment |
|-----------|------|------------|
| **Sidecar** | Uploads Prometheus TSDB blocks to object storage; proxies queries to Prometheus | Sidecar container in Prometheus pod |
| **Store Gateway** | Serves historical data from object storage | Stateless deployment |
| **Query** | Global PromQL endpoint; deduplicates and merges results | Stateless deployment |
| **Compactor** | Downsamples and compacts blocks in object storage | Single instance (or sharded) |
| **Ruler** | Evaluates recording and alerting rules across all data | Stateful deployment |
| **Receive** | Accepts remote_write from Prometheus (alternative to sidecar) | Stateful deployment |

### 3.3 Thanos Sidecar Configuration

```yaml
# Prometheus deployment with Thanos sidecar
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: prometheus
spec:
  template:
    spec:
      containers:
        - name: prometheus
          image: prom/prometheus:v2.51.0
          args:
            - "--config.file=/etc/prometheus/prometheus.yml"
            - "--storage.tsdb.path=/prometheus"
            - "--storage.tsdb.retention.time=2h"      # Short retention (sidecar uploads)
            - "--storage.tsdb.min-block-duration=2h"   # Required for Thanos
            - "--storage.tsdb.max-block-duration=2h"   # Required for Thanos
            - "--web.enable-lifecycle"
          volumeMounts:
            - name: prometheus-data
              mountPath: /prometheus

        - name: thanos-sidecar
          image: quay.io/thanos/thanos:v0.34.1
          args:
            - "sidecar"
            - "--tsdb.path=/prometheus"
            - "--prometheus.url=http://localhost:9090"
            - "--objstore.config-file=/etc/thanos/objstore.yml"
          volumeMounts:
            - name: prometheus-data
              mountPath: /prometheus
            - name: thanos-objstore-config
              mountPath: /etc/thanos

      volumes:
        - name: thanos-objstore-config
          secret:
            secretName: thanos-objstore
```

```yaml
# thanos-objstore.yml (S3 backend)
type: S3
config:
  bucket: "thanos-metrics"
  endpoint: "s3.us-east-1.amazonaws.com"
  region: "us-east-1"
  access_key: "${AWS_ACCESS_KEY_ID}"
  secret_key: "${AWS_SECRET_ACCESS_KEY}"
```

### 3.4 Thanos Compactor and Downsampling

The Compactor reduces storage costs by downsampling old data:

| Resolution | Retention | Data Points Per Day | Use Case |
|-----------|-----------|--------------------:|----------|
| **Raw** (scrape interval) | 14 days | ~5,760 (15s interval) | Recent debugging |
| **5 minute** | 90 days | 288 | Medium-range analysis |
| **1 hour** | 1 year+ | 24 | Long-range trends |

```yaml
# Thanos Compactor deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: thanos-compactor
spec:
  replicas: 1   # MUST be single instance
  template:
    spec:
      containers:
        - name: compactor
          image: quay.io/thanos/thanos:v0.34.1
          args:
            - "compact"
            - "--data-dir=/var/thanos/compact"
            - "--objstore.config-file=/etc/thanos/objstore.yml"
            - "--retention.resolution-raw=14d"
            - "--retention.resolution-5m=90d"
            - "--retention.resolution-1h=365d"
            - "--compact.concurrency=4"
            - "--downsample.concurrency=4"
            - "--wait"
```

---

## 4. Grafana Mimir

### 4.1 Mimir vs Thanos

| Feature | Thanos | Mimir |
|---------|--------|-------|
| **Architecture** | Sidecar + object storage | Receive-path (remote_write) |
| **Prometheus changes** | Minimal (sidecar) | Remote_write config only |
| **Multi-tenancy** | Basic (external labels) | Native (X-Scope-OrgID header) |
| **Scaling model** | Per-component scaling | Microservice or monolithic mode |
| **Query performance** | Good (store gateway caching) | Excellent (query frontend + caching) |
| **Operational complexity** | Medium (many components) | Medium-High (more moving parts) |
| **Best for** | Extending existing Prometheus | Greenfield, multi-tenant |

### 4.2 Mimir Architecture

```
┌──────────────┐     ┌──────────────┐
│ Prometheus 1 │     │ Prometheus 2 │
│ remote_write │     │ remote_write │
└──────┬───────┘     └──────┬───────┘
       │                    │
       └──────────┬─────────┘
                  │
       ┌──────────▼───────────┐
       │   Mimir Distributor   │  (Receives, validates, shards)
       └──────────┬───────────┘
                  │
       ┌──────────▼───────────┐
       │    Mimir Ingester     │  (Writes to TSDB, replicates)
       │   (Stateful, 3 replicas)│
       └──────────┬───────────┘
                  │
       ┌──────────▼───────────┐
       │   Object Storage      │  (S3/GCS/Azure -- long-term)
       └──────────┬───────────┘
                  │
       ┌──────────▼───────────┐
       │  Store Gateway        │  (Reads historical blocks)
       └──────────┬───────────┘
                  │
       ┌──────────▼───────────┐
       │   Query Frontend      │  (Splitting, caching, scheduling)
       └──────────┬───────────┘
                  │
       ┌──────────▼───────────┐
       │      Querier          │  (Merges ingester + store gateway)
       └──────────────────────┘
```

### 4.3 Prometheus Remote Write to Mimir

```yaml
# prometheus.yml -- remote write to Mimir
remote_write:
  - url: "http://mimir-distributor:8080/api/v1/push"
    headers:
      X-Scope-OrgID: "team-payments"    # Multi-tenant identifier
    queue_config:
      capacity: 10000
      max_shards: 30
      max_samples_per_send: 5000
      batch_send_deadline: 5s
    write_relabel_configs:
      # Drop expensive metrics before sending
      - source_labels: [__name__]
        regex: "go_.*"
        action: drop
```

---

## 5. Cardinality Management

### 5.1 Understanding Cardinality

Cardinality = number of unique time series. Each unique combination of metric name + label values creates a new time series.

```
# 1 metric × 3 methods × 5 endpoints × 3 statuses = 45 time series
http_requests_total{method="GET", endpoint="/api/users", status="200"}
http_requests_total{method="GET", endpoint="/api/users", status="404"}
http_requests_total{method="POST", endpoint="/api/orders", status="201"}
... (45 total)

# Adding user_id with 100K users: 45 × 100,000 = 4,500,000 time series!
```

### 5.2 Cardinality Monitoring

```promql
# Total active time series
prometheus_tsdb_head_series

# Time series created per scrape
sum(scrape_series_added) by (job)

# Top 10 metrics by cardinality
topk(10, count by (__name__) ({__name__!=""}))

# Cardinality per label (find the explosion source)
count by (__name__) ({__name__=~"http_requests_total.*"})

# TSDB head chunks (memory pressure indicator)
prometheus_tsdb_head_chunks
```

### 5.3 Cardinality Reduction Techniques

**1. Metric relabeling (drop at ingestion):**

```yaml
# prometheus.yml -- drop high-cardinality labels
scrape_configs:
  - job_name: "webapp"
    metric_relabel_configs:
      # Drop metrics entirely
      - source_labels: [__name__]
        regex: "go_gc_.*|process_.*"
        action: drop

      # Remove a specific label (reduce cardinality)
      - regex: "instance"
        action: labeldrop

      # Replace high-cardinality endpoint with route template
      - source_labels: [endpoint]
        regex: "/api/users/[0-9]+"
        target_label: endpoint
        replacement: "/api/users/:id"

      # Limit number of series per scrape
    sample_limit: 10000
```

**2. Recording rules (pre-aggregate):**

```yaml
groups:
  - name: cardinality_reduction
    rules:
      # Pre-aggregate per-instance metrics to per-job
      - record: job:http_requests_total:rate5m
        expr: sum by (job, method, status) (rate(http_requests_total[5m]))

      # Pre-aggregate per-pod to per-deployment
      - record: deployment:cpu_usage:avg
        expr: avg by (deployment) (rate(container_cpu_usage_seconds_total[5m]))
```

**3. Histogram bucket optimization:**

```python
# BAD: Default buckets create too many series for high-cardinality metrics
# (.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10) = 11 buckets + sum + count = 13 series per label set

# GOOD: Tailored buckets for your service's latency distribution
from prometheus_client import Histogram

# For a fast API (most requests < 100ms)
api_latency = Histogram(
    "http_request_duration_seconds",
    "Request duration",
    labelnames=["method", "route"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],  # 7 buckets
)

# For a batch job (seconds to minutes)
batch_duration = Histogram(
    "batch_job_duration_seconds",
    "Batch job duration",
    labelnames=["job_type"],
    buckets=[1, 5, 15, 30, 60, 120, 300, 600],  # 8 buckets
)
```

### 5.4 Cardinality Limits

```yaml
# Prometheus global limits
global:
  scrape_interval: 15s
  # Per-scrape limit (drops entire scrape if exceeded)
  sample_limit: 50000
  # Per-target label limit
  label_limit: 30
  label_name_length_limit: 200
  label_value_length_limit: 1000

# Mimir per-tenant limits
limits:
  max_global_series_per_user: 5000000
  max_global_series_per_metric: 50000
  max_label_names_per_series: 30
  max_label_name_length: 1024
  max_label_value_length: 2048
```

---

## 6. Recording Rules

### 6.1 Why Recording Rules

Recording rules pre-compute frequently needed or expensive PromQL expressions:

| Without Recording Rules | With Recording Rules |
|------------------------|---------------------|
| Dashboard query computed on every refresh | Pre-computed every `evaluation_interval` |
| Slow over large time ranges | Fast regardless of range |
| High memory/CPU on Prometheus | Minimal query-time cost |
| Inconsistent between dashboards and alerts | Single source of truth |

### 6.2 Recording Rule Naming Convention

Follow the Prometheus naming convention:

```
level:metric:operations

Examples:
  job:http_requests_total:rate5m          # per-job request rate
  instance:node_cpu:ratio                  # per-instance CPU ratio
  cluster:http_request_duration:p99        # per-cluster p99 latency
```

### 6.3 Comprehensive Recording Rules

```yaml
groups:
  - name: service_sli_recording_rules
    interval: 30s
    rules:
      # --- Request Rate ---
      - record: job:http_requests_total:rate5m
        expr: sum by (job) (rate(http_requests_total[5m]))

      - record: job_method:http_requests_total:rate5m
        expr: sum by (job, method) (rate(http_requests_total[5m]))

      # --- Error Rate ---
      - record: job:http_errors:ratio_rate5m
        expr: |
          sum by (job) (rate(http_requests_total{status=~"5.."}[5m]))
          / sum by (job) (rate(http_requests_total[5m]))

      # --- Latency Percentiles ---
      - record: job:http_request_duration_seconds:p50
        expr: |
          histogram_quantile(0.50,
            sum by (job, le) (rate(http_request_duration_seconds_bucket[5m]))
          )

      - record: job:http_request_duration_seconds:p95
        expr: |
          histogram_quantile(0.95,
            sum by (job, le) (rate(http_request_duration_seconds_bucket[5m]))
          )

      - record: job:http_request_duration_seconds:p99
        expr: |
          histogram_quantile(0.99,
            sum by (job, le) (rate(http_request_duration_seconds_bucket[5m]))
          )

  - name: infrastructure_recording_rules
    interval: 60s
    rules:
      # --- CPU ---
      - record: instance:node_cpu_utilisation:ratio_rate5m
        expr: |
          1 - avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m]))

      - record: cluster:node_cpu_utilisation:avg
        expr: avg(instance:node_cpu_utilisation:ratio_rate5m)

      # --- Memory ---
      - record: instance:node_memory_utilisation:ratio
        expr: |
          1 - (
            node_memory_MemAvailable_bytes
            / node_memory_MemTotal_bytes
          )

      # --- Disk ---
      - record: instance:node_filesystem_avail:ratio
        expr: |
          node_filesystem_avail_bytes{mountpoint="/"}
          / node_filesystem_size_bytes{mountpoint="/"}
```

---

## 7. Remote Write and Remote Read

### 7.1 Remote Write Architecture

```yaml
# prometheus.yml -- remote write to multiple backends
remote_write:
  # Primary: Mimir for long-term storage
  - url: "http://mimir:8080/api/v1/push"
    queue_config:
      capacity: 10000
      max_shards: 30
      min_shards: 1
      max_samples_per_send: 5000
      batch_send_deadline: 5s
      min_backoff: 30ms
      max_backoff: 5s
    write_relabel_configs:
      # Only send important metrics to long-term storage
      - source_labels: [__name__]
        regex: "(http_requests_total|http_request_duration_seconds_bucket|up|kube_.*|node_.*)"
        action: keep

  # Secondary: Datadog for business dashboards
  - url: "https://api.datadoghq.com/api/v1/series"
    bearer_token: "${DD_API_KEY}"
    write_relabel_configs:
      - source_labels: [__name__]
        regex: "(business_.*|revenue_.*|orders_.*)"
        action: keep
```

### 7.2 Remote Read for Historical Queries

```yaml
# prometheus.yml -- remote read from Thanos/Mimir
remote_read:
  - url: "http://thanos-query:9090/api/v1/read"
    read_recent: false    # Only read remote for data older than local retention
    required_matchers:
      job: ".*"           # Read all jobs from remote
```

---

## 8. High Availability

### 8.1 Prometheus HA with Thanos Deduplication

Run two identical Prometheus instances scraping the same targets. Thanos Query deduplicates:

```yaml
# Prometheus instance A
global:
  external_labels:
    cluster: "production"
    replica: "A"          # Different per replica

# Prometheus instance B
global:
  external_labels:
    cluster: "production"
    replica: "B"          # Different per replica

# Thanos Query deduplication
# thanos query --query.replica-label="replica"
# Result: identical series from A and B are merged, gaps are filled
```

### 8.2 HA Alerting

With HA Prometheus, ensure alerts fire exactly once (not twice):

```yaml
# Alertmanager cluster mode
# alertmanager --cluster.listen-address=0.0.0.0:9094
# alertmanager --cluster.peer=alertmanager-1:9094
# alertmanager --cluster.peer=alertmanager-2:9094

# Both Prometheus replicas send alerts to the Alertmanager cluster.
# Alertmanager deduplicates based on alert fingerprint (name + labels).
```

---

## 9. Cost Optimization

### 9.1 Metrics Cost Model

```
Monthly Cost = Active Series × Ingestion Price
             + Stored Samples × Storage Price
             + Queries × Query Price

Example (Grafana Cloud pricing model):
  1M active series × $8/1000 series = $8,000/mo
  Reducing to 500K series = $4,000/mo (50% savings)
```

### 9.2 Cost Reduction Strategies

| Strategy | Impact | Effort |
|----------|--------|--------|
| Drop unused metrics (`action: drop`) | 20-40% reduction | Low |
| Reduce histogram buckets | 10-30% reduction | Medium |
| Pre-aggregate with recording rules | 10-20% reduction | Medium |
| Shorten scrape interval for non-critical targets | 5-15% reduction | Low |
| Remove redundant labels (`action: labeldrop`) | 10-25% reduction | Low |

### 9.3 Identifying Unused Metrics

```promql
# Metrics that exist but are never queried (requires Grafana Mimirtool or similar)
# Run mimirtool analyze to find unused metrics:
# mimirtool analyze prometheus --address http://prometheus:9090 --grafana-address http://grafana:3000

# Manual: check if a metric is used in any dashboard or alert
# If topk(1, count_over_time(some_metric_total[30d])) returns data but
# the metric appears in zero dashboards and zero alert rules, consider dropping it.

# Find metrics with highest cardinality (candidates for reduction)
topk(20,
  count by (__name__) ({__name__!=""})
)
```

---

## 10. Next Steps

- [23_OpenTelemetry_Pipelines.md](./23_OpenTelemetry_Pipelines.md) -- Design OTel Collector pipelines for production
- [24_eBPF_Observability.md](./24_eBPF_Observability.md) -- Kernel-level observability with eBPF

---

## Exercises

### Exercise 1: Architecture Design

You are designing a metrics architecture for an organization with:
- 3 Kubernetes clusters (us-east, eu-west, ap-southeast)
- 200 microservices (average 500 metrics each)
- 30-day detailed retention, 1-year aggregated retention
- Global dashboards and cross-cluster alerting required

Choose between Thanos and Mimir, justify your decision, and draw the architecture. Estimate the total number of active time series.

<details>
<summary>Show Answer</summary>

**Cardinality estimate:**
```
200 services × 500 metrics × 3 clusters × ~5 label combinations = ~1,500,000 active time series
With recording rules and aggregations: ~2,000,000 active time series
```

**Architecture choice: Thanos** (justification):
- Organization already has Prometheus deployed in each cluster (minimal disruption)
- Sidecar model requires no changes to existing Prometheus configuration
- Object storage (S3) provides cost-effective long-term retention
- Compactor handles downsampling automatically (5m → 1h for long-range)

**Architecture:**
```
                    ┌─────────────────────────┐
                    │     Thanos Query         │
                    │  (global PromQL endpoint) │
                    │  --query.replica-label=   │
                    │    "replica"              │
                    └──┬────────┬────────┬─────┘
                       │        │        │
              ┌────────▼──┐  ┌──▼──────┐ ┌▼──────────┐
              │ US-EAST    │  │ EU-WEST │ │ AP-SE     │
              │ Prom A + B │  │ Prom A+B│ │ Prom A+B  │
              │ + Sidecar  │  │ +Sidecar│ │ +Sidecar  │
              └─────┬──────┘  └────┬────┘ └──────┬────┘
                    │              │              │
                    └──────────────┼──────────────┘
                                   ▼
                    ┌─────────────────────────┐
                    │    S3 Bucket             │
                    │  (Thanos Store Gateway)  │
                    └─────────────────────────┘
                                   │
                    ┌──────────────▼──────────┐
                    │    Thanos Compactor      │
                    │  Raw: 14d, 5m: 90d,     │
                    │  1h: 365d               │
                    └─────────────────────────┘
```

**Key decisions:**
- Each cluster has 2 Prometheus replicas (HA) with Thanos sidecars
- Thanos Query deduplicates across replicas
- Store Gateway serves historical data from S3
- Compactor runs as a singleton, applying retention and downsampling
- Recording rules in each cluster pre-aggregate to reduce federation load

</details>

### Exercise 2: Cardinality Explosion

A developer deployed a new metric that caused Prometheus memory usage to triple. The metric is:

```python
request_trace = Counter(
    "request_trace_total",
    "Request traces",
    labelnames=["method", "path", "status", "trace_id", "user_agent", "source_ip"]
)
```

Diagnose the cardinality issue, calculate the theoretical maximum series count (given 5 methods, 1000 unique paths, 20 statuses, unbounded trace_ids, 500 user agents, 10000 IPs), and write the metric_relabel_configs to fix it.

<details>
<summary>Show Answer</summary>

**Diagnosis:**
```
Theoretical max = 5 × 1000 × 20 × ∞ × 500 × 10,000 = ∞ (unbounded!)

Even without trace_id: 5 × 1000 × 20 × 500 × 10,000 = 500,000,000,000
This is catastrophic.
```

**Root causes:**
1. `trace_id` -- unbounded cardinality (one per request). MUST be removed.
2. `source_ip` -- 10K unique values. Too high for a label.
3. `user_agent` -- 500 unique values. Too high for a label.
4. `path` -- 1000 unique paths (likely includes IDs). Should be route templates.
5. `status` -- 20 statuses. Should be bucketed to classes.

**Fix -- metric_relabel_configs:**

```yaml
metric_relabel_configs:
  # Option 1: Drop the entire metric (immediate fix)
  - source_labels: [__name__]
    regex: "request_trace_total"
    action: drop

  # Option 2: Fix the metric (requires code change too)
  # Remove unbounded and high-cardinality labels
  - source_labels: [__name__]
    regex: "request_trace_total"
    action: keep
  - regex: "trace_id|source_ip|user_agent"
    action: labeldrop
  # Normalize paths to route templates
  - source_labels: [path]
    regex: "/api/users/[0-9]+"
    target_label: path
    replacement: "/api/users/:id"
  - source_labels: [path]
    regex: "/api/orders/[0-9]+"
    target_label: path
    replacement: "/api/orders/:id"
```

**Recommended code change:**

```python
# FIXED metric
request_counter = Counter(
    "http_requests_total",
    "HTTP requests",
    labelnames=["method", "route", "status_class"],  # Low cardinality only
)

# For per-request data, use traces (not metrics)
with tracer.start_as_current_span("handle_request") as span:
    span.set_attribute("http.user_agent", user_agent)  # Fine in traces
    span.set_attribute("net.peer.ip", source_ip)       # Fine in traces
    span.set_attribute("http.trace_id", trace_id)      # Fine in traces
```

**Fixed cardinality:** 5 methods × 30 routes × 5 status classes = 750 time series

</details>

### Exercise 3: Recording Rules

Write a complete set of recording rules for an e-commerce platform that pre-computes:
1. Per-service request rate, error rate, and p99 latency
2. Per-endpoint request rate (top 50 endpoints only)
3. Infrastructure utilization (CPU, memory, disk) per node and per cluster
4. Business metrics: orders per minute, average order value, payment success rate

<details>
<summary>Show Answer</summary>

```yaml
groups:
  - name: service_sli_rules
    interval: 30s
    rules:
      # Per-service request rate
      - record: job:http_requests:rate5m
        expr: sum by (job) (rate(http_requests_total[5m]))

      # Per-service error rate
      - record: job:http_errors:ratio_rate5m
        expr: |
          sum by (job) (rate(http_requests_total{status=~"5.."}[5m]))
          / sum by (job) (rate(http_requests_total[5m]))

      # Per-service p99 latency
      - record: job:http_request_duration:p99_5m
        expr: |
          histogram_quantile(0.99,
            sum by (job, le) (rate(http_request_duration_seconds_bucket[5m])))

  - name: endpoint_rules
    interval: 30s
    rules:
      # Per-endpoint request rate (keep top 50 by volume)
      - record: job_route:http_requests:rate5m
        expr: |
          topk(50,
            sum by (job, route) (rate(http_requests_total[5m]))
          )

      # Per-endpoint error rate (only for top endpoints)
      - record: job_route:http_errors:ratio_rate5m
        expr: |
          sum by (job, route) (rate(http_requests_total{status=~"5.."}[5m]))
          / sum by (job, route) (rate(http_requests_total[5m]))

  - name: infrastructure_rules
    interval: 60s
    rules:
      # Per-node CPU utilization
      - record: instance:node_cpu:ratio_rate5m
        expr: 1 - avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m]))

      # Cluster average CPU
      - record: cluster:node_cpu:avg_ratio
        expr: avg(instance:node_cpu:ratio_rate5m)

      # Per-node memory utilization
      - record: instance:node_memory:ratio
        expr: |
          1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)

      # Cluster average memory
      - record: cluster:node_memory:avg_ratio
        expr: avg(instance:node_memory:ratio)

      # Per-node disk utilization (root filesystem)
      - record: instance:node_disk:ratio
        expr: |
          1 - (
            node_filesystem_avail_bytes{mountpoint="/"}
            / node_filesystem_size_bytes{mountpoint="/"}
          )

  - name: business_rules
    interval: 30s
    rules:
      # Orders per minute
      - record: business:orders:rate1m
        expr: sum(rate(orders_created_total[1m])) * 60

      # Average order value (5-minute window for smoothing)
      - record: business:order_value:avg5m
        expr: |
          rate(order_value_dollars_sum[5m])
          / rate(order_value_dollars_count[5m])

      # Payment success rate
      - record: business:payment_success:ratio_rate5m
        expr: |
          sum(rate(payments_total{status="success"}[5m]))
          / sum(rate(payments_total[5m]))

      # Revenue rate (dollars per minute)
      - record: business:revenue:rate1m
        expr: sum(rate(order_value_dollars_sum[1m])) * 60
```

</details>

---

## References

- [Prometheus Federation](https://prometheus.io/docs/prometheus/latest/federation/)
- [Thanos Documentation](https://thanos.io/tip/thanos/getting-started.md/)
- [Grafana Mimir Documentation](https://grafana.com/docs/mimir/latest/)
- [Prometheus Recording Rules](https://prometheus.io/docs/prometheus/latest/configuration/recording_rules/)
- [Robust Perception -- Cardinality](https://www.robustperception.io/cardinality-is-key/)
- [Prometheus Remote Write Specification](https://prometheus.io/docs/concepts/remote_write_spec/)

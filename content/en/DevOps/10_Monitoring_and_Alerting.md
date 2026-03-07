# Monitoring and Alerting

**Previous**: [Kubernetes Orchestration](./09_Kubernetes_Orchestration.md) | **Next**: [Logging Infrastructure](./11_Logging_Infrastructure.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the role of monitoring in DevOps and distinguish between the four types of metrics (counter, gauge, histogram, summary)
2. Describe Prometheus architecture and its pull-based model for metrics collection
3. Write PromQL queries to aggregate, filter, and alert on time-series data
4. Design Grafana dashboards that visualize key infrastructure and application metrics
5. Configure alerting rules with appropriate thresholds, severities, and routing via Alertmanager
6. Apply monitoring best practices including the USE method, RED method, and the four golden signals

---

Monitoring is the foundation of operational visibility. Without it, you are flying blind -- unable to detect problems before they affect users, unable to understand system behavior under load, and unable to make data-driven capacity decisions. This lesson covers Prometheus as the de facto open-source monitoring standard, PromQL for querying metrics, Grafana for visualization, and Alertmanager for intelligent alert routing.

> **Analogy -- Dashboard of a Car**: Monitoring is like the instrument cluster in a car. Gauges show speed (throughput), fuel level (resource usage), and engine temperature (saturation). Warning lights (alerts) fire only when a threshold is breached, and each light routes to a different action: low fuel means refuel, overheating means pull over. Without the dashboard, you would not know your engine was failing until smoke billowed from the hood.

## 1. Why Monitoring Matters

### 1.1 The Monitoring Feedback Loop

Monitoring closes the feedback loop between deployment and operations:

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Deploy  │────→│ Observe  │────→│  Detect  │────→│ Respond  │
│  Change  │     │ Metrics  │     │ Anomaly  │     │  / Fix   │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
      ↑                                                  │
      └──────────────────────────────────────────────────┘
                         Feedback Loop
```

### 1.2 The Three Pillars of Observability

| Pillar | Purpose | Tools |
|--------|---------|-------|
| **Metrics** | Numerical measurements over time (CPU, latency, error rates) | Prometheus, Datadog, CloudWatch |
| **Logs** | Discrete event records with context | ELK Stack, Loki, Splunk |
| **Traces** | Request flow across distributed services | Jaeger, Zipkin, OpenTelemetry |

This lesson focuses on **metrics and alerting**. Logs and traces are covered in lessons 11 and 12.

### 1.3 Monitoring Methodologies

**The Four Golden Signals** (from Google SRE):

| Signal | Description | Example Metric |
|--------|-------------|----------------|
| **Latency** | Time to serve a request | `http_request_duration_seconds` |
| **Traffic** | Demand on the system | `http_requests_total` |
| **Errors** | Rate of failed requests | `http_requests_total{status=~"5.."}` |
| **Saturation** | How full the system is | CPU utilization, memory pressure |

**The USE Method** (for infrastructure resources):
- **U**tilization -- percentage of time the resource is busy
- **S**aturation -- amount of work the resource cannot serve (queue depth)
- **E**rrors -- count of error events

**The RED Method** (for request-driven services):
- **R**ate -- requests per second
- **E**rrors -- failed requests per second
- **D**uration -- distribution of request latencies

---

## 2. Prometheus Architecture

### 2.1 Core Components

Prometheus is an open-source monitoring system built for reliability and multi-dimensional data collection.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Prometheus Ecosystem                         │
│                                                                  │
│  ┌────────────┐    pull     ┌──────────────┐                    │
│  │  Targets   │◄───────────│  Prometheus   │                    │
│  │ (exporters │    /metrics │   Server      │                    │
│  │  & apps)   │            │  ┌──────────┐ │    ┌────────────┐  │
│  └────────────┘            │  │  TSDB    │ │───→│ Alertmanager│  │
│                            │  │(storage) │ │    │  (routing,  │  │
│  ┌────────────┐            │  └──────────┘ │    │  grouping)  │  │
│  │ Pushgateway│───push────→│  ┌──────────┐ │    └────────────┘  │
│  │(short-lived│            │  │  PromQL  │ │           │        │
│  │   jobs)    │            │  │ (query)  │ │    ┌──────┴─────┐  │
│  └────────────┘            │  └──────────┘ │    │   Slack /  │  │
│                            └──────────────┘    │  PagerDuty │  │
│  ┌────────────┐                   │            │   / Email  │  │
│  │  Service   │                   │            └────────────┘  │
│  │ Discovery  │──────────────────→│                             │
│  │(K8s, DNS,  │                   ▼                             │
│  │ Consul)    │            ┌──────────────┐                    │
│  └────────────┘            │   Grafana    │                    │
│                            │(visualization)│                    │
│                            └──────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Pull-Based vs Push-Based Models

| Aspect | Pull-Based (Prometheus) | Push-Based (Datadog, Graphite) |
|--------|------------------------|-------------------------------|
| **Direction** | Server scrapes targets | Targets push to server |
| **Discovery** | Prometheus discovers targets | Targets must know the server |
| **Health detection** | Scrape failure = target down | No data = ambiguous |
| **Short-lived jobs** | Needs Pushgateway | Natural fit |
| **Network** | Prometheus needs access to targets | Targets need access to server |
| **Scaling** | Federation for large scale | Aggregation proxies |

### 2.3 Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s          # How often to scrape targets
  evaluation_interval: 15s      # How often to evaluate rules
  scrape_timeout: 10s           # Timeout per scrape

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - "alertmanager:9093"

# Rule files for alerts and recording rules
rule_files:
  - "alerts/*.yml"
  - "recording_rules/*.yml"

# Scrape targets
scrape_configs:
  # Prometheus self-monitoring
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]

  # Node Exporter (system metrics)
  - job_name: "node"
    static_configs:
      - targets:
          - "node1:9100"
          - "node2:9100"
          - "node3:9100"

  # Application metrics
  - job_name: "webapp"
    metrics_path: "/metrics"
    scrape_interval: 10s
    static_configs:
      - targets: ["webapp:8080"]
        labels:
          environment: "production"
          team: "backend"

  # Kubernetes service discovery
  - job_name: "kubernetes-pods"
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
```

### 2.4 Instrumenting Applications

**Python (prometheus_client):**

```python
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
import time

# Counter: monotonically increasing value (total requests, errors)
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

# Gauge: value that can go up and down (temperature, connections)
ACTIVE_CONNECTIONS = Gauge(
    "active_connections",
    "Number of active connections"
)

# Histogram: distribution of values in configurable buckets
REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Summary: similar to histogram but calculates quantiles on the client side
REQUEST_LATENCY = Summary(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method"]
)

def handle_request(method, endpoint):
    ACTIVE_CONNECTIONS.inc()
    start = time.time()

    try:
        # Process request
        process(method, endpoint)
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status="200").inc()
    except Exception:
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status="500").inc()
        raise
    finally:
        duration = time.time() - start
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
        REQUEST_LATENCY.labels(method=method).observe(duration)
        ACTIVE_CONNECTIONS.dec()

# Start metrics server on port 8000
start_http_server(8000)
```

**Go (prometheus/client_golang):**

```go
package main

import (
    "net/http"
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
    requestsTotal = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "http_requests_total",
            Help: "Total HTTP requests",
        },
        []string{"method", "endpoint", "status"},
    )

    requestDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "http_request_duration_seconds",
            Help:    "HTTP request duration in seconds",
            Buckets: prometheus.DefBuckets,
        },
        []string{"method", "endpoint"},
    )
)

func init() {
    prometheus.MustRegister(requestsTotal)
    prometheus.MustRegister(requestDuration)
}

func main() {
    http.Handle("/metrics", promhttp.Handler())
    http.ListenAndServe(":8080", nil)
}
```

---

## 3. Metric Types

### 3.1 The Four Metric Types

| Type | Behavior | Use Case | Example |
|------|----------|----------|---------|
| **Counter** | Monotonically increasing | Total requests, errors, bytes | `http_requests_total` |
| **Gauge** | Goes up and down | Temperature, memory, queue size | `node_memory_MemAvailable_bytes` |
| **Histogram** | Distributes values into buckets | Latency distribution, response sizes | `http_request_duration_seconds` |
| **Summary** | Calculates quantiles client-side | Latency percentiles | `go_gc_duration_seconds` |

### 3.2 Histogram vs Summary

| Aspect | Histogram | Summary |
|--------|-----------|---------|
| **Quantile calculation** | Server-side (PromQL) | Client-side (application) |
| **Aggregation** | Aggregatable across instances | Not aggregatable |
| **Bucket configuration** | Must define buckets in advance | Define quantile targets (e.g., 0.5, 0.95, 0.99) |
| **Accuracy** | Depends on bucket boundaries | Configurable error margin |
| **CPU cost** | Low on client | Higher on client |
| **Best for** | Most use cases, SLO tracking | When exact quantiles are needed per instance |

**Recommendation**: Prefer histograms in most cases. They can be aggregated across multiple instances, and Prometheus can calculate quantiles from bucket data using `histogram_quantile()`.

### 3.3 Naming Conventions

Follow Prometheus naming best practices:

```
# Format: <namespace>_<name>_<unit>
# - Use snake_case
# - Include unit as suffix (_seconds, _bytes, _total)
# - _total suffix for counters
# - _info suffix for info metrics (gauge with value 1)

# Good
http_requests_total
http_request_duration_seconds
node_memory_MemAvailable_bytes
process_cpu_seconds_total

# Bad
httpRequests              # camelCase, no unit
request_latency           # ambiguous unit
num_errors                # use _total suffix for counters
```

---

## 4. PromQL (Prometheus Query Language)

### 4.1 Basic Queries

```promql
# Instant vector: current value of a metric
http_requests_total

# Filter by labels
http_requests_total{method="GET", status="200"}

# Regex matching
http_requests_total{status=~"5.."}        # All 5xx status codes
http_requests_total{method!="OPTIONS"}     # Exclude OPTIONS
http_requests_total{endpoint=~"/api/.*"}   # Endpoints starting with /api/
```

### 4.2 Range Vectors and Functions

```promql
# Range vector: values over a time range
http_requests_total[5m]     # Last 5 minutes of data points

# Rate: per-second average rate of increase (for counters)
rate(http_requests_total[5m])

# irate: instant rate based on last two data points (more volatile)
irate(http_requests_total[5m])

# increase: total increase over the range
increase(http_requests_total[1h])   # Total requests in last hour

# When to use rate vs irate:
# - rate()  → smoothed average, better for alerts and dashboards
# - irate() → captures spikes, better for volatile short-term graphs
```

### 4.3 Aggregation Operators

```promql
# Sum across all instances
sum(rate(http_requests_total[5m]))

# Sum grouped by status code
sum by (status) (rate(http_requests_total[5m]))

# Average request duration per endpoint
avg by (endpoint) (rate(http_request_duration_seconds_sum[5m])
  / rate(http_request_duration_seconds_count[5m]))

# Top 5 endpoints by request rate
topk(5, sum by (endpoint) (rate(http_requests_total[5m])))

# Count number of targets with high error rate
count(rate(http_requests_total{status=~"5.."}[5m]) > 0.1)

# Quantile from histogram
histogram_quantile(0.95,
  sum by (le) (rate(http_request_duration_seconds_bucket[5m]))
)

# 99th percentile latency per endpoint
histogram_quantile(0.99,
  sum by (le, endpoint) (rate(http_request_duration_seconds_bucket[5m]))
)
```

### 4.4 Useful PromQL Patterns

```promql
# Error rate (percentage of 5xx responses)
sum(rate(http_requests_total{status=~"5.."}[5m]))
  / sum(rate(http_requests_total[5m])) * 100

# Availability (percentage of successful responses)
1 - (
  sum(rate(http_requests_total{status=~"5.."}[5m]))
  / sum(rate(http_requests_total[5m]))
)

# Saturation: CPU usage percentage
100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Memory usage percentage
(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100

# Disk space remaining (hours until full, based on last 24h trend)
predict_linear(node_filesystem_avail_bytes[24h], 3600 * 24)

# Request rate change compared to 1 week ago
rate(http_requests_total[5m])
  / rate(http_requests_total[5m] offset 1w)
```

---

## 5. Grafana Dashboards

### 5.1 Data Source Configuration

```yaml
# grafana/provisioning/datasources/prometheus.yml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    jsonData:
      timeInterval: "15s"
      httpMethod: POST
```

### 5.2 Dashboard Design Principles

**Layout Strategy:**

```
┌─────────────────────────────────────────────────────────────┐
│  Row 1: Overview (single-stat panels)                       │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐   │
│  │ RPS  │ │ p50  │ │ p95  │ │ p99  │ │Error%│ │Uptime│   │
│  │ 1.2k │ │ 45ms │ │120ms │ │350ms │ │ 0.3% │ │99.97%│   │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘   │
├─────────────────────────────────────────────────────────────┤
│  Row 2: Request & Error Rates (time-series graphs)          │
│  ┌────────────────────────┐ ┌────────────────────────┐     │
│  │  Request Rate by       │ │  Error Rate by          │     │
│  │  Endpoint (stacked)    │ │  Status Code            │     │
│  └────────────────────────┘ └────────────────────────┘     │
├─────────────────────────────────────────────────────────────┤
│  Row 3: Latency (heatmap + percentile graph)                │
│  ┌────────────────────────┐ ┌────────────────────────┐     │
│  │  Latency Heatmap       │ │  p50/p95/p99 Latency   │     │
│  └────────────────────────┘ └────────────────────────┘     │
├─────────────────────────────────────────────────────────────┤
│  Row 4: Infrastructure (CPU, Memory, Disk, Network)         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ CPU %    │ │ Memory % │ │ Disk I/O │ │ Network  │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
└─────────────────────────────────────────────────────────────┘
```

**Best Practices:**
1. Put the most critical panels (golden signals) at the top
2. Use consistent color schemes (green = healthy, yellow = warning, red = critical)
3. Include time comparison (current vs. 1 day ago / 1 week ago)
4. Use template variables for filtering (environment, service, instance)
5. Add annotations for deployments and incidents

### 5.3 Dashboard as Code (JSON Model)

```json
{
  "dashboard": {
    "title": "Service Overview",
    "uid": "service-overview",
    "tags": ["production", "backend"],
    "timezone": "browser",
    "refresh": "30s",
    "templating": {
      "list": [
        {
          "name": "environment",
          "type": "query",
          "query": "label_values(http_requests_total, environment)",
          "current": { "text": "production", "value": "production" }
        },
        {
          "name": "service",
          "type": "query",
          "query": "label_values(http_requests_total{environment=\"$environment\"}, job)"
        }
      ]
    },
    "panels": [
      {
        "title": "Request Rate",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 },
        "targets": [
          {
            "expr": "sum by (endpoint) (rate(http_requests_total{environment=\"$environment\", job=\"$service\"}[5m]))",
            "legendFormat": "{{endpoint}}"
          }
        ]
      },
      {
        "title": "Error Rate %",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 0 },
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{environment=\"$environment\", job=\"$service\", status=~\"5..\"}[5m])) / sum(rate(http_requests_total{environment=\"$environment\", job=\"$service\"}[5m])) * 100",
            "legendFormat": "Error %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "steps": [
                { "color": "green", "value": null },
                { "color": "yellow", "value": 1 },
                { "color": "red", "value": 5 }
              ]
            }
          }
        }
      }
    ]
  }
}
```

---

## 6. Alerting Rules

### 6.1 Prometheus Alerting Rules

```yaml
# alerts/application.yml
groups:
  - name: application_alerts
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m]))
          / sum(rate(http_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
          team: backend
        annotations:
          summary: "High error rate detected"
          description: >
            Error rate is {{ $value | humanizePercentage }} over
            the last 5 minutes (threshold: 5%).
          runbook_url: "https://wiki.example.com/runbooks/high-error-rate"

      # High latency (p95 > 1 second)
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            sum by (le) (rate(http_request_duration_seconds_bucket[5m]))
          ) > 1.0
        for: 10m
        labels:
          severity: warning
          team: backend
        annotations:
          summary: "p95 latency exceeds 1 second"
          description: "p95 latency is {{ $value }}s (threshold: 1s)."

      # Target down
      - alert: TargetDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Target {{ $labels.instance }} is down"
          description: "{{ $labels.job }}/{{ $labels.instance }} has been unreachable for 1 minute."

  - name: infrastructure_alerts
    rules:
      # High CPU usage
      - alert: HighCPUUsage
        expr: |
          100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 85
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage on {{ $labels.instance }}"
          description: "CPU usage is {{ $value }}% (threshold: 85%)."

      # Disk space running low
      - alert: DiskSpaceLow
        expr: |
          (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100 < 10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Disk space below 10% on {{ $labels.instance }}"
          description: "Available disk space is {{ $value }}%."

      # Disk will fill within 24 hours
      - alert: DiskWillFillIn24Hours
        expr: |
          predict_linear(node_filesystem_avail_bytes{mountpoint="/"}[6h], 24 * 3600) < 0
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "Disk on {{ $labels.instance }} predicted to fill within 24 hours"
```

### 6.2 The `for` Clause and Pending State

```
Alert States:
┌──────────┐         ┌──────────┐         ┌──────────┐
│ Inactive │──expr──→│ Pending  │──for────→│  Firing  │──────→ Notification
│          │  true   │          │ elapsed  │          │
└──────────┘         └──────────┘         └──────────┘
      ↑                   │                     │
      └───────────────────┘                     │
        expr becomes false                      │
      ↑                                         │
      └─────────────────────────────────────────┘
        expr becomes false (sends "resolved")
```

- **No `for`**: Alert fires immediately when expression is true (noisy)
- **`for: 5m`**: Expression must be continuously true for 5 minutes before firing (reduces false positives)
- **Too long `for`**: Delays genuine alerts; use 1-5 minutes for critical, 10-15 for warning

---

## 7. Alertmanager

### 7.1 Alertmanager Configuration

```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m
  smtp_smarthost: "smtp.example.com:587"
  smtp_from: "alerts@example.com"
  smtp_auth_username: "alerts@example.com"

# Inhibition rules: suppress lower-severity alerts when higher-severity fires
inhibit_rules:
  - source_matchers:
      - severity="critical"
    target_matchers:
      - severity="warning"
    equal: ["alertname", "instance"]

# Routing tree
route:
  receiver: "default-slack"
  group_by: ["alertname", "team"]
  group_wait: 30s        # Wait before sending first notification
  group_interval: 5m     # Wait before sending updates to a group
  repeat_interval: 4h    # Re-send if alert is still firing
  routes:
    # Critical alerts → PagerDuty
    - matchers:
        - severity="critical"
      receiver: "pagerduty-critical"
      repeat_interval: 1h
      continue: false

    # Warning alerts → Slack
    - matchers:
        - severity="warning"
      receiver: "team-slack"
      repeat_interval: 4h

    # Infrastructure alerts → infra team
    - matchers:
        - team="infrastructure"
      receiver: "infra-slack"

receivers:
  - name: "default-slack"
    slack_configs:
      - api_url: "https://hooks.slack.com/services/T.../B.../..."
        channel: "#alerts-general"
        title: '{{ .CommonAnnotations.summary }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: "pagerduty-critical"
    pagerduty_configs:
      - routing_key: "<pagerduty-service-key>"
        severity: "critical"

  - name: "team-slack"
    slack_configs:
      - api_url: "https://hooks.slack.com/services/T.../B.../..."
        channel: "#alerts-backend"

  - name: "infra-slack"
    slack_configs:
      - api_url: "https://hooks.slack.com/services/T.../B.../..."
        channel: "#alerts-infrastructure"
```

### 7.2 Alert Routing Flow

```
              Incoming Alert
                    │
                    ▼
           ┌───────────────┐
           │  Group by     │  group_by: [alertname, team]
           │  (aggregate)  │
           └───────┬───────┘
                   │
                   ▼
           ┌───────────────┐
           │  Match route  │  severity="critical"?
           │  (top-down)   │
           └───────┬───────┘
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
    PagerDuty   Slack    Email
    (critical) (warning) (default)
```

### 7.3 Silences and Maintenance Windows

```bash
# Create a silence for planned maintenance (via amtool)
amtool silence add \
  alertname="HighCPUUsage" \
  instance="node3:9100" \
  --duration=2h \
  --comment="Scheduled maintenance on node3" \
  --author="ops-team"

# List active silences
amtool silence query

# Expire a silence early
amtool silence expire <silence-id>
```

---

## 8. Common Exporters

### 8.1 Popular Prometheus Exporters

| Exporter | Metrics | Default Port |
|----------|---------|-------------|
| **Node Exporter** | CPU, memory, disk, network (Linux) | 9100 |
| **Windows Exporter** | CPU, memory, disk (Windows) | 9182 |
| **Blackbox Exporter** | HTTP/DNS/TCP/ICMP probes | 9115 |
| **MySQL Exporter** | Queries, connections, replication | 9104 |
| **PostgreSQL Exporter** | Connections, locks, replication lag | 9187 |
| **Redis Exporter** | Memory, commands, keyspace | 9121 |
| **Nginx Exporter** | Connections, requests | 9113 |
| **cAdvisor** | Container CPU, memory, network | 8080 |
| **kube-state-metrics** | Kubernetes object states | 8080 |

### 8.2 Blackbox Exporter (Endpoint Probing)

```yaml
# blackbox.yml
modules:
  http_2xx:
    prober: http
    timeout: 5s
    http:
      valid_http_versions: ["HTTP/1.1", "HTTP/2.0"]
      valid_status_codes: [200]
      method: GET
      follow_redirects: true
  tcp_connect:
    prober: tcp
    timeout: 5s

# prometheus.yml scrape config for blackbox
scrape_configs:
  - job_name: "blackbox-http"
    metrics_path: /probe
    params:
      module: [http_2xx]
    static_configs:
      - targets:
          - "https://example.com"
          - "https://api.example.com/health"
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - source_labels: [__param_target]
        target_label: instance
      - target_label: __address__
        replacement: "blackbox-exporter:9115"
```

---

## 9. Recording Rules and Federation

### 9.1 Recording Rules

Recording rules pre-compute expensive queries and store results as new time series:

```yaml
# recording_rules/aggregations.yml
groups:
  - name: request_rate_rules
    interval: 30s
    rules:
      # Pre-compute request rate by service
      - record: job:http_requests_total:rate5m
        expr: sum by (job) (rate(http_requests_total[5m]))

      # Pre-compute error percentage
      - record: job:http_errors:ratio_rate5m
        expr: |
          sum by (job) (rate(http_requests_total{status=~"5.."}[5m]))
          / sum by (job) (rate(http_requests_total[5m]))

      # Pre-compute p95 latency
      - record: job:http_request_duration_seconds:p95
        expr: |
          histogram_quantile(0.95,
            sum by (job, le) (rate(http_request_duration_seconds_bucket[5m]))
          )
```

**Benefits of recording rules:**
- Dashboard queries load faster (pre-computed vs. on-demand)
- Enables long-range queries without high cardinality overhead
- Alert rules reference recording rules for consistency

### 9.2 Federation

For large-scale deployments, use Prometheus federation to aggregate metrics from multiple Prometheus servers:

```yaml
# Global Prometheus scrapes from regional Prometheus instances
scrape_configs:
  - job_name: "federate"
    honor_labels: true
    metrics_path: "/federate"
    params:
      'match[]':
        - '{job=~".+"}'               # All job metrics
        - 'job:http_requests_total:rate5m'  # Recording rules
    static_configs:
      - targets:
          - "prometheus-us-east:9090"
          - "prometheus-eu-west:9090"
          - "prometheus-ap-east:9090"
```

---

## 10. Next Steps

- [11_Logging_Infrastructure.md](./11_Logging_Infrastructure.md) - Centralized logging with ELK and Loki
- [12_Distributed_Tracing.md](./12_Distributed_Tracing.md) - Request tracing with OpenTelemetry and Jaeger

---

## Exercises

### Exercise 1: Metric Type Selection

For each scenario, choose the correct Prometheus metric type (Counter, Gauge, Histogram, or Summary) and explain why:

1. Tracking the total number of HTTP 500 errors since the process started.
2. Measuring the current number of items in a message queue.
3. Recording the distribution of API response times to calculate percentiles across multiple instances.
4. Tracking the total bytes of data transferred.

<details>
<summary>Show Answer</summary>

1. **Counter** -- Error counts only increase (or reset on process restart). You use `rate()` to compute the per-second error rate. Counters are the correct type for any cumulative total.

2. **Gauge** -- Queue depth goes up and down as items are enqueued and dequeued. Gauges are the correct type for any value that fluctuates.

3. **Histogram** -- You need to calculate percentiles across multiple instances, which requires server-side aggregation. Histograms store values in buckets that can be summed across instances, then `histogram_quantile()` computes the percentile. Summaries calculate quantiles per-instance and cannot be meaningfully aggregated.

4. **Counter** -- Total bytes transferred is a monotonically increasing value. Use `rate(bytes_transferred_total[5m])` to compute throughput in bytes per second.

</details>

### Exercise 2: Write PromQL Queries

Write PromQL queries for the following scenarios:

1. Calculate the total request rate (requests per second) across all instances of the `webapp` job.
2. Find the 99th percentile request duration for the `/api/orders` endpoint over the last 10 minutes.
3. Calculate the error rate as a percentage (5xx responses / total responses) for the `payment-service` job.
4. Alert when any instance's available disk space is below 15%.

<details>
<summary>Show Answer</summary>

1. Total request rate:
```promql
sum(rate(http_requests_total{job="webapp"}[5m]))
```

2. 99th percentile latency for `/api/orders`:
```promql
histogram_quantile(0.99,
  sum by (le) (rate(http_request_duration_seconds_bucket{endpoint="/api/orders"}[10m]))
)
```

3. Error rate percentage:
```promql
sum(rate(http_requests_total{job="payment-service", status=~"5.."}[5m]))
/ sum(rate(http_requests_total{job="payment-service"}[5m])) * 100
```

4. Disk space alert expression:
```promql
(node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100 < 15
```

</details>

### Exercise 3: Alert Design

Design an alerting rule for the following scenario:

A payment processing service must have less than 0.1% error rate and p99 latency below 500ms. If either SLO is violated for more than 3 minutes, a critical alert should fire. A warning should fire at 0.05% error rate and 300ms p99 latency with a 10-minute window.

Write the complete Prometheus alerting rules YAML.

<details>
<summary>Show Answer</summary>

```yaml
groups:
  - name: payment_slo_alerts
    rules:
      # Critical: error rate > 0.1% for 3 minutes
      - alert: PaymentHighErrorRate
        expr: |
          sum(rate(http_requests_total{job="payment-service", status=~"5.."}[5m]))
          / sum(rate(http_requests_total{job="payment-service"}[5m])) > 0.001
        for: 3m
        labels:
          severity: critical
          team: payments
        annotations:
          summary: "Payment service error rate SLO violation"
          description: >
            Error rate is {{ $value | humanizePercentage }}
            (SLO threshold: 0.1%).
          runbook_url: "https://wiki.example.com/runbooks/payment-errors"

      # Critical: p99 latency > 500ms for 3 minutes
      - alert: PaymentHighLatency
        expr: |
          histogram_quantile(0.99,
            sum by (le) (rate(http_request_duration_seconds_bucket{job="payment-service"}[5m]))
          ) > 0.5
        for: 3m
        labels:
          severity: critical
          team: payments
        annotations:
          summary: "Payment service p99 latency SLO violation"
          description: "p99 latency is {{ $value }}s (SLO threshold: 500ms)."

      # Warning: error rate > 0.05% for 10 minutes
      - alert: PaymentElevatedErrorRate
        expr: |
          sum(rate(http_requests_total{job="payment-service", status=~"5.."}[5m]))
          / sum(rate(http_requests_total{job="payment-service"}[5m])) > 0.0005
        for: 10m
        labels:
          severity: warning
          team: payments
        annotations:
          summary: "Payment service error rate approaching SLO threshold"
          description: "Error rate is {{ $value | humanizePercentage }} (warning at 0.05%)."

      # Warning: p99 latency > 300ms for 10 minutes
      - alert: PaymentElevatedLatency
        expr: |
          histogram_quantile(0.99,
            sum by (le) (rate(http_request_duration_seconds_bucket{job="payment-service"}[5m]))
          ) > 0.3
        for: 10m
        labels:
          severity: warning
          team: payments
        annotations:
          summary: "Payment service p99 latency approaching SLO threshold"
          description: "p99 latency is {{ $value }}s (warning at 300ms)."
```

**Key design decisions:**
- The critical alert uses `for: 3m` to catch violations quickly.
- The warning alert uses `for: 10m` to avoid noise from transient spikes.
- Both use `rate()[5m]` to smooth over brief fluctuations within the evaluation window.
- The inhibition rule in Alertmanager (Section 7.1) ensures that when the critical alert fires, the warning alert is suppressed.

</details>

### Exercise 4: Dashboard Planning

You are building a Grafana dashboard for a microservice that handles user authentication. List the panels you would include, the PromQL query for each panel, and explain why each metric matters for this specific service.

<details>
<summary>Show Answer</summary>

**Row 1: Overview (Stat panels)**

| Panel | PromQL | Rationale |
|-------|--------|-----------|
| Login Rate | `sum(rate(auth_login_attempts_total[5m]))` | Shows current demand on the auth service |
| Login Success Rate | `sum(rate(auth_login_attempts_total{result="success"}[5m])) / sum(rate(auth_login_attempts_total[5m])) * 100` | Drops indicate credential issues or attacks |
| p95 Latency | `histogram_quantile(0.95, sum by (le) (rate(auth_request_duration_seconds_bucket[5m])))` | Auth latency directly impacts every downstream service |
| Active Sessions | `auth_active_sessions` (gauge) | Capacity indicator |

**Row 2: Request & Error Detail (Time-series)**

| Panel | PromQL | Rationale |
|-------|--------|-----------|
| Login Attempts by Result | `sum by (result) (rate(auth_login_attempts_total[5m]))` | Distinguishes success, wrong password, account locked, MFA failed |
| Token Operations | `sum by (operation) (rate(auth_token_operations_total[5m]))` | Tracks token issuance, refresh, and revocation rates |

**Row 3: Security (Time-series)**

| Panel | PromQL | Rationale |
|-------|--------|-----------|
| Failed Login Rate by IP | `topk(10, sum by (source_ip) (rate(auth_login_attempts_total{result="failed"}[5m])))` | Detects brute force attacks from specific IPs |
| Account Lockouts | `sum(rate(auth_account_lockouts_total[5m]))` | Spike in lockouts may indicate credential stuffing |

**Row 4: Infrastructure (Time-series)**

| Panel | PromQL | Rationale |
|-------|--------|-----------|
| CPU | `rate(process_cpu_seconds_total{job="auth-service"}[5m])` | Auth services can be CPU-intensive (bcrypt hashing) |
| Memory | `process_resident_memory_bytes{job="auth-service"}` | Session stores can cause memory growth |
| DB Connection Pool | `auth_db_pool_active_connections` | Auth services are DB-bound; pool exhaustion causes cascading failures |

</details>

---

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [PromQL Cheat Sheet](https://promlabs.com/promql-cheat-sheet/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Google SRE Book - Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)
- [The USE Method](https://www.brendangregg.com/usemethod.html)
- [The RED Method](https://grafana.com/blog/2018/08/02/the-red-method-how-to-instrument-your-services/)

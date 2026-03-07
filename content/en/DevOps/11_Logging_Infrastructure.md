# Logging Infrastructure

**Previous**: [Monitoring and Alerting](./10_Monitoring_and_Alerting.md) | **Next**: [Distributed Tracing](./12_Distributed_Tracing.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the role of centralized logging in DevOps and why log aggregation is essential for distributed systems
2. Design a logging pipeline using the ELK stack (Elasticsearch, Logstash, Kibana) and understand each component's responsibility
3. Compare the ELK stack with the Loki/Grafana stack and choose the right solution based on scale and cost constraints
4. Implement structured logging in applications using JSON format and consistent field conventions
5. Configure log levels, retention policies, and index lifecycle management for production environments
6. Build effective log queries and dashboards for incident investigation and operational visibility

---

Logs are the narrative record of everything that happens in your systems. While metrics tell you that something is wrong (error rate spiked to 5%), logs tell you why (a database connection pool was exhausted because a migration locked a table). In distributed systems, logs from dozens of services must be aggregated into a single searchable store, or debugging becomes a hopeless exercise of SSH-ing into individual machines. This lesson covers the two dominant open-source logging stacks -- ELK and Loki -- along with structured logging practices and retention strategies.

> **Analogy -- Flight Data Recorder**: Logs are like an airplane's black box flight recorder. Every event is captured in sequence with timestamps, and when something goes wrong, investigators reconstruct the chain of events from the recordings. Without the black box (centralized logging), crash investigators would have to interview passengers one by one (SSH into each server) -- slow, incomplete, and unreliable.

## 1. Why Centralized Logging

### 1.1 The Problem with Local Logs

In a distributed system, logs scattered across individual hosts create critical problems:

```
┌─────────────────────────────────────────────────────────────┐
│  Without Centralized Logging                                 │
│                                                              │
│  Service A (host-1)     Service B (host-2)     Service C     │
│  /var/log/app.log       /var/log/app.log     (host-3, 4, 5) │
│                                                              │
│  Problems:                                                   │
│  1. Which host has the log for request X?                    │
│  2. Containers die → logs are lost                           │
│  3. Cannot correlate events across services                  │
│  4. No full-text search across all logs                      │
│  5. Cannot set alerts on log patterns                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  With Centralized Logging                                    │
│                                                              │
│  Service A ──┐                                               │
│  Service B ──┼──→ Log Aggregator ──→ Search & Visualize     │
│  Service C ──┘    (single store)     (single dashboard)      │
│                                                              │
│  Benefits:                                                   │
│  1. Search all logs from one interface                       │
│  2. Correlate events across services using request IDs       │
│  3. Survive container restarts and host failures             │
│  4. Alert on error patterns                                  │
│  5. Comply with audit and retention requirements             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Log Levels

Standard log levels follow a severity hierarchy:

| Level | Numeric | Purpose | Example |
|-------|---------|---------|---------|
| **TRACE** | 5 | Fine-grained debugging | Function entry/exit, variable values |
| **DEBUG** | 10 | Diagnostic information | SQL query details, cache hit/miss |
| **INFO** | 20 | Normal operations | Request served, job completed, user logged in |
| **WARNING** | 30 | Unexpected but handled | Retry succeeded, deprecated API called |
| **ERROR** | 40 | Operation failed | Database connection failed, API call returned 500 |
| **CRITICAL/FATAL** | 50 | System is unusable | Out of memory, cannot bind to port |

**Production guidelines:**
- Set default log level to `INFO`
- Use `DEBUG` only in development or temporarily during incidents
- Never log sensitive data (passwords, tokens, PII) at any level
- Every `ERROR` log should be actionable -- if you cannot act on it, it should be `WARNING`

---

## 2. Structured Logging

### 2.1 Unstructured vs Structured Logs

**Unstructured (bad for parsing):**
```
2024-03-15 14:23:45 INFO User john@example.com logged in from 192.168.1.100 using Chrome
```

**Structured JSON (machine-parseable):**
```json
{
  "timestamp": "2024-03-15T14:23:45.123Z",
  "level": "INFO",
  "logger": "auth.service",
  "message": "User logged in",
  "user_email": "john@example.com",
  "source_ip": "192.168.1.100",
  "user_agent": "Chrome/122.0",
  "request_id": "req-abc123",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "service": "auth-service",
  "environment": "production"
}
```

### 2.2 Structured Logging in Python

```python
import structlog
import logging

# Configure structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer()        # Human-readable in dev
        # structlog.processors.JSONRenderer()  # JSON in production
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger()

# Usage
log.info("user_login", user_email="john@example.com", source_ip="192.168.1.100")
log.error("payment_failed", order_id="ord-456", reason="insufficient_funds", amount=99.99)

# Bind context for the duration of a request
log = log.bind(request_id="req-abc123", service="auth-service")
log.info("processing_request")    # Automatically includes request_id and service
log.info("request_complete", duration_ms=45)
```

### 2.3 Structured Logging in Go

```go
package main

import (
    "go.uber.org/zap"
)

func main() {
    // Production logger (JSON output)
    logger, _ := zap.NewProduction()
    defer logger.Sync()

    // Structured fields
    logger.Info("user_login",
        zap.String("user_email", "john@example.com"),
        zap.String("source_ip", "192.168.1.100"),
        zap.String("request_id", "req-abc123"),
    )

    // With context (child logger)
    reqLogger := logger.With(
        zap.String("request_id", "req-abc123"),
        zap.String("service", "auth-service"),
    )
    reqLogger.Info("processing_request")
    reqLogger.Info("request_complete", zap.Int("duration_ms", 45))
}
```

### 2.4 Recommended Standard Fields

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | ISO 8601 | Event time in UTC |
| `level` | string | Log level (INFO, ERROR, etc.) |
| `message` | string | Human-readable event description |
| `service` | string | Service name |
| `environment` | string | prod, staging, dev |
| `request_id` | string | Unique request identifier |
| `trace_id` | string | Distributed trace ID (for correlation with traces) |
| `user_id` | string | Authenticated user (if applicable) |
| `duration_ms` | number | Operation duration |
| `error` | string | Error message (for ERROR level) |
| `stack_trace` | string | Stack trace (for ERROR level) |

---

## 3. ELK Stack (Elasticsearch, Logstash, Kibana)

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       ELK Stack Architecture                     │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ App Logs │──→│ Filebeat │──→│  Kafka /  │──→│ Logstash │    │
│  │ (stdout) │   │ (shipper)│   │  Redis    │   │(transform│    │
│  └──────────┘   └──────────┘   │ (buffer)  │   │  & enrich│    │
│                                └──────────┘   └─────┬────┘    │
│  ┌──────────┐   ┌──────────┐                        │          │
│  │Container │──→│Fluentd / │────────────────────────┘          │
│  │  Logs    │   │Fluent Bit│                        │          │
│  └──────────┘   └──────────┘                        ▼          │
│                                              ┌──────────┐      │
│                                              │Elastic-  │      │
│                                              │search    │      │
│                                              │(store &  │      │
│                                              │ index)   │      │
│                                              └─────┬────┘      │
│                                                    │           │
│                                                    ▼           │
│                                              ┌──────────┐      │
│                                              │  Kibana  │      │
│                                              │(visualize│      │
│                                              │& search) │      │
│                                              └──────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Responsibilities

| Component | Role | Key Features |
|-----------|------|-------------|
| **Beats (Filebeat)** | Lightweight log shipper | Tail log files, forward to Logstash/Elasticsearch, backpressure handling |
| **Logstash** | Log processing pipeline | Parse, transform, enrich, filter logs; plugin ecosystem |
| **Elasticsearch** | Search and analytics engine | Full-text search, inverted index, distributed, near-real-time |
| **Kibana** | Visualization and exploration | Dashboards, log exploration (Discover), alerting |

### 3.3 Filebeat Configuration

```yaml
# filebeat.yml
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /var/log/app/*.log
    json.keys_under_root: true
    json.add_error_key: true
    fields:
      service: "webapp"
      environment: "production"
    fields_under_root: true

  - type: container
    paths:
      - /var/lib/docker/containers/*/*.log
    processors:
      - add_kubernetes_metadata: ~

output.logstash:
  hosts: ["logstash:5044"]
  loadbalance: true

# Or send directly to Elasticsearch (skip Logstash)
# output.elasticsearch:
#   hosts: ["elasticsearch:9200"]
#   index: "app-logs-%{+yyyy.MM.dd}"
```

### 3.4 Logstash Pipeline

```ruby
# logstash/pipeline/main.conf

input {
  beats {
    port => 5044
  }
}

filter {
  # Parse JSON logs
  if [message] =~ /^\{/ {
    json {
      source => "message"
    }
  }

  # Parse non-JSON logs with grok
  if ![level] {
    grok {
      match => {
        "message" => "%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} %{GREEDYDATA:log_message}"
      }
    }
  }

  # Parse timestamp
  date {
    match => ["timestamp", "ISO8601", "yyyy-MM-dd HH:mm:ss"]
    target => "@timestamp"
  }

  # GeoIP lookup for IP addresses
  if [source_ip] {
    geoip {
      source => "source_ip"
      target => "geoip"
    }
  }

  # Remove sensitive fields
  mutate {
    remove_field => ["password", "token", "authorization"]
  }

  # Add derived fields
  if [level] == "ERROR" or [level] == "CRITICAL" {
    mutate {
      add_tag => ["error"]
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "app-logs-%{+yyyy.MM.dd}"
    # ILM (Index Lifecycle Management) policy
    ilm_rollover_alias => "app-logs"
    ilm_pattern => "{now/d}-000001"
    ilm_policy => "app-logs-policy"
  }

  # Also send errors to a dedicated index
  if "error" in [tags] {
    elasticsearch {
      hosts => ["elasticsearch:9200"]
      index => "error-logs-%{+yyyy.MM.dd}"
    }
  }
}
```

### 3.5 Elasticsearch Index Lifecycle Management (ILM)

```json
PUT _ilm/policy/app-logs-policy
{
  "policy": {
    "phases": {
      "hot": {
        "min_age": "0ms",
        "actions": {
          "rollover": {
            "max_primary_shard_size": "50gb",
            "max_age": "1d"
          },
          "set_priority": { "priority": 100 }
        }
      },
      "warm": {
        "min_age": "7d",
        "actions": {
          "shrink": { "number_of_shards": 1 },
          "forcemerge": { "max_num_segments": 1 },
          "set_priority": { "priority": 50 }
        }
      },
      "cold": {
        "min_age": "30d",
        "actions": {
          "searchable_snapshot": {
            "snapshot_repository": "my-repo"
          },
          "set_priority": { "priority": 0 }
        }
      },
      "delete": {
        "min_age": "90d",
        "actions": {
          "delete": {}
        }
      }
    }
  }
}
```

**ILM Phase Lifecycle:**

```
Hot (0-7d)          Warm (7-30d)        Cold (30-90d)       Delete (90d+)
├─ SSD storage      ├─ HDD storage      ├─ Snapshot storage  ├─ Removed
├─ Full replicas    ├─ Shrink shards    ├─ Frozen index      └─ Free space
├─ Write + read     ├─ Force merge      └─ Read-only
└─ High priority    └─ Medium priority
```

---

## 4. Loki/Grafana Stack

### 4.1 Architecture Overview

Loki is a log aggregation system designed to be cost-effective and easy to operate. Unlike Elasticsearch, Loki does **not** index log contents -- it only indexes labels (metadata). Log lines are stored compressed in object storage.

```
┌─────────────────────────────────────────────────────────────────┐
│                      Loki Stack Architecture                     │
│                                                                  │
│  ┌──────────┐                                                   │
│  │ App Logs │──→ Promtail ──→ ┌──────────┐   ┌──────────┐     │
│  │ (stdout) │    (agent)      │   Loki   │──→│  Grafana │     │
│  └──────────┘                 │ (store)  │   │(visualize│     │
│                               │          │   │& explore)│     │
│  ┌──────────┐                 │ Labels   │   └──────────┘     │
│  │Container │──→ Promtail ──→ │ indexed  │                     │
│  │  Logs    │    (agent)      │ Content  │   Object Storage    │
│  └──────────┘                 │ NOT      │──→ (S3, GCS,       │
│                               │ indexed  │    MinIO)           │
│                               └──────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Loki vs Elasticsearch

| Aspect | Loki | Elasticsearch |
|--------|------|---------------|
| **Indexing** | Labels only (like Prometheus) | Full-text inverted index |
| **Storage cost** | Low (compressed chunks in object storage) | High (full index on disk) |
| **Query speed** | Fast for label-filtered queries; slower for grep-like searches | Fast for any full-text search |
| **Operational complexity** | Low (stateless components, object storage) | High (cluster management, JVM tuning) |
| **Integration** | Native Grafana | Kibana |
| **Best for** | Teams already using Prometheus/Grafana, cost-sensitive | Complex search needs, compliance, large-scale analytics |

### 4.3 Promtail Configuration

```yaml
# promtail-config.yml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  # Static file scraping
  - job_name: app-logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: webapp
          environment: production
          __path__: /var/log/app/*.log

  # Kubernetes pod log discovery
  - job_name: kubernetes-pods
    kubernetes_sd_configs:
      - role: pod
    pipeline_stages:
      # Parse JSON logs
      - json:
          expressions:
            level: level
            message: message
            trace_id: trace_id
      # Set log level as a label
      - labels:
          level:
      # Drop debug logs in production
      - match:
          selector: '{level="DEBUG"}'
          action: drop
      # Extract metrics from logs
      - metrics:
          log_lines_total:
            type: Counter
            description: "Total log lines"
            source: level
            config:
              action: inc
```

### 4.4 LogQL (Loki Query Language)

```logql
# Basic label filtering
{job="webapp", environment="production"}

# Log line filter (contains)
{job="webapp"} |= "error"

# Log line filter (does not contain)
{job="webapp"} != "health_check"

# Regex filter
{job="webapp"} |~ "status=(4|5)\\d{2}"

# JSON parsing and field extraction
{job="webapp"} | json | level="ERROR"

# JSON parsing with specific field filter
{job="webapp"} | json | duration_ms > 1000

# Log line formatting
{job="webapp"} | json | line_format "{{.timestamp}} [{{.level}}] {{.message}}"

# Aggregation: count error logs per minute
count_over_time({job="webapp"} |= "error" [1m])

# Aggregation: rate of errors by service
sum by (service) (rate({environment="production"} |= "error" [5m]))

# Aggregation: p95 latency from log fields
quantile_over_time(0.95, {job="webapp"} | json | unwrap duration_ms [5m]) by (endpoint)
```

---

## 5. Log Aggregation Patterns

### 5.1 Sidecar Pattern (Kubernetes)

```yaml
# kubernetes deployment with logging sidecar
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp
spec:
  template:
    spec:
      containers:
        # Application container
        - name: webapp
          image: webapp:latest
          volumeMounts:
            - name: log-volume
              mountPath: /var/log/app

        # Logging sidecar
        - name: promtail
          image: grafana/promtail:latest
          args:
            - -config.file=/etc/promtail/config.yml
          volumeMounts:
            - name: log-volume
              mountPath: /var/log/app
              readOnly: true
            - name: promtail-config
              mountPath: /etc/promtail

      volumes:
        - name: log-volume
          emptyDir: {}
        - name: promtail-config
          configMap:
            name: promtail-config
```

### 5.2 DaemonSet Pattern (Kubernetes)

```yaml
# Node-level log collector (preferred for Kubernetes)
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: promtail
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: promtail
  template:
    spec:
      containers:
        - name: promtail
          image: grafana/promtail:latest
          volumeMounts:
            - name: varlog
              mountPath: /var/log
              readOnly: true
            - name: containers
              mountPath: /var/lib/docker/containers
              readOnly: true
      volumes:
        - name: varlog
          hostPath:
            path: /var/log
        - name: containers
          hostPath:
            path: /var/lib/docker/containers
```

### 5.3 Pattern Comparison

| Pattern | How It Works | Pros | Cons |
|---------|-------------|------|------|
| **Sidecar** | Agent per pod | Fine-grained config per app, isolated | Higher resource overhead, more containers |
| **DaemonSet** | Agent per node | Lower overhead, centralized config | Shared config for all pods on node |
| **Direct push** | App sends logs via SDK | No agent needed, immediate | Coupling, retry logic in app |

---

## 6. Log Retention Policies

### 6.1 Designing Retention Policies

| Log Type | Hot (searchable) | Warm (archived) | Total Retention | Rationale |
|----------|-----------------|-----------------|-----------------|-----------|
| **Application logs** | 7-14 days | 30-90 days | 90 days | Most incidents investigated within days |
| **Access logs** | 7 days | 90-365 days | 365 days | Compliance (PCI-DSS, SOC 2) |
| **Audit logs** | 30 days | 1-7 years | 7 years | Regulatory (HIPAA, SOX) |
| **Security logs** | 30 days | 1-3 years | 3 years | Forensics, compliance |
| **Debug logs** | 1-3 days | None | 3 days | High volume, low long-term value |

### 6.2 Loki Retention Configuration

```yaml
# loki-config.yml
limits_config:
  retention_period: 744h    # 31 days default

compactor:
  working_directory: /tmp/loki/compactor
  shared_store: s3
  compaction_interval: 10m
  retention_enabled: true
  retention_delete_delay: 2h
  retention_delete_worker_count: 150

# Per-tenant retention (multi-tenant)
overrides:
  tenant-a:
    retention_period: 2160h   # 90 days
  tenant-b:
    retention_period: 720h    # 30 days
```

### 6.3 Cost Optimization Strategies

1. **Drop unnecessary logs early** -- Filter out health checks, debug logs in production
2. **Use sampling for high-volume logs** -- Log 10% of successful requests, 100% of errors
3. **Compress aggressively** -- Loki uses snappy/gzip; Elasticsearch force-merges warm indices
4. **Tier storage** -- Hot (SSD) for recent, Cold (S3/GCS) for archive
5. **Set index granularity** -- Daily indices for high volume, weekly for low volume

---

## 7. Practical Docker Compose Setup

### 7.1 ELK Stack

```yaml
# docker-compose-elk.yml
version: "3.8"
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.12.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
    ports:
      - "9200:9200"
    volumes:
      - es-data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:8.12.0
    ports:
      - "5044:5044"
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.12.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

  filebeat:
    image: docker.elastic.co/beats/filebeat:8.12.0
    volumes:
      - ./filebeat.yml:/usr/share/filebeat/filebeat.yml:ro
      - /var/log:/var/log:ro
    depends_on:
      - logstash

volumes:
  es-data:
```

### 7.2 Loki Stack

```yaml
# docker-compose-loki.yml
version: "3.8"
services:
  loki:
    image: grafana/loki:2.9.4
    ports:
      - "3100:3100"
    volumes:
      - ./loki-config.yml:/etc/loki/config.yml
      - loki-data:/loki
    command: -config.file=/etc/loki/config.yml

  promtail:
    image: grafana/promtail:2.9.4
    volumes:
      - ./promtail-config.yml:/etc/promtail/config.yml
      - /var/log:/var/log:ro
    command: -config.file=/etc/promtail/config.yml
    depends_on:
      - loki

  grafana:
    image: grafana/grafana:10.3.0
    ports:
      - "3000:3000"
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Admin
    volumes:
      - grafana-data:/var/lib/grafana
    depends_on:
      - loki

volumes:
  loki-data:
  grafana-data:
```

---

## 8. Log-Based Alerting

### 8.1 Kibana Alerting Rules

In Kibana (Stack Management > Rules), create rules that trigger on log patterns:

```
Rule: High Error Rate in Logs
Condition: count of documents where level="ERROR"
           over the last 5 minutes > 50
Action: Send to Slack channel #alerts-backend
```

### 8.2 Loki Alerting Rules (via Ruler)

```yaml
# loki-rules.yml
groups:
  - name: log_alerts
    rules:
      # Alert on high error log rate
      - alert: HighErrorLogRate
        expr: |
          sum(rate({environment="production"} |= "ERROR" [5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error log rate in production"
          description: "More than 10 error logs per second over 5 minutes."

      # Alert on specific critical error pattern
      - alert: DatabaseConnectionFailure
        expr: |
          count_over_time({job="webapp"} |= "database connection refused" [5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failures detected"

      # Alert on panic/crash patterns
      - alert: ApplicationPanic
        expr: |
          count_over_time({environment="production"} |~ "panic|FATAL|segfault" [1m]) > 0
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "Application panic/crash detected"
```

---

## 9. Next Steps

- [12_Distributed_Tracing.md](./12_Distributed_Tracing.md) - Request tracing with OpenTelemetry and Jaeger
- [10_Monitoring_and_Alerting.md](./10_Monitoring_and_Alerting.md) - Metrics monitoring with Prometheus

---

## Exercises

### Exercise 1: Structured Logging Design

Your team is building a payment processing service. Design a structured log schema for the following events:

1. Payment request received
2. Payment authorized by payment gateway
3. Payment failed (declined by bank)

For each event, specify the JSON fields, their types, and which fields should be indexed as labels in Loki.

<details>
<summary>Show Answer</summary>

**1. Payment request received:**
```json
{
  "timestamp": "2024-03-15T14:23:45.123Z",
  "level": "INFO",
  "message": "Payment request received",
  "service": "payment-service",
  "request_id": "req-abc123",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "order_id": "ord-789",
  "amount": 99.99,
  "currency": "USD",
  "payment_method": "credit_card",
  "card_last_four": "4242",
  "user_id": "user-456"
}
```

**2. Payment authorized:**
```json
{
  "timestamp": "2024-03-15T14:23:45.890Z",
  "level": "INFO",
  "message": "Payment authorized",
  "service": "payment-service",
  "request_id": "req-abc123",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "order_id": "ord-789",
  "amount": 99.99,
  "currency": "USD",
  "gateway": "stripe",
  "gateway_transaction_id": "ch_3abc123",
  "authorization_code": "AUTH123",
  "duration_ms": 767
}
```

**3. Payment failed:**
```json
{
  "timestamp": "2024-03-15T14:23:46.100Z",
  "level": "ERROR",
  "message": "Payment declined by issuing bank",
  "service": "payment-service",
  "request_id": "req-abc123",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "order_id": "ord-789",
  "amount": 99.99,
  "currency": "USD",
  "gateway": "stripe",
  "decline_code": "insufficient_funds",
  "decline_message": "Your card has insufficient funds",
  "duration_ms": 523,
  "user_id": "user-456"
}
```

**Loki labels** (indexed, low cardinality): `service`, `level`, `environment`, `gateway`

**Not as labels** (high cardinality -- would explode index): `request_id`, `order_id`, `user_id`, `trace_id`

The key principle is that Loki labels should have low cardinality (few distinct values). High-cardinality fields are stored in the log line and searched with `| json | order_id="ord-789"`.

</details>

### Exercise 2: ELK vs Loki Decision

Your company runs 50 microservices generating 500 GB of logs per day. The team currently uses Elasticsearch but is evaluating Loki. Analyze the trade-offs and recommend a solution for each of the following use cases:

1. Application debugging during incidents
2. Security audit log compliance (7-year retention)
3. Real-time error pattern detection and alerting

<details>
<summary>Show Answer</summary>

1. **Application debugging during incidents -- Loki is sufficient**
   - During incidents, engineers typically know the service, time window, and error type. They filter by labels (`{service="payment-service", level="ERROR"}`) and then grep through log lines.
   - Loki handles this well because label filtering narrows the search, and the log content search operates on a small subset.
   - Elasticsearch is faster for ad-hoc full-text searches across all services, but this is rarely needed during incidents.

2. **Security audit log compliance (7-year retention) -- Loki is better**
   - At 500 GB/day, 7-year retention in Elasticsearch would require petabytes of indexed storage -- extremely expensive.
   - Loki stores log lines as compressed chunks in object storage (S3/GCS), which is 5-10x cheaper per GB than Elasticsearch.
   - Audit logs are rarely queried (only during investigations), so the slower grep-like search in Loki is acceptable.
   - Cost estimate: S3 at ~$0.023/GB/month vs Elasticsearch EBS at ~$0.10/GB/month.

3. **Real-time error pattern detection and alerting -- Both work**
   - Loki Ruler can alert on `count_over_time({level="ERROR"} |= "pattern" [5m]) > threshold`.
   - Elasticsearch has Watcher/Kibana Alerts that can detect patterns in real-time.
   - For simple pattern matching, Loki is sufficient and lower-cost. For complex correlations (e.g., "alert when error in service A is followed by error in service B within 30 seconds"), Elasticsearch's query capabilities are stronger.

**Recommendation for 500 GB/day**: Use a **hybrid approach**. Send all logs to Loki for cost-effective storage and basic querying. Send security and audit logs to both Loki (long-term) and a smaller Elasticsearch cluster (30-day window) for compliance search requirements that need full-text indexing.

</details>

### Exercise 3: LogQL Query Writing

Write LogQL queries for the following scenarios:

1. Find all ERROR-level logs from the `payment-service` in the last hour.
2. Count the number of logs per minute from each service in production.
3. Find requests that took longer than 2 seconds (assuming structured JSON logs with a `duration_ms` field).
4. Calculate the 95th percentile request duration from log data grouped by endpoint.

<details>
<summary>Show Answer</summary>

1. Error logs from payment-service:
```logql
{job="payment-service", level="ERROR"}
```
(Time range is set in the Grafana time picker to "Last 1 hour")

2. Log count per minute by service:
```logql
sum by (service) (count_over_time({environment="production"}[1m]))
```

3. Slow requests (> 2 seconds):
```logql
{job="webapp"} | json | duration_ms > 2000
```

4. P95 request duration by endpoint:
```logql
quantile_over_time(0.95,
  {job="webapp"} | json | unwrap duration_ms [5m]
) by (endpoint)
```

Note: `unwrap` extracts a numeric field from the log line for aggregation. Without `unwrap`, Loki treats all log content as strings and cannot perform numeric operations.

</details>

### Exercise 4: Retention Policy Design

Design a log retention policy for a healthcare SaaS application subject to HIPAA regulations. Specify the retention period, storage tier, and access controls for each log category.

<details>
<summary>Show Answer</summary>

| Log Category | Hot (Fast Search) | Warm (Archived) | Cold (Compliance) | Total Retention | Access Control |
|-------------|------------------|-----------------|-------------------|-----------------|---------------|
| **Application logs** | 14 days (SSD) | 76 days (HDD) | None | 90 days | Dev team, on-call |
| **Access logs (PHI)** | 30 days (SSD) | 335 days (HDD) | 5 years (S3 Glacier) | 6 years | Security team only |
| **Audit trail** | 30 days (SSD) | 335 days (HDD) | 6 years (S3 Glacier) | 7 years | Compliance team, read-only |
| **Authentication logs** | 30 days (SSD) | 335 days (HDD) | 2 years (S3) | 3 years | Security team |
| **Debug logs** | 3 days (SSD) | None | None | 3 days | Dev team |
| **Infrastructure logs** | 7 days (SSD) | 23 days (HDD) | None | 30 days | Ops team |

**HIPAA-specific requirements:**
- All logs containing PHI (Protected Health Information) must be encrypted at rest (AES-256) and in transit (TLS 1.2+).
- Access to PHI logs must be logged in the audit trail (log access logging).
- Audit trail must be immutable (write-once storage, e.g., S3 Object Lock).
- Log deletion must follow a documented retention schedule and be auditable.

**Implementation in Elasticsearch ILM:**
- Hot → Warm: `min_age: 30d`, shrink to 1 shard, force merge
- Warm → Cold: `min_age: 365d`, searchable snapshot to S3
- Cold → Delete: `min_age: 2555d` (7 years) for audit, `min_age: 90d` for app logs

</details>

---

## References

- [Elasticsearch Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/)
- [Logstash Documentation](https://www.elastic.co/guide/en/logstash/current/)
- [Kibana Documentation](https://www.elastic.co/guide/en/kibana/current/)
- [Grafana Loki Documentation](https://grafana.com/docs/loki/latest/)
- [LogQL Documentation](https://grafana.com/docs/loki/latest/logql/)
- [structlog (Python)](https://www.structlog.org/)
- [Zap Logger (Go)](https://pkg.go.dev/go.uber.org/zap)

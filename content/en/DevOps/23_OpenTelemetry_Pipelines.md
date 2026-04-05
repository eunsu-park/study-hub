# 23. OpenTelemetry Pipelines

**Previous**: [Advanced Metrics Architecture](./22_Advanced_Metrics_Architecture.md) | **Next**: [eBPF Observability](./24_eBPF_Observability.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the OpenTelemetry Collector architecture and its receiver-processor-exporter pipeline model
2. Configure production-grade Collector pipelines with batching, memory limiting, and retry logic
3. Implement tail-based sampling strategies that retain high-value traces while controlling costs
4. Design multi-tier Collector deployments using agent and gateway patterns
5. Build custom processors for attribute manipulation, filtering, and routing
6. Monitor Collector health and troubleshoot pipeline bottlenecks

---

The OpenTelemetry Collector is the central nervous system of a modern observability stack. It receives telemetry data from applications, processes it (filter, transform, sample, enrich), and exports it to one or more backends. A well-designed Collector pipeline determines the quality, cost, and reliability of your entire observability platform.

> **Analogy -- Water Treatment Plant**: Raw water (telemetry) enters the plant through intake pipes (receivers). It passes through treatment stages -- filtration removes debris (filtering), chlorination kills bacteria (sampling removes noise), and fluoridation adds beneficial minerals (attribute enrichment). Finally, clean water is distributed through output pipes (exporters) to homes (backends). Without the treatment plant, you would either get no water or contaminated water.

## 1. Collector Architecture

### 1.1 Pipeline Model

```
┌─────────────────────────────────────────────────────────┐
│                  OTel Collector                           │
│                                                          │
│  ┌───────────┐    ┌────────────┐    ┌───────────────┐   │
│  │ Receivers  │───→│ Processors │───→│  Exporters    │   │
│  │            │    │            │    │               │   │
│  │ - OTLP     │    │ - batch    │    │ - OTLP        │   │
│  │ - Jaeger   │    │ - filter   │    │ - Prometheus  │   │
│  │ - Prometheus│   │ - attributes│   │ - Loki        │   │
│  │ - Kafka    │    │ - sampling │    │ - Kafka       │   │
│  │ - Filelog  │    │ - transform│    │ - Debug       │   │
│  └───────────┘    └────────────┘    └───────────────┘   │
│                                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Connectors                            │  │
│  │  (connect pipelines: traces→metrics, etc.)        │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────┐  ┌────────────────────────┐     │
│  │ Extensions          │  │ Service                │     │
│  │ - health_check      │  │ - telemetry            │     │
│  │ - pprof             │  │ - pipelines            │     │
│  │ - zpages            │  │                        │     │
│  └────────────────────┘  └────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Collector Distributions

| Distribution | Contents | Use Case |
|-------------|----------|----------|
| **Core** (`otelcol`) | Minimal receivers/exporters (OTLP only) | Simple deployments |
| **Contrib** (`otelcol-contrib`) | 100+ community components | Most production deployments |
| **Custom** (OCB builder) | Only the components you need | Minimal attack surface, reduced binary size |

### 1.3 Building a Custom Collector

```yaml
# builder-config.yaml -- OpenTelemetry Collector Builder (OCB)
dist:
  name: my-otelcol
  description: Custom collector for production
  output_path: ./build
  version: 0.96.0

receivers:
  - gomod: go.opentelemetry.io/collector/receiver/otlpreceiver v0.96.0
  - gomod: github.com/open-telemetry/opentelemetry-collector-contrib/receiver/prometheusreceiver v0.96.0
  - gomod: github.com/open-telemetry/opentelemetry-collector-contrib/receiver/filelogreceiver v0.96.0

processors:
  - gomod: go.opentelemetry.io/collector/processor/batchprocessor v0.96.0
  - gomod: go.opentelemetry.io/collector/processor/memorylimiterprocessor v0.96.0
  - gomod: github.com/open-telemetry/opentelemetry-collector-contrib/processor/filterprocessor v0.96.0
  - gomod: github.com/open-telemetry/opentelemetry-collector-contrib/processor/attributesprocessor v0.96.0
  - gomod: github.com/open-telemetry/opentelemetry-collector-contrib/processor/tailsamplingprocessor v0.96.0

exporters:
  - gomod: go.opentelemetry.io/collector/exporter/otlpexporter v0.96.0
  - gomod: go.opentelemetry.io/collector/exporter/otlphttpexporter v0.96.0
  - gomod: github.com/open-telemetry/opentelemetry-collector-contrib/exporter/prometheusexporter v0.96.0

connectors:
  - gomod: github.com/open-telemetry/opentelemetry-collector-contrib/connector/spanmetricsconnector v0.96.0

extensions:
  - gomod: go.opentelemetry.io/collector/extension/zpagesextension v0.96.0
  - gomod: github.com/open-telemetry/opentelemetry-collector-contrib/extension/healthcheckextension v0.96.0
```

```bash
# Build the custom collector
ocb --config builder-config.yaml
./build/my-otelcol --config collector-config.yaml
```

---

## 2. Receivers

### 2.1 OTLP Receiver

The primary receiver for OpenTelemetry-native applications:

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
        max_recv_msg_size_mib: 4        # Max message size
        max_concurrent_streams: 100      # gRPC stream limit
        keepalive:
          server_parameters:
            max_connection_idle: 11s
            max_connection_age: 30s
        tls:                             # Enable TLS for production
          cert_file: /certs/server.crt
          key_file: /certs/server.key

      http:
        endpoint: 0.0.0.0:4318
        cors:
          allowed_origins: ["https://app.example.com"]
          allowed_headers: ["Content-Type"]
```

### 2.2 Prometheus Receiver

Scrape Prometheus-format metrics from targets:

```yaml
receivers:
  prometheus:
    config:
      scrape_configs:
        - job_name: "kubernetes-pods"
          kubernetes_sd_configs:
            - role: pod
          relabel_configs:
            - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
              action: keep
              regex: true
            - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
              action: replace
              target_label: __address__
              regex: (.+)
              replacement: ${1}:${2}
```

### 2.3 Filelog Receiver

Collect logs from files (replaces Promtail/Fluentd for some use cases):

```yaml
receivers:
  filelog:
    include:
      - /var/log/pods/*/*/*.log
    exclude:
      - /var/log/pods/*/otel-collector/*.log    # Avoid self-collection
    start_at: end                                # Start from end of file
    include_file_path: true
    include_file_name: true
    operators:
      # Parse JSON logs
      - type: json_parser
        timestamp:
          parse_from: attributes.timestamp
          layout: "%Y-%m-%dT%H:%M:%S.%LZ"
        severity:
          parse_from: attributes.level

      # Extract trace_id for correlation
      - type: move
        from: attributes.trace_id
        to: attributes["trace_id"]

      # Add Kubernetes metadata
      - type: regex_parser
        regex: '^/var/log/pods/(?P<namespace>[^_]+)_(?P<pod>[^_]+)_'
        parse_from: attributes["log.file.path"]
```

---

## 3. Processors

### 3.1 Essential Processors

**Memory Limiter** (always first in the pipeline):

```yaml
processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: 512             # Hard limit
    spike_limit_mib: 128       # Soft limit (triggers GC)
    # When limit is hit: incoming data is rejected (backpressure)
```

**Batch Processor** (always last before exporters):

```yaml
processors:
  batch:
    send_batch_size: 1024       # Batch size in spans/metrics/logs
    send_batch_max_size: 2048   # Max batch size (prevents oversized batches)
    timeout: 5s                  # Max wait before sending incomplete batch
```

### 3.2 Filter Processor

```yaml
processors:
  filter/traces:
    error_mode: ignore
    traces:
      span:
        # Drop health check traces
        - 'attributes["http.route"] == "/healthz"'
        - 'attributes["http.route"] == "/readyz"'
        # Drop internal service mesh traces
        - 'attributes["http.user_agent"] == "kube-probe/1.28"'

  filter/metrics:
    error_mode: ignore
    metrics:
      metric:
        # Drop Go runtime metrics (high cardinality, rarely useful)
        - 'name == "go_gc_duration_seconds"'
        - 'name == "go_goroutines"'
        - 'name == "go_memstats_alloc_bytes"'
        - 'HasAttrKeyOnDatapoint("user_id")'  # Drop metrics with user_id label

  filter/logs:
    error_mode: ignore
    logs:
      log_record:
        # Drop DEBUG logs in production
        - 'severity_number < SEVERITY_NUMBER_INFO'
        # Drop noisy health check logs
        - 'body == "GET /healthz 200"'
```

### 3.3 Attributes Processor

```yaml
processors:
  attributes/insert:
    actions:
      # Add environment and deployment info
      - key: deployment.environment
        value: "production"
        action: insert
      - key: deployment.region
        from_context: metadata.region
        action: insert

  attributes/delete:
    actions:
      # Remove sensitive data
      - key: http.request.header.authorization
        action: delete
      - key: db.statement
        action: delete    # Remove SQL queries (may contain PII)
      - key: user.email
        action: delete    # Remove PII

  attributes/hash:
    actions:
      # Hash sensitive values instead of deleting
      - key: user.id
        action: hash      # SHA-256 hash preserves cardinality without exposing value
```

### 3.4 Transform Processor

For complex transformations using the OTTL (OpenTelemetry Transformation Language):

```yaml
processors:
  transform/traces:
    error_mode: ignore
    trace_statements:
      - context: span
        statements:
          # Truncate long attribute values
          - truncate_all(attributes, 256)
          # Normalize HTTP routes (remove path parameters)
          - replace_pattern(attributes["http.route"], "/users/[0-9]+", "/users/:id")
          - replace_pattern(attributes["http.route"], "/orders/[a-f0-9-]+", "/orders/:id")
          # Set span status based on HTTP status code
          - set(status.code, STATUS_CODE_ERROR) where attributes["http.status_code"] >= 500

  transform/logs:
    error_mode: ignore
    log_statements:
      - context: log
        statements:
          # Mask credit card numbers
          - replace_pattern(body, "\\b\\d{4}[- ]?\\d{4}[- ]?\\d{4}[- ]?\\d{4}\\b", "****-****-****-****")
          # Mask email addresses
          - replace_pattern(body, "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}", "***@***.***")
```

---

## 4. Tail-Based Sampling

### 4.1 Head Sampling vs Tail Sampling

| Aspect | Head Sampling | Tail Sampling |
|--------|-------------|--------------|
| **Decision point** | At trace start (before processing) | After trace completes |
| **Available info** | None (random decision) | Full trace: duration, status, attributes |
| **Resource cost** | Very low | High (must buffer complete traces) |
| **Implementation** | SDK-level `TraceIdRatioBased` sampler | Collector-level `tailsampling` processor |
| **Quality** | Statistically representative but misses rare events | Keeps interesting traces, discards boring ones |

### 4.2 Tail Sampling Configuration

```yaml
processors:
  tail_sampling:
    decision_wait: 30s          # Wait for trace to complete
    num_traces: 100000          # Max traces in memory
    expected_new_traces_per_sec: 1000

    policies:
      # Policy 1: Always keep error traces
      - name: errors
        type: status_code
        status_code:
          status_codes: [ERROR]

      # Policy 2: Always keep slow traces (> 2s)
      - name: slow-traces
        type: latency
        latency:
          threshold_ms: 2000

      # Policy 3: Always keep traces from critical services
      - name: critical-services
        type: string_attribute
        string_attribute:
          key: service.name
          values:
            - payment-service
            - auth-service
            - order-service

      # Policy 4: Sample 5% of successful traces
      - name: probabilistic-sample
        type: probabilistic
        probabilistic:
          sampling_percentage: 5

      # Policy 5: Always keep traces with specific flags
      - name: debug-flag
        type: string_attribute
        string_attribute:
          key: debug
          values: ["true"]

      # Policy 6: Rate-limited sampling for high-volume services
      - name: rate-limited
        type: rate_limiting
        rate_limiting:
          spans_per_second: 100

      # Composite: combine policies with AND/OR logic
      - name: composite-policy
        type: and
        and:
          and_sub_policy:
            - name: is-health-check
              type: string_attribute
              string_attribute:
                key: http.route
                values: ["/healthz", "/readyz"]
            - name: drop-all
              type: probabilistic
              probabilistic:
                sampling_percentage: 0
```

### 4.3 Tail Sampling Best Practices

| Practice | Reason |
|----------|--------|
| Place tail sampling in a **gateway** Collector, not agents | Needs complete traces (all spans from all services) |
| Set `decision_wait` to at least the max expected trace duration | Prevents premature decisions on incomplete traces |
| Monitor `otelcol_processor_tail_sampling_count_traces_sampled` | Track sampling effectiveness |
| Use `num_traces` based on memory budget | Each buffered trace uses ~1-10 KB |
| Combine with head sampling at the SDK level | Reduce volume before it reaches the Collector |

---

## 5. Multi-Tier Deployment

### 5.1 Agent + Gateway Architecture

```
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│  Pod 1  │  │  Pod 2  │  │  Pod 3  │  │  Pod N  │
│  + App  │  │  + App  │  │  + App  │  │  + App  │
└────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
     │            │            │            │
     └────────┬───┴──────┬─────┘            │
              │          │                  │
     ┌────────▼──┐  ┌────▼──────┐  ┌───────▼───┐
     │ Agent     │  │ Agent     │  │ Agent     │  ← DaemonSet
     │ (Node 1)  │  │ (Node 2)  │  │ (Node 3)  │    (one per node)
     │ - batch   │  │ - batch   │  │ - batch   │
     │ - filter  │  │ - filter  │  │ - filter  │
     │ - memory  │  │ - memory  │  │ - memory  │
     │   limiter │  │   limiter │  │   limiter │
     └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
           │               │               │
           └───────────────┼───────────────┘
                           │ OTLP
                    ┌──────▼──────┐
                    │   Gateway    │ ← Deployment (2+ replicas)
                    │  - tail     │
                    │    sampling │
                    │  - routing  │
                    │  - transform│
                    └──┬──────┬──┘
                       │      │
              ┌────────▼┐  ┌──▼────────┐
              │  Tempo  │  │Prometheus │
              └─────────┘  └───────────┘
```

### 5.2 Agent Configuration (DaemonSet)

```yaml
# otel-agent-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: 256
    spike_limit_mib: 64

  batch:
    send_batch_size: 512
    timeout: 5s

  # Agent-level filtering (reduces network traffic)
  filter/drop-health:
    error_mode: ignore
    traces:
      span:
        - 'attributes["http.route"] == "/healthz"'
    logs:
      log_record:
        - 'severity_number < SEVERITY_NUMBER_INFO'

  resource/add-node:
    attributes:
      - key: k8s.node.name
        from_attribute: HOSTNAME
        action: insert

exporters:
  otlp/gateway:
    endpoint: otel-gateway.observability.svc:4317
    tls:
      insecure: true
    retry_on_failure:
      enabled: true
      initial_interval: 5s
      max_interval: 30s
      max_elapsed_time: 300s
    sending_queue:
      enabled: true
      num_consumers: 10
      queue_size: 5000

extensions:
  health_check:
    endpoint: 0.0.0.0:13133

service:
  extensions: [health_check]
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, resource/add-node, filter/drop-health, batch]
      exporters: [otlp/gateway]
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, resource/add-node, batch]
      exporters: [otlp/gateway]
    logs:
      receivers: [otlp]
      processors: [memory_limiter, resource/add-node, filter/drop-health, batch]
      exporters: [otlp/gateway]
```

### 5.3 Gateway Configuration (Deployment)

```yaml
# otel-gateway-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: 1024
    spike_limit_mib: 256

  batch:
    send_batch_size: 2048
    timeout: 10s

  tail_sampling:
    decision_wait: 30s
    num_traces: 200000
    policies:
      - name: errors
        type: status_code
        status_code: {status_codes: [ERROR]}
      - name: slow
        type: latency
        latency: {threshold_ms: 2000}
      - name: sample-rest
        type: probabilistic
        probabilistic: {sampling_percentage: 10}

  transform/pii:
    error_mode: ignore
    trace_statements:
      - context: span
        statements:
          - replace_pattern(attributes["db.statement"], "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\\b", "***@***.***")

connectors:
  spanmetrics:
    histogram:
      explicit:
        buckets: [5ms, 10ms, 50ms, 100ms, 500ms, 1s, 5s]
    exemplars:
      enabled: true

exporters:
  otlp/tempo:
    endpoint: tempo.observability.svc:4317
    tls:
      insecure: true

  prometheus:
    endpoint: 0.0.0.0:8889
    resource_to_telemetry_conversion:
      enabled: true

  loki:
    endpoint: http://loki.observability.svc:3100/loki/api/v1/push

extensions:
  health_check:
    endpoint: 0.0.0.0:13133
  zpages:
    endpoint: 0.0.0.0:55679
  pprof:
    endpoint: 0.0.0.0:1888

service:
  extensions: [health_check, zpages, pprof]
  telemetry:
    metrics:
      level: detailed
      address: 0.0.0.0:8888
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, transform/pii, tail_sampling, batch]
      exporters: [otlp/tempo, spanmetrics]
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [prometheus]
    metrics/spanmetrics:
      receivers: [spanmetrics]
      exporters: [prometheus]
    logs:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [loki]
```

---

## 6. Pipeline Routing

### 6.1 Routing by Service or Environment

```yaml
processors:
  routing:
    from_attribute: service.name
    table:
      # Critical services → dedicated Tempo instance with higher retention
      - value: payment-service
        exporters: [otlp/tempo-critical]
      - value: auth-service
        exporters: [otlp/tempo-critical]
    default_exporters: [otlp/tempo-standard]

exporters:
  otlp/tempo-critical:
    endpoint: tempo-critical.observability.svc:4317
  otlp/tempo-standard:
    endpoint: tempo-standard.observability.svc:4317
```

### 6.2 Fan-Out to Multiple Backends

```yaml
# Send traces to both Tempo AND Jaeger (migration period)
service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [otlp/tempo, otlp/jaeger]  # Fan-out to both

exporters:
  otlp/tempo:
    endpoint: tempo:4317
  otlp/jaeger:
    endpoint: jaeger-collector:4317
```

---

## 7. Collector Monitoring

### 7.1 Key Collector Metrics

```promql
# Receiver: data received
rate(otelcol_receiver_accepted_spans[5m])          # Traces received/sec
rate(otelcol_receiver_refused_spans[5m])           # Traces rejected/sec (backpressure)

# Processor: data processed
otelcol_processor_batch_batch_send_size_sum        # Batch sizes
rate(otelcol_processor_dropped_spans[5m])          # Spans dropped by filters
otelcol_processor_tail_sampling_count_traces_sampled  # Sampling decisions

# Exporter: data exported
rate(otelcol_exporter_sent_spans[5m])              # Traces exported/sec
rate(otelcol_exporter_send_failed_spans[5m])       # Export failures/sec
otelcol_exporter_queue_size                         # Export queue depth
otelcol_exporter_queue_capacity                     # Export queue capacity

# Overall health
process_runtime_total_alloc_bytes                   # Memory usage
otelcol_process_uptime                              # Collector uptime
```

### 7.2 Collector Dashboard

```
┌─────────────────────────────────────────────────┐
│ OTel Collector Health Dashboard                  │
├──────────────┬──────────────┬───────────────────┤
│ Received     │ Processed    │ Exported          │
│ 15,230 /sec  │ 14,100 /sec  │ 14,050 /sec      │
│ (traces)     │ (after filter)│ (to backends)    │
├──────────────┴──────────────┴───────────────────┤
│ Pipeline: traces                                 │
│ Receiver → Filter → Sampling → Batch → Export    │
│   15230     14100     7050      7050    7050     │
│             drop:1130 sample:50% queue:ok        │
├─────────────────────────────────────────────────┤
│ Memory: 380/512 MB  │  CPU: 0.8 cores           │
│ Queue: 120/5000     │  Errors: 2/min            │
└─────────────────────────────────────────────────┘
```

### 7.3 Alerting on Collector Health

```yaml
# Prometheus alert rules for OTel Collector
groups:
  - name: otel_collector_alerts
    rules:
      - alert: OTelCollectorHighMemory
        expr: process_runtime_total_alloc_bytes / 1024 / 1024 > 450  # 450 MB
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "OTel Collector memory usage high"

      - alert: OTelCollectorExportFailures
        expr: rate(otelcol_exporter_send_failed_spans[5m]) > 0
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "OTel Collector failing to export spans"

      - alert: OTelCollectorQueueFull
        expr: otelcol_exporter_queue_size / otelcol_exporter_queue_capacity > 0.8
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "OTel Collector export queue nearly full"

      - alert: OTelCollectorDataLoss
        expr: rate(otelcol_receiver_refused_spans[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "OTel Collector refusing incoming spans (data loss)"
```

---

## 8. Kubernetes Deployment

### 8.1 DaemonSet Agent

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: otel-agent
  namespace: observability
spec:
  selector:
    matchLabels:
      app: otel-agent
  template:
    metadata:
      labels:
        app: otel-agent
    spec:
      serviceAccountName: otel-agent
      containers:
        - name: otel-agent
          image: otel/opentelemetry-collector-contrib:0.96.0
          args: ["--config=/etc/otel/config.yaml"]
          ports:
            - containerPort: 4317    # OTLP gRPC
            - containerPort: 4318    # OTLP HTTP
            - containerPort: 13133   # Health check
          resources:
            limits:
              cpu: 500m
              memory: 512Mi
            requests:
              cpu: 100m
              memory: 128Mi
          livenessProbe:
            httpGet:
              path: /
              port: 13133
            initialDelaySeconds: 10
          readinessProbe:
            httpGet:
              path: /
              port: 13133
          volumeMounts:
            - name: config
              mountPath: /etc/otel
            - name: varlog
              mountPath: /var/log
              readOnly: true
      volumes:
        - name: config
          configMap:
            name: otel-agent-config
        - name: varlog
          hostPath:
            path: /var/log
```

### 8.2 Gateway Deployment with HPA

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: otel-gateway
  namespace: observability
spec:
  replicas: 2
  selector:
    matchLabels:
      app: otel-gateway
  template:
    metadata:
      labels:
        app: otel-gateway
    spec:
      containers:
        - name: otel-gateway
          image: otel/opentelemetry-collector-contrib:0.96.0
          args: ["--config=/etc/otel/config.yaml"]
          resources:
            limits:
              cpu: 2
              memory: 2Gi
            requests:
              cpu: 500m
              memory: 512Mi
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: otel-gateway-hpa
  namespace: observability
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: otel-gateway
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 75
```

---

## 9. Troubleshooting

### 9.1 Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Data loss** | Receiver refused spans > 0 | Increase memory limit or add more Collector replicas |
| **High latency** | Export queue growing | Increase `num_consumers`, add batch processor |
| **OOM kills** | Collector pod restarts | Enable `memory_limiter`, reduce `num_traces` in tail sampling |
| **Missing spans** | Traces have gaps | Check context propagation, verify all services export to Collector |
| **Duplicate data** | Same span exported twice | Check for overlapping pipelines, verify at-least-once delivery |

### 9.2 Debug Exporter

```yaml
# Temporarily add debug exporter to see data flowing through the pipeline
exporters:
  debug:
    verbosity: detailed     # basic | normal | detailed
    sampling_initial: 5     # First N items logged
    sampling_thereafter: 100 # Then 1 in N items

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp/tempo, debug]  # Add debug alongside real exporter
```

### 9.3 zPages

zPages provide in-process debugging pages:

```yaml
extensions:
  zpages:
    endpoint: 0.0.0.0:55679

# Access:
# http://collector:55679/debug/tracez    -- Recent traces through the collector
# http://collector:55679/debug/pipelinez -- Pipeline status and stats
```

---

## 10. Next Steps

- [24_eBPF_Observability.md](./24_eBPF_Observability.md) -- Kernel-level observability with eBPF
- [25_Continuous_Profiling.md](./25_Continuous_Profiling.md) -- CPU and memory profiling in production

---

## Exercises

### Exercise 1: Pipeline Design

Design an OTel Collector pipeline for a company with:
- 50 microservices generating ~10,000 spans/second
- Requirements: keep all error traces, sample 10% of successful traces, drop health check traces
- Cost target: reduce trace storage by 80%
- Must preserve exemplar links from metrics to traces

Write the complete Collector configuration.

<details>
<summary>Show Answer</summary>

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

processors:
  memory_limiter:
    check_interval: 1s
    limit_mib: 1024
    spike_limit_mib: 256

  # Step 1: Drop health checks (eliminates ~20% of traces)
  filter/health:
    error_mode: ignore
    traces:
      span:
        - 'attributes["http.route"] == "/healthz"'
        - 'attributes["http.route"] == "/readyz"'
        - 'attributes["http.route"] == "/livez"'

  # Step 2: Tail sampling (reduces remaining by ~85%)
  tail_sampling:
    decision_wait: 30s
    num_traces: 200000
    expected_new_traces_per_sec: 8000  # After health check filtering
    policies:
      # Always keep errors
      - name: keep-errors
        type: status_code
        status_code:
          status_codes: [ERROR]
      # Always keep slow traces
      - name: keep-slow
        type: latency
        latency:
          threshold_ms: 2000
      # Sample 10% of the rest
      - name: sample-success
        type: probabilistic
        probabilistic:
          sampling_percentage: 10

  batch:
    send_batch_size: 2048
    timeout: 10s

connectors:
  # Generate metrics FROM traces BEFORE sampling
  # This ensures exemplars reference sampled traces
  spanmetrics:
    histogram:
      explicit:
        buckets: [5ms, 10ms, 50ms, 100ms, 500ms, 1s, 5s]
    exemplars:
      enabled: true

exporters:
  otlp/tempo:
    endpoint: tempo:4317
    tls:
      insecure: true
  prometheus:
    endpoint: 0.0.0.0:8889

service:
  pipelines:
    # Traces flow: receive → filter → spanmetrics (generates metrics) → sample → export
    traces/pre-sample:
      receivers: [otlp]
      processors: [memory_limiter, filter/health]
      exporters: [spanmetrics]  # Generate metrics from ALL traces

    traces/post-sample:
      receivers: [otlp]
      processors: [memory_limiter, filter/health, tail_sampling, batch]
      exporters: [otlp/tempo]   # Only sampled traces to storage

    metrics/spanmetrics:
      receivers: [spanmetrics]
      processors: [batch]
      exporters: [prometheus]
```

**Cost analysis:**
- Health check filtering: 10,000 → 8,000 spans/sec (20% reduction)
- Tail sampling (10% + errors + slow): 8,000 → ~1,500 spans/sec (~81% reduction)
- Total reduction: ~85% (meets the 80% target)
- Exemplars work because spanmetrics generates from pre-sampled traces

</details>

### Exercise 2: Troubleshooting

Your OTel Collector is experiencing the following symptoms:
- `otelcol_receiver_refused_spans` is increasing at 500/sec
- `process_runtime_total_alloc_bytes` is at 1.8 GB (limit: 2 GB)
- `otelcol_exporter_queue_size` is at 4,800 out of 5,000 capacity
- Export latency to Tempo has increased from 50ms to 5s

Diagnose the root cause and propose a fix for each symptom.

<details>
<summary>Show Answer</summary>

**Root cause analysis:**

The symptoms form a cascade starting from the export path:

1. **Export latency increase (50ms → 5s)**: Tempo is slow (overloaded, network issue, or disk I/O bottleneck). This is the **root cause** of the cascade.

2. **Queue full (4,800/5,000)**: Because exports are slow, the queue fills up. Items wait in the queue instead of being exported quickly.

3. **High memory (1.8 GB)**: The full queue and buffered data consume memory. Tail sampling (if used) also buffers traces in memory.

4. **Refused spans (500/sec)**: When memory approaches the limit, the `memory_limiter` processor starts refusing incoming data to prevent OOM.

**Fixes:**

| Symptom | Immediate Fix | Long-term Fix |
|---------|--------------|---------------|
| Export latency | Check Tempo health; restart if needed; check network | Scale Tempo (more ingesters); add Tempo query frontend |
| Queue full | Increase `queue_size` to 10000 | Add more Collector replicas; increase `num_consumers` |
| High memory | Increase `limit_mib` to 2048 and pod memory to 3Gi | Reduce `num_traces` in tail sampling; filter more data upstream |
| Refused spans | Deploy more agent-level Collectors to distribute load | Implement backpressure-aware routing with load balancing exporter |

**Configuration changes:**

```yaml
processors:
  memory_limiter:
    limit_mib: 2048          # Increase from current
    spike_limit_mib: 512

exporters:
  otlp/tempo:
    endpoint: tempo:4317
    retry_on_failure:
      enabled: true
      initial_interval: 1s
      max_interval: 30s
    sending_queue:
      enabled: true
      num_consumers: 20      # Increase from default 10
      queue_size: 10000      # Increase from 5000
    timeout: 30s             # Increase timeout for slow backend
```

</details>

---

## References

- [OpenTelemetry Collector Documentation](https://opentelemetry.io/docs/collector/)
- [OpenTelemetry Collector Contrib](https://github.com/open-telemetry/opentelemetry-collector-contrib)
- [OTel Collector Builder (OCB)](https://opentelemetry.io/docs/collector/custom-collector/)
- [Tail Sampling Processor](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/processor/tailsamplingprocessor)
- [OTTL (OpenTelemetry Transformation Language)](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/pkg/ottl)

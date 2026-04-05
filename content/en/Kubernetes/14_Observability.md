# 14. Observability

**Previous**: [Autoscaling](./13_Autoscaling.md) | **Next**: [Multi-Cluster](./15_Multi_Cluster.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Deploy and configure a Prometheus-based metrics pipeline with Grafana dashboards
2. Set up centralized logging using the EFK stack and Grafana Loki
3. Implement distributed tracing with OpenTelemetry and Jaeger
4. Configure health checks (liveness, readiness, startup probes) for robust service management
5. Build alerting pipelines with Alertmanager and apply debugging techniques for production incidents

---

You cannot operate what you cannot observe. Kubernetes clusters generate enormous amounts of data -- metrics from every pod, node, and control plane component; logs from every container; traces from every request. The challenge is not collecting data but building a coherent observability system that lets you answer questions about your system's health, performance, and behavior. This lesson covers the three pillars of observability -- metrics, logs, and traces -- along with health checks, alerting, and debugging techniques.

## Table of Contents

- [1. The Three Pillars of Observability](#1-the-three-pillars-of-observability)
- [2. Metrics with Prometheus and Grafana](#2-metrics-with-prometheus-and-grafana)
- [3. Kubernetes Metrics Pipeline](#3-kubernetes-metrics-pipeline)
- [4. Logging with EFK and Loki](#4-logging-with-efk-and-loki)
- [5. Distributed Tracing](#5-distributed-tracing)
- [6. Health Checks](#6-health-checks)
- [7. Alerting with Alertmanager](#7-alerting-with-alertmanager)
- [8. Debugging Techniques](#8-debugging-techniques)
- [Exercises](#exercises)

---

## 1. The Three Pillars of Observability

### 1.1 Overview

```
                    ┌───────────────────────────────────────┐
                    │          Observability                 │
                    │                                       │
        ┌───────────┼────────────┬──────────────┐          │
        │           │            │              │          │
        ▼           ▼            ▼              ▼          │
   ┌─────────┐ ┌─────────┐ ┌─────────┐  ┌───────────┐    │
   │ Metrics  │ │  Logs   │ │ Traces  │  │  Health   │    │
   │          │ │         │ │         │  │  Checks   │    │
   │Prometheus│ │EFK/Loki │ │Jaeger/  │  │Liveness/  │    │
   │ Grafana  │ │         │ │OTel     │  │Readiness  │    │
   └─────────┘ └─────────┘ └─────────┘  └───────────┘    │
        │           │            │              │          │
        └───────────┴────────────┴──────────────┘          │
                            │                              │
                    ┌───────▼──────┐                       │
                    │  Alerting    │                       │
                    │ Alertmanager │                       │
                    └──────────────┘                       │
                    └───────────────────────────────────────┘
```

| Pillar | Question It Answers | Tool |
|---|---|---|
| Metrics | What is happening? How much? | Prometheus, Grafana |
| Logs | Why did it happen? | EFK, Loki |
| Traces | Where did it happen in the call chain? | Jaeger, OpenTelemetry |

---

## 2. Metrics with Prometheus and Grafana

### 2.1 Prometheus Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Prometheus Server                         │
│                                                              │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐  │
│  │  Retrieval    │  │    TSDB       │  │  HTTP Server     │  │
│  │  (scrape      │  │  (time series │  │  (PromQL API)    │  │
│  │   targets)    │  │   database)   │  │                  │  │
│  └──────┬───────┘  └───────────────┘  └────────┬─────────┘  │
│         │                                       │            │
└─────────┼───────────────────────────────────────┼────────────┘
          │ scrape                                │ query
          ▼                                       ▼
   ┌──────────────┐                      ┌──────────────────┐
   │  Targets      │                      │   Grafana        │
   │  - Pods       │                      │   Alertmanager   │
   │  - Nodes      │                      │   API clients    │
   │  - Services   │                      └──────────────────┘
   └──────────────┘
```

### 2.2 Installing the Prometheus Stack

```bash
# Install kube-prometheus-stack (Prometheus + Grafana + Alertmanager + exporters)
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.storageClassName=gp3 \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=100Gi \
  --set grafana.adminPassword=admin \
  --set grafana.persistence.enabled=true \
  --set grafana.persistence.size=10Gi

# Verify
kubectl get pods -n monitoring
kubectl get svc -n monitoring
```

### 2.3 ServiceMonitor for Application Metrics

```yaml
# Tell Prometheus to scrape your application
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: web-app-monitor
  namespace: production
  labels:
    release: monitoring  # Must match Prometheus operator label selector
spec:
  selector:
    matchLabels:
      app: web-app
  endpoints:
  - port: metrics
    path: /metrics
    interval: 15s
    scrapeTimeout: 10s
  namespaceSelector:
    matchNames:
    - production
```

### 2.4 PodMonitor for Sidecar Metrics

```yaml
# For pods without a Service (e.g., Jobs, sidecars)
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: envoy-sidecar-monitor
  namespace: production
spec:
  selector:
    matchLabels:
      sidecar: envoy
  podMetricsEndpoints:
  - port: admin
    path: /stats/prometheus
    interval: 30s
```

### 2.5 Key PromQL Queries

```promql
# CPU usage by namespace
sum(rate(container_cpu_usage_seconds_total{namespace="production"}[5m])) by (pod)

# Memory usage percentage
sum(container_memory_working_set_bytes{namespace="production"}) by (pod) /
sum(container_spec_memory_limit_bytes{namespace="production"}) by (pod) * 100

# Pod restart count
increase(kube_pod_container_status_restarts_total{namespace="production"}[1h])

# Request rate per service (from application metrics)
sum(rate(http_requests_total{namespace="production"}[5m])) by (service)

# Error rate (5xx)
sum(rate(http_requests_total{status=~"5.."}[5m])) /
sum(rate(http_requests_total[5m])) * 100

# P99 request latency
histogram_quantile(0.99,
  sum(rate(http_request_duration_seconds_bucket{namespace="production"}[5m])) by (le, service)
)

# Node disk pressure
kube_node_status_condition{condition="DiskPressure", status="true"} == 1

# Pods not ready for more than 5 minutes
min_over_time(kube_pod_status_ready{condition="true"}[5m]) == 0
```

### 2.6 Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "Kubernetes Cluster Overview",
    "panels": [
      {
        "title": "CPU Usage by Namespace",
        "type": "timeseries",
        "targets": [
          {
            "expr": "sum(rate(container_cpu_usage_seconds_total{container!=\"\"}[5m])) by (namespace)",
            "legendFormat": "{{namespace}}"
          }
        ]
      },
      {
        "title": "Memory Usage by Namespace",
        "type": "timeseries",
        "targets": [
          {
            "expr": "sum(container_memory_working_set_bytes{container!=\"\"}) by (namespace) / 1024 / 1024 / 1024",
            "legendFormat": "{{namespace}} (GiB)"
          }
        ]
      },
      {
        "title": "Pod Restart Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(increase(kube_pod_container_status_restarts_total[1h]))"
          }
        ]
      }
    ]
  }
}
```

---

## 3. Kubernetes Metrics Pipeline

### 3.1 Metrics-Server

Metrics-server provides real-time CPU and memory metrics used by `kubectl top` and HPA:

```bash
# Install metrics-server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# For local clusters (minikube/kind), add --kubelet-insecure-tls
kubectl patch deployment metrics-server -n kube-system \
  --type='json' \
  -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-insecure-tls"}]'

# Verify
kubectl top nodes
# NAME          CPU(cores)   CPU%   MEMORY(bytes)   MEMORY%
# node-1        450m         11%    3200Mi          41%
# node-2        380m         9%     2800Mi          36%

kubectl top pods -n production --sort-by=cpu
```

### 3.2 kube-state-metrics

kube-state-metrics generates metrics about the state of Kubernetes objects (not resource usage):

```bash
# Installed automatically with kube-prometheus-stack
# Or install standalone:
helm install kube-state-metrics prometheus-community/kube-state-metrics \
  --namespace monitoring
```

Key metrics from kube-state-metrics:

```promql
# Deployment replicas status
kube_deployment_spec_replicas
kube_deployment_status_replicas_available
kube_deployment_status_replicas_unavailable

# Pod status
kube_pod_status_phase{phase="Running"}
kube_pod_status_phase{phase="Pending"}
kube_pod_status_phase{phase="Failed"}

# Container status
kube_pod_container_status_waiting_reason
kube_pod_container_status_terminated_reason

# Resource quotas
kube_resourcequota{type="hard"}
kube_resourcequota{type="used"}
```

### 3.3 Node Exporter

node-exporter provides hardware and OS-level metrics from each node:

```promql
# Node CPU usage
100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) by (instance) * 100)

# Node memory usage
(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100

# Disk usage percentage
(1 - node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100

# Network traffic
rate(node_network_receive_bytes_total{device="eth0"}[5m])
rate(node_network_transmit_bytes_total{device="eth0"}[5m])
```

---

## 4. Logging with EFK and Loki

### 4.1 EFK Stack Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Node 1              Node 2              Node 3          │
│  ┌──────────┐       ┌──────────┐       ┌──────────┐    │
│  │ Fluentd  │       │ Fluentd  │       │ Fluentd  │    │
│  │(DaemonSet)│       │(DaemonSet)│       │(DaemonSet)│    │
│  └────┬─────┘       └────┬─────┘       └────┬─────┘    │
│       │                  │                  │           │
│       └──────────────────┼──────────────────┘           │
│                          │                              │
│                    ┌─────▼──────┐                       │
│                    │Elasticsearch│                       │
│                    │ (storage)   │                       │
│                    └─────┬──────┘                       │
│                          │                              │
│                    ┌─────▼──────┐                       │
│                    │  Kibana    │                       │
│                    │ (visualize)│                       │
│                    └────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Fluentd DaemonSet

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd
  namespace: logging
spec:
  selector:
    matchLabels:
      app: fluentd
  template:
    metadata:
      labels:
        app: fluentd
    spec:
      serviceAccountName: fluentd
      tolerations:
      - key: node-role.kubernetes.io/control-plane
        effect: NoSchedule
      containers:
      - name: fluentd
        image: fluent/fluentd-kubernetes-daemonset:v1.16-debian-elasticsearch8-1
        env:
        - name: FLUENT_ELASTICSEARCH_HOST
          value: "elasticsearch.logging.svc"
        - name: FLUENT_ELASTICSEARCH_PORT
          value: "9200"
        - name: FLUENT_ELASTICSEARCH_SCHEME
          value: "https"
        - name: FLUENT_ELASTICSEARCH_SSL_VERIFY
          value: "false"
        resources:
          requests:
            cpu: 100m
            memory: 200Mi
          limits:
            cpu: 500m
            memory: 500Mi
        volumeMounts:
        - name: varlog
          mountPath: /var/log
        - name: containers
          mountPath: /var/lib/docker/containers
          readOnly: true
        - name: fluentd-config
          mountPath: /fluentd/etc/conf.d
      volumes:
      - name: varlog
        hostPath:
          path: /var/log
      - name: containers
        hostPath:
          path: /var/lib/docker/containers
      - name: fluentd-config
        configMap:
          name: fluentd-config
```

### 4.3 Fluentd Configuration

```yaml
# fluentd-config ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
  namespace: logging
data:
  kubernetes.conf: |
    <source>
      @type tail
      path /var/log/containers/*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      read_from_head true
      <parse>
        @type json
        time_key time
        time_format %Y-%m-%dT%H:%M:%S.%NZ
      </parse>
    </source>

    <filter kubernetes.**>
      @type kubernetes_metadata
      @id filter_kube_metadata
    </filter>

    # Drop health check logs to reduce noise
    <filter kubernetes.**>
      @type grep
      <exclude>
        key $.kubernetes.container_name
        pattern /^(healthz|readyz)$/
      </exclude>
    </filter>

    <match kubernetes.**>
      @type elasticsearch
      host elasticsearch.logging.svc
      port 9200
      logstash_format true
      logstash_prefix k8s
      include_tag_key true
      <buffer>
        @type file
        path /var/log/fluentd-buffers/kubernetes.system.buffer
        flush_mode interval
        flush_interval 5s
        retry_max_interval 30
        chunk_limit_size 8M
        total_limit_size 512M
        overflow_action drop_oldest_chunk
      </buffer>
    </match>
```

### 4.4 Grafana Loki (Lightweight Alternative)

Loki is a log aggregation system designed to be cost-effective and easy to operate:

```bash
# Install Loki stack (Loki + Promtail)
helm repo add grafana https://grafana.github.io/helm-charts
helm install loki grafana/loki-stack \
  --namespace logging \
  --create-namespace \
  --set loki.persistence.enabled=true \
  --set loki.persistence.size=50Gi \
  --set promtail.enabled=true \
  --set grafana.enabled=false  # Use existing Grafana
```

### 4.5 Promtail Configuration

```yaml
# Promtail pipeline to enrich and filter logs
apiVersion: v1
kind: ConfigMap
metadata:
  name: promtail-config
  namespace: logging
data:
  promtail.yaml: |
    server:
      http_listen_port: 3101
    positions:
      filename: /tmp/positions.yaml
    clients:
    - url: http://loki.logging.svc:3100/loki/api/v1/push
    scrape_configs:
    - job_name: kubernetes-pods
      kubernetes_sd_configs:
      - role: pod
      pipeline_stages:
      - cri: {}
      - json:
          expressions:
            level: level
            msg: msg
            timestamp: timestamp
      - labels:
          level:
      - match:
          selector: '{app="nginx"}'
          stages:
          - regex:
              expression: '^(?P<remote_addr>[\w.]+) - .* "(?P<method>\w+) (?P<path>[^ ]+)'
          - labels:
              method:
              path:
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        target_label: app
      - source_labels: [__meta_kubernetes_namespace]
        target_label: namespace
      - source_labels: [__meta_kubernetes_pod_name]
        target_label: pod
```

### 4.6 LogQL Queries (Loki)

```logql
# All logs from a specific pod
{namespace="production", pod="web-app-7f8b9c6d-x2k4q"}

# Error logs from all pods in production
{namespace="production"} |= "error"

# JSON structured logs with level filter
{app="api-server"} | json | level="error"

# Rate of error logs per minute
rate({namespace="production"} |= "error" [1m])

# Top 10 error messages
topk(10, sum by (msg) (rate({app="api-server"} | json | level="error" [5m])))

# Log volume by namespace
sum by (namespace) (rate({job="kubernetes-pods"}[5m]))
```

---

## 5. Distributed Tracing

### 5.1 OpenTelemetry Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Application Pods                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐         │
│  │  Service A  │──│  Service B  │──│  Service C  │         │
│  │  (OTel SDK) │  │  (OTel SDK) │  │  (OTel SDK) │         │
│  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘         │
│         │               │               │                │
│         └───────────────┼───────────────┘                │
│                         │ OTLP                           │
│                         ▼                                │
│              ┌──────────────────────┐                    │
│              │  OTel Collector      │                    │
│              │  (DaemonSet or       │                    │
│              │   Deployment)        │                    │
│              └──────────┬───────────┘                    │
│                         │                                │
│           ┌─────────────┼──────────────┐                 │
│           ▼             ▼              ▼                 │
│    ┌──────────┐  ┌──────────┐  ┌──────────────┐         │
│    │  Jaeger  │  │Prometheus│  │    Loki      │         │
│    │ (traces) │  │ (metrics)│  │   (logs)     │         │
│    └──────────┘  └──────────┘  └──────────────┘         │
└──────────────────────────────────────────────────────────┘
```

### 5.2 OpenTelemetry Collector

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: otel-collector
  namespace: observability
spec:
  selector:
    matchLabels:
      app: otel-collector
  template:
    metadata:
      labels:
        app: otel-collector
    spec:
      containers:
      - name: collector
        image: otel/opentelemetry-collector-contrib:0.92.0
        ports:
        - containerPort: 4317  # OTLP gRPC
        - containerPort: 4318  # OTLP HTTP
        - containerPort: 8888  # Metrics
        volumeMounts:
        - name: config
          mountPath: /etc/otelcol-contrib
        resources:
          requests:
            cpu: 200m
            memory: 256Mi
          limits:
            cpu: "1"
            memory: 1Gi
      volumes:
      - name: config
        configMap:
          name: otel-collector-config
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: otel-collector-config
  namespace: observability
data:
  config.yaml: |
    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317
          http:
            endpoint: 0.0.0.0:4318

    processors:
      batch:
        timeout: 5s
        send_batch_size: 1024
      memory_limiter:
        check_interval: 1s
        limit_mib: 800
        spike_limit_mib: 200
      resource:
        attributes:
        - key: cluster.name
          value: production
          action: upsert

    exporters:
      otlp/jaeger:
        endpoint: jaeger-collector.observability.svc:4317
        tls:
          insecure: true
      prometheus:
        endpoint: 0.0.0.0:8889
      loki:
        endpoint: http://loki.logging.svc:3100/loki/api/v1/push

    service:
      pipelines:
        traces:
          receivers: [otlp]
          processors: [memory_limiter, batch, resource]
          exporters: [otlp/jaeger]
        metrics:
          receivers: [otlp]
          processors: [memory_limiter, batch]
          exporters: [prometheus]
        logs:
          receivers: [otlp]
          processors: [memory_limiter, batch]
          exporters: [loki]
```

### 5.3 Instrumenting a Go Application

```go
package main

import (
    "context"
    "log"
    "net/http"
    "time"

    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/attribute"
    "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
    "go.opentelemetry.io/otel/propagation"
    "go.opentelemetry.io/otel/sdk/resource"
    sdktrace "go.opentelemetry.io/otel/sdk/trace"
    semconv "go.opentelemetry.io/otel/semconv/v1.24.0"
    "go.opentelemetry.io/otel/trace"
    "go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
)

func initTracer(ctx context.Context) (*sdktrace.TracerProvider, error) {
    exporter, err := otlptracegrpc.New(ctx,
        otlptracegrpc.WithEndpoint("otel-collector.observability.svc:4317"),
        otlptracegrpc.WithInsecure(),
    )
    if err != nil {
        return nil, err
    }

    tp := sdktrace.NewTracerProvider(
        sdktrace.WithBatcher(exporter,
            sdktrace.WithBatchTimeout(5*time.Second),
        ),
        sdktrace.WithResource(resource.NewWithAttributes(
            semconv.SchemaURL,
            semconv.ServiceNameKey.String("order-service"),
            semconv.ServiceVersionKey.String("1.0.0"),
            attribute.String("environment", "production"),
        )),
        sdktrace.WithSampler(sdktrace.TraceIDRatioBased(0.1)), // Sample 10%
    )

    otel.SetTracerProvider(tp)
    otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
        propagation.TraceContext{},
        propagation.Baggage{},
    ))

    return tp, nil
}

var tracer = otel.Tracer("order-service")

func handleOrder(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context()

    // Create a child span for database query
    ctx, span := tracer.Start(ctx, "process-order",
        trace.WithAttributes(
            attribute.String("order.id", r.URL.Query().Get("id")),
        ),
    )
    defer span.End()

    // Simulate database call
    _, dbSpan := tracer.Start(ctx, "db.query",
        trace.WithAttributes(
            attribute.String("db.system", "postgresql"),
            attribute.String("db.statement", "SELECT * FROM orders WHERE id = ?"),
        ),
    )
    time.Sleep(50 * time.Millisecond)
    dbSpan.End()

    w.WriteHeader(http.StatusOK)
    w.Write([]byte(`{"status": "processed"}`))
}

func main() {
    ctx := context.Background()
    tp, err := initTracer(ctx)
    if err != nil {
        log.Fatal(err)
    }
    defer tp.Shutdown(ctx)

    handler := otelhttp.NewHandler(http.HandlerFunc(handleOrder), "order-handler")
    http.Handle("/order", handler)
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

### 5.4 Jaeger Installation

```bash
# Install Jaeger operator
kubectl create namespace observability
helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
helm install jaeger-operator jaegertracing/jaeger-operator \
  --namespace observability \
  --set rbac.clusterRole=true

# Create a Jaeger instance
kubectl apply -f - <<EOF
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: jaeger
  namespace: observability
spec:
  strategy: production
  storage:
    type: elasticsearch
    options:
      es:
        server-urls: https://elasticsearch.logging.svc:9200
    secretName: jaeger-es-secret
  collector:
    replicas: 2
    resources:
      requests:
        cpu: 200m
        memory: 256Mi
  query:
    replicas: 1
EOF
```

---

## 6. Health Checks

### 6.1 Probe Types

Kubernetes provides three types of health probes:

| Probe | Purpose | Failure Effect |
|---|---|---|
| **Liveness** | Is the container running correctly? | Container is restarted |
| **Readiness** | Can the container serve traffic? | Removed from Service endpoints |
| **Startup** | Has the container finished starting? | Liveness/readiness probes are paused |

### 6.2 Probe Configuration

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: web-app
spec:
  containers:
  - name: web-app
    image: example.com/web-app:v1
    ports:
    - containerPort: 8080

    # Startup probe: allow slow startup (up to 5 min)
    startupProbe:
      httpGet:
        path: /healthz
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 10
      failureThreshold: 30    # 30 * 10s = 5 min max startup time

    # Liveness probe: restart if stuck
    livenessProbe:
      httpGet:
        path: /healthz
        port: 8080
      initialDelaySeconds: 0   # Starts after startup probe passes
      periodSeconds: 10
      timeoutSeconds: 5
      failureThreshold: 3      # Restart after 3 consecutive failures
      successThreshold: 1

    # Readiness probe: manage traffic
    readinessProbe:
      httpGet:
        path: /readyz
        port: 8080
      initialDelaySeconds: 0
      periodSeconds: 5
      timeoutSeconds: 3
      failureThreshold: 2      # Remove from endpoints after 2 failures
      successThreshold: 2      # Require 2 successes to be added back
```

### 6.3 Probe Mechanisms

```yaml
# HTTP GET probe
livenessProbe:
  httpGet:
    path: /healthz
    port: 8080
    httpHeaders:
    - name: Custom-Header
      value: "probe"

# TCP socket probe (useful for databases)
livenessProbe:
  tcpSocket:
    port: 5432

# Exec probe (run a command inside the container)
livenessProbe:
  exec:
    command:
    - /bin/sh
    - -c
    - pg_isready -U postgres

# gRPC probe (Kubernetes 1.27+)
livenessProbe:
  grpc:
    port: 50051
    service: "health"  # gRPC health checking protocol
```

### 6.4 Health Check Implementation (Go)

```go
package main

import (
    "database/sql"
    "encoding/json"
    "net/http"
    "sync/atomic"
)

var (
    ready   int32 = 0
    healthy int32 = 1
)

type HealthResponse struct {
    Status string            `json:"status"`
    Checks map[string]string `json:"checks,omitempty"`
}

func healthzHandler(db *sql.DB) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        if atomic.LoadInt32(&healthy) == 0 {
            w.WriteHeader(http.StatusServiceUnavailable)
            json.NewEncoder(w).Encode(HealthResponse{Status: "unhealthy"})
            return
        }

        checks := map[string]string{}

        // Check database connectivity
        if err := db.Ping(); err != nil {
            checks["database"] = "down: " + err.Error()
            w.WriteHeader(http.StatusServiceUnavailable)
            json.NewEncoder(w).Encode(HealthResponse{Status: "unhealthy", Checks: checks})
            return
        }
        checks["database"] = "up"

        w.WriteHeader(http.StatusOK)
        json.NewEncoder(w).Encode(HealthResponse{Status: "healthy", Checks: checks})
    }
}

func readyzHandler() http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        if atomic.LoadInt32(&ready) == 0 {
            w.WriteHeader(http.StatusServiceUnavailable)
            json.NewEncoder(w).Encode(HealthResponse{Status: "not ready"})
            return
        }
        w.WriteHeader(http.StatusOK)
        json.NewEncoder(w).Encode(HealthResponse{Status: "ready"})
    }
}

// Call this after startup tasks complete
func markReady() {
    atomic.StoreInt32(&ready, 1)
}
```

### 6.5 Anti-Patterns

| Anti-Pattern | Problem | Solution |
|---|---|---|
| Liveness probe checks dependencies | External service down causes cascading restarts | Only check process health, not dependencies |
| Same endpoint for liveness and readiness | Cannot independently control traffic and restarts | Use separate `/healthz` and `/readyz` endpoints |
| No startup probe for slow-starting apps | Liveness probe kills container before startup | Use startup probe with generous failureThreshold |
| Aggressive probe intervals | High CPU from constant health checks | Use 10-30s intervals for liveness |

---

## 7. Alerting with Alertmanager

### 7.1 Alertmanager Architecture

```
Prometheus ──(firing alerts)──▶ Alertmanager
                                    │
                              ┌─────┼─────┐
                              ▼     ▼     ▼
                           Route  Group  Silence
                              │     │     │
                              ▼     ▼     ▼
                        ┌──────────────────────┐
                        │     Receivers         │
                        │  - Slack              │
                        │  - PagerDuty          │
                        │  - Email              │
                        │  - Webhook            │
                        └──────────────────────┘
```

### 7.2 PrometheusRule for Alerting

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: kubernetes-alerts
  namespace: monitoring
  labels:
    release: monitoring
spec:
  groups:
  - name: kubernetes.pod.alerts
    rules:
    - alert: PodCrashLooping
      expr: |
        increase(kube_pod_container_status_restarts_total[1h]) > 5
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "Pod {{ $labels.namespace }}/{{ $labels.pod }} is crash looping"
        description: "Pod has restarted {{ $value }} times in the last hour."
        runbook_url: "https://runbooks.example.com/pod-crash-loop"

    - alert: PodNotReady
      expr: |
        kube_pod_status_ready{condition="true"} == 0
      for: 15m
      labels:
        severity: warning
      annotations:
        summary: "Pod {{ $labels.namespace }}/{{ $labels.pod }} has been not ready for 15 minutes"

    - alert: DeploymentReplicasMismatch
      expr: |
        kube_deployment_spec_replicas != kube_deployment_status_ready_replicas
      for: 10m
      labels:
        severity: critical
      annotations:
        summary: "Deployment {{ $labels.namespace }}/{{ $labels.deployment }} has mismatched replicas"
        description: "Expected {{ $value }} replicas but only {{ $labels.ready_replicas }} are ready."

  - name: kubernetes.node.alerts
    rules:
    - alert: NodeHighCPU
      expr: |
        100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 85
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "Node {{ $labels.instance }} CPU usage is above 85%"

    - alert: NodeHighMemory
      expr: |
        (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100 > 90
      for: 10m
      labels:
        severity: critical
      annotations:
        summary: "Node {{ $labels.instance }} memory usage is above 90%"

    - alert: NodeDiskPressure
      expr: |
        (1 - node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100 > 85
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Node {{ $labels.instance }} disk usage is above 85%"
```

### 7.3 Alertmanager Configuration

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: alertmanager-config
  namespace: monitoring
stringData:
  alertmanager.yaml: |
    global:
      resolve_timeout: 5m
      slack_api_url: 'https://hooks.slack.com/services/T00/B00/XXX'

    route:
      receiver: 'default'
      group_by: ['alertname', 'namespace']
      group_wait: 30s
      group_interval: 5m
      repeat_interval: 4h
      routes:
      - receiver: 'critical-pagerduty'
        match:
          severity: critical
        group_wait: 10s
        repeat_interval: 1h
      - receiver: 'warning-slack'
        match:
          severity: warning
        repeat_interval: 4h

    receivers:
    - name: 'default'
      slack_configs:
      - channel: '#alerts-default'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ end }}'

    - name: 'critical-pagerduty'
      pagerduty_configs:
      - service_key: '<pagerduty-service-key>'
        severity: 'critical'
        description: '{{ .GroupLabels.alertname }}: {{ .CommonAnnotations.summary }}'

    - name: 'warning-slack'
      slack_configs:
      - channel: '#alerts-warning'
        title: '[{{ .Status | toUpper }}] {{ .GroupLabels.alertname }}'
        text: >-
          {{ range .Alerts }}
          *Alert:* {{ .Annotations.summary }}
          *Namespace:* {{ .Labels.namespace }}
          *Description:* {{ .Annotations.description }}
          {{ end }}

    inhibit_rules:
    - source_match:
        severity: 'critical'
      target_match:
        severity: 'warning'
      equal: ['alertname', 'namespace']
```

---

## 8. Debugging Techniques

### 8.1 Pod Debugging Flowchart

```
Pod issue detected
    │
    ├── Status: Pending?
    │       └── kubectl describe pod → Check Events
    │           ├── Insufficient resources → Scale nodes or reduce requests
    │           ├── Unschedulable → Check taints/tolerations, node selector
    │           └── Image pull error → Check image name, pull secrets
    │
    ├── Status: CrashLoopBackOff?
    │       └── kubectl logs <pod> --previous → Check crash logs
    │           ├── OOMKilled → Increase memory limits
    │           ├── Application error → Fix code
    │           └── Missing config → Check ConfigMap/Secret mounts
    │
    ├── Status: Running but unhealthy?
    │       └── kubectl describe pod → Check probe failures
    │           ├── Liveness failing → Check /healthz endpoint
    │           ├── Readiness failing → Check dependencies
    │           └── High restart count → Check resource limits
    │
    └── Performance issue?
            └── kubectl top pod → Check resource usage
                ├── High CPU → Profile application, scale out
                └── High memory → Check for leaks, increase limits
```

### 8.2 Essential Debugging Commands

```bash
# Pod diagnostics
kubectl get pods -o wide                       # Show node placement
kubectl describe pod <pod>                     # Full pod details and events
kubectl logs <pod> -c <container>              # Container logs
kubectl logs <pod> --previous                  # Previous container logs (crash)
kubectl logs <pod> --tail=100 -f               # Follow last 100 lines

# Resource usage
kubectl top pods --sort-by=cpu                 # Sort by CPU
kubectl top pods --sort-by=memory              # Sort by memory
kubectl top pods --containers                  # Per-container breakdown

# Network debugging
kubectl exec -it <pod> -- curl localhost:8080/healthz
kubectl exec -it <pod> -- nslookup kubernetes.default
kubectl exec -it <pod> -- wget -qO- http://service-name:port/path

# Events (cluster-wide issues)
kubectl get events --sort-by=.lastTimestamp -A
kubectl get events --field-selector type=Warning -A
```

### 8.3 Ephemeral Debug Containers

```bash
# Attach a debug container to a running pod (without restarting it)
kubectl debug -it <pod> --image=busybox --target=<container>

# Create a copy of the pod with a debug container
kubectl debug <pod> -it --copy-to=debug-pod --container=debug --image=ubuntu

# Debug a node
kubectl debug node/<node-name> -it --image=ubuntu

# Use nicolaka/netshoot for network debugging
kubectl debug -it <pod> --image=nicolaka/netshoot --target=<container>
# Inside the debug container:
# tcpdump -i eth0 port 8080
# ss -tlnp
# curl -v http://service:port
```

### 8.4 Common Issues and Solutions

| Symptom | Diagnostic | Solution |
|---|---|---|
| `ImagePullBackOff` | `kubectl describe pod` | Fix image name, add imagePullSecret |
| `CrashLoopBackOff` | `kubectl logs --previous` | Fix application crash, check config |
| `OOMKilled` | `kubectl describe pod` (last state) | Increase memory limits |
| `Pending` (no events) | `kubectl describe pod` | Insufficient cluster resources |
| `Evicted` | `kubectl describe pod` | Node under disk/memory pressure |
| DNS resolution fails | `kubectl exec -- nslookup` | Check CoreDNS pods and ConfigMap |
| Service unreachable | `kubectl get endpoints` | Check selector labels, pod readiness |

---

## Exercises

### Exercise 1: Prometheus Monitoring Setup

Write the complete manifests to monitor a custom application: (a) a ServiceMonitor that scrapes metrics from pods labeled `app: payment-service` on port `metrics` every 30 seconds, (b) a PrometheusRule with three alerts: high error rate (>5% for 5 minutes), high latency (p99 > 500ms for 10 minutes), and pod restarts (>3 in 1 hour), (c) the PromQL queries for a Grafana dashboard showing request rate, error rate, and latency percentiles.

<details>
<summary>Show Answer</summary>

```yaml
# (a) ServiceMonitor
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: payment-service-monitor
  namespace: production
  labels:
    release: monitoring
spec:
  selector:
    matchLabels:
      app: payment-service
  endpoints:
  - port: metrics
    path: /metrics
    interval: 30s
    scrapeTimeout: 10s
  namespaceSelector:
    matchNames:
    - production
---
# (b) PrometheusRule
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: payment-service-alerts
  namespace: monitoring
  labels:
    release: monitoring
spec:
  groups:
  - name: payment-service
    rules:
    - alert: PaymentHighErrorRate
      expr: |
        sum(rate(http_requests_total{app="payment-service", status=~"5.."}[5m]))
        /
        sum(rate(http_requests_total{app="payment-service"}[5m])) > 0.05
      for: 5m
      labels:
        severity: critical
        service: payment
      annotations:
        summary: "Payment service error rate is above 5%"
        description: "Current error rate: {{ $value | humanizePercentage }}"

    - alert: PaymentHighLatency
      expr: |
        histogram_quantile(0.99,
          sum(rate(http_request_duration_seconds_bucket{app="payment-service"}[5m])) by (le)
        ) > 0.5
      for: 10m
      labels:
        severity: warning
        service: payment
      annotations:
        summary: "Payment service p99 latency above 500ms"
        description: "Current p99: {{ $value | humanizeDuration }}"

    - alert: PaymentPodRestarts
      expr: |
        increase(kube_pod_container_status_restarts_total{
          namespace="production",
          pod=~"payment-service.*"
        }[1h]) > 3
      for: 5m
      labels:
        severity: warning
        service: payment
      annotations:
        summary: "Payment service pod {{ $labels.pod }} restarting frequently"
```

```promql
# (c) Grafana dashboard queries

# Request rate
sum(rate(http_requests_total{app="payment-service"}[5m])) by (method, path)

# Error rate percentage
sum(rate(http_requests_total{app="payment-service", status=~"5.."}[5m]))
/
sum(rate(http_requests_total{app="payment-service"}[5m])) * 100

# Latency percentiles
histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket{app="payment-service"}[5m])) by (le))
histogram_quantile(0.90, sum(rate(http_request_duration_seconds_bucket{app="payment-service"}[5m])) by (le))
histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{app="payment-service"}[5m])) by (le))
```

</details>

### Exercise 2: Logging Pipeline

Design and configure a logging pipeline using Grafana Loki: (a) write the Promtail configuration to collect logs from all pods, parse JSON-formatted logs, and extract `level`, `msg`, and `trace_id` fields as labels, (b) write 5 LogQL queries to find: all errors in the last hour, slow requests (>1s), logs for a specific trace ID, rate of errors per service, and top 5 error messages, (c) explain how to set up log-based alerting in Grafana.

<details>
<summary>Show Answer</summary>

**(a) Promtail configuration:**

```yaml
scrape_configs:
- job_name: kubernetes-pods
  kubernetes_sd_configs:
  - role: pod
  pipeline_stages:
  - cri: {}
  - json:
      expressions:
        level: level
        msg: msg
        trace_id: trace_id
        duration_ms: duration_ms
  - labels:
      level:
  - template:
      source: level
      template: '{{ ToLower .Value }}'
  - output:
      source: msg
  relabel_configs:
  - source_labels: [__meta_kubernetes_pod_label_app]
    target_label: app
  - source_labels: [__meta_kubernetes_namespace]
    target_label: namespace
  - source_labels: [__meta_kubernetes_pod_name]
    target_label: pod
  - source_labels: [__meta_kubernetes_pod_container_name]
    target_label: container
```

**(b) LogQL queries:**

```logql
# 1. All errors in the last hour
{namespace="production"} | json | level="error" | line_format "{{.msg}}"

# 2. Slow requests (duration > 1000ms)
{namespace="production"} | json | duration_ms > 1000

# 3. Logs for a specific trace ID
{namespace="production"} |= "abc123-trace-id-here"

# 4. Rate of errors per service (per minute)
sum by (app) (rate({namespace="production"} | json | level="error" [1m]))

# 5. Top 5 error messages
topk(5, sum by (msg) (count_over_time({namespace="production"} | json | level="error" [1h])))
```

**(c) Log-based alerting in Grafana:** In Grafana, create a new Alert Rule with data source set to Loki. Use a LogQL metric query like `sum(rate({namespace="production"} | json | level="error" [5m])) > 10` as the condition. Set the evaluation interval (e.g., every 1 minute) and the pending period (e.g., 5 minutes). Configure notification channels (Slack, PagerDuty) in the contact points. This fires an alert when the error log rate exceeds 10 per second for 5 consecutive minutes.

</details>

### Exercise 3: Distributed Tracing Implementation

Write the Go code for a two-service tracing setup: (a) Service A receives HTTP requests, creates a root span, makes an HTTP call to Service B, and returns the response, (b) Service B receives the request, creates a child span, queries a database (simulated), and returns data, (c) both services export traces to an OpenTelemetry Collector. Include the OTel Collector ConfigMap and Jaeger deployment.

<details>
<summary>Show Answer</summary>

```go
// Service A - api-gateway
package main

import (
    "context"
    "io"
    "log"
    "net/http"
    "time"

    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
    "go.opentelemetry.io/otel/propagation"
    "go.opentelemetry.io/otel/sdk/resource"
    sdktrace "go.opentelemetry.io/otel/sdk/trace"
    semconv "go.opentelemetry.io/otel/semconv/v1.24.0"
    "go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
)

var httpClient = &http.Client{Transport: otelhttp.NewTransport(http.DefaultTransport)}

func initTracer(ctx context.Context, serviceName string) *sdktrace.TracerProvider {
    exp, _ := otlptracegrpc.New(ctx,
        otlptracegrpc.WithEndpoint("otel-collector:4317"),
        otlptracegrpc.WithInsecure(),
    )
    tp := sdktrace.NewTracerProvider(
        sdktrace.WithBatcher(exp),
        sdktrace.WithResource(resource.NewWithAttributes(
            semconv.SchemaURL, semconv.ServiceNameKey.String(serviceName),
        )),
    )
    otel.SetTracerProvider(tp)
    otel.SetTextMapPropagator(propagation.TraceContext{})
    return tp
}

func handleRequest(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context()
    tracer := otel.Tracer("api-gateway")
    ctx, span := tracer.Start(ctx, "handle-request")
    defer span.End()

    req, _ := http.NewRequestWithContext(ctx, "GET", "http://user-service:8081/user", nil)
    resp, err := httpClient.Do(req)
    if err != nil {
        http.Error(w, err.Error(), 500)
        return
    }
    defer resp.Body.Close()
    body, _ := io.ReadAll(resp.Body)
    w.Write(body)
}

func main() {
    ctx := context.Background()
    tp := initTracer(ctx, "api-gateway")
    defer tp.Shutdown(ctx)

    http.Handle("/api", otelhttp.NewHandler(http.HandlerFunc(handleRequest), "api"))
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

```go
// Service B - user-service
package main

import (
    "context"
    "log"
    "net/http"
    "time"

    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/attribute"
    "go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
)

func handleUser(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context()
    tracer := otel.Tracer("user-service")

    _, dbSpan := tracer.Start(ctx, "db.query.user",
        // Add semantic attributes
    )
    dbSpan.SetAttributes(
        attribute.String("db.system", "postgresql"),
        attribute.String("db.statement", "SELECT * FROM users WHERE id=$1"),
    )
    time.Sleep(30 * time.Millisecond) // Simulate DB call
    dbSpan.End()

    w.Write([]byte(`{"id": 1, "name": "Alice"}`))
}

func main() {
    ctx := context.Background()
    tp := initTracer(ctx, "user-service") // Same initTracer as Service A
    defer tp.Shutdown(ctx)

    http.Handle("/user", otelhttp.NewHandler(http.HandlerFunc(handleUser), "user"))
    log.Fatal(http.ListenAndServe(":8081", nil))
}
```

```yaml
# OTel Collector ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: otel-collector-config
  namespace: observability
data:
  config.yaml: |
    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317
    processors:
      batch:
        timeout: 5s
    exporters:
      otlp/jaeger:
        endpoint: jaeger-collector:4317
        tls:
          insecure: true
    service:
      pipelines:
        traces:
          receivers: [otlp]
          processors: [batch]
          exporters: [otlp/jaeger]
---
# Jaeger all-in-one (for development)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jaeger
  namespace: observability
spec:
  replicas: 1
  selector:
    matchLabels:
      app: jaeger
  template:
    metadata:
      labels:
        app: jaeger
    spec:
      containers:
      - name: jaeger
        image: jaegertracing/all-in-one:1.53
        ports:
        - containerPort: 16686  # UI
        - containerPort: 4317   # OTLP gRPC
        - containerPort: 14268  # HTTP collector
        env:
        - name: COLLECTOR_OTLP_ENABLED
          value: "true"
```

</details>

### Exercise 4: Health Check Design

Design health checks for a Java Spring Boot application that: (a) takes 90 seconds to start (loads ML model into memory), (b) depends on PostgreSQL and Redis, (c) should not receive traffic during graceful shutdown. Write the complete pod spec with all three probe types. Explain your choice of probe parameters. Also write the Go code for a custom health check endpoint that checks dependency health without causing cascading failures.

<details>
<summary>Show Answer</summary>

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-service
spec:
  terminationGracePeriodSeconds: 30
  containers:
  - name: ml-service
    image: example.com/ml-service:v1
    ports:
    - containerPort: 8080
    lifecycle:
      preStop:
        exec:
          command: ["/bin/sh", "-c", "sleep 5"]  # Allow in-flight requests to complete

    # Startup: allow up to 120s (12 * 10s) for ML model loading
    startupProbe:
      httpGet:
        path: /actuator/health/startup
        port: 8080
      initialDelaySeconds: 10
      periodSeconds: 10
      failureThreshold: 12
      timeoutSeconds: 5

    # Liveness: only check if the process is healthy (NOT dependencies)
    livenessProbe:
      httpGet:
        path: /actuator/health/liveness
        port: 8080
      periodSeconds: 15
      failureThreshold: 3
      timeoutSeconds: 5

    # Readiness: check if the service can handle traffic (includes dependencies)
    readinessProbe:
      httpGet:
        path: /actuator/health/readiness
        port: 8080
      periodSeconds: 5
      failureThreshold: 2
      successThreshold: 2
      timeoutSeconds: 3
```

**Parameter rationale:**
- Startup probe: `failureThreshold: 12` with `periodSeconds: 10` gives 120 seconds for the ML model to load (90s needed + 30s buffer).
- Liveness probe: `periodSeconds: 15` is not too aggressive. Only checks process health, not dependencies, to avoid cascading restarts when PostgreSQL is briefly unavailable.
- Readiness probe: `periodSeconds: 5` for quick response to dependency failures. `successThreshold: 2` prevents flapping.

```go
// Custom health check that avoids cascading failures
func readinessHandler(db *sql.DB, redisClient *redis.Client) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
        defer cancel()

        checks := map[string]string{}
        healthy := true

        // Check PostgreSQL with timeout
        dbCh := make(chan error, 1)
        go func() { dbCh <- db.PingContext(ctx) }()
        select {
        case err := <-dbCh:
            if err != nil {
                checks["postgresql"] = "down"
                healthy = false
            } else {
                checks["postgresql"] = "up"
            }
        case <-ctx.Done():
            checks["postgresql"] = "timeout"
            healthy = false
        }

        // Check Redis with timeout
        redisCh := make(chan error, 1)
        go func() { redisCh <- redisClient.Ping(ctx).Err() }()
        select {
        case err := <-redisCh:
            if err != nil {
                checks["redis"] = "down"
                healthy = false
            } else {
                checks["redis"] = "up"
            }
        case <-ctx.Done():
            checks["redis"] = "timeout"
            healthy = false
        }

        resp := HealthResponse{Checks: checks}
        if healthy {
            resp.Status = "ready"
            w.WriteHeader(http.StatusOK)
        } else {
            resp.Status = "not ready"
            w.WriteHeader(http.StatusServiceUnavailable)
        }
        json.NewEncoder(w).Encode(resp)
    }
}

// Liveness handler -- only checks process health, never dependencies
func livenessHandler() http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        // Only check that the process can respond
        w.WriteHeader(http.StatusOK)
        json.NewEncoder(w).Encode(HealthResponse{Status: "alive"})
    }
}
```

</details>

### Exercise 5: Production Debugging Scenario

A production deployment with 10 pods is experiencing intermittent 503 errors. About 20% of requests fail. Write the step-by-step debugging procedure: (a) commands to identify which pods are failing, (b) how to check if the issue is networking, application, or resource-related, (c) how to use ephemeral debug containers to inspect a running pod, (d) how to correlate logs, metrics, and traces to find the root cause. Provide all kubectl commands and PromQL queries.

<details>
<summary>Show Answer</summary>

```bash
# Step 1: Identify failing pods
# Check pod status and restarts
kubectl get pods -l app=my-app -o wide
kubectl get pods -l app=my-app -o custom-columns="NAME:.metadata.name,STATUS:.status.phase,RESTARTS:.status.containerStatuses[0].restartCount,NODE:.spec.nodeName,READY:.status.containerStatuses[0].ready"

# Check endpoints -- which pods are ready to receive traffic?
kubectl get endpoints my-app-service
kubectl get pods -l app=my-app -o jsonpath='{range .items[*]}{.metadata.name} ready={.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}'

# Step 2: Check resource usage
kubectl top pods -l app=my-app --sort-by=cpu
kubectl top pods -l app=my-app --sort-by=memory

# Check for OOM events
kubectl get events --field-selector reason=OOMKilling -A

# Step 3: Check recent logs from failing pods
for pod in $(kubectl get pods -l app=my-app -o jsonpath='{.items[*].metadata.name}'); do
  echo "=== $pod ==="
  kubectl logs "$pod" --tail=20 | grep -i error
done

# Step 4: Check networking
# Test connectivity from a debug pod to each application pod
kubectl run debug --image=nicolaka/netshoot --rm -it -- bash
# Inside: for i in $(seq 0 9); do curl -s -o /dev/null -w "%{http_code}\n" http://my-app-$i.my-app-headless:8080/healthz; done

# Step 5: Ephemeral debug container
kubectl debug -it <failing-pod> --image=nicolaka/netshoot --target=my-app
# Inside the debug container:
# ss -tlnp          # Check listening ports
# curl localhost:8080/healthz   # Test local health
# tcpdump -i eth0 port 8080 -c 20  # Capture traffic

# Step 6: Check node health for affected pods
kubectl describe node <node-of-failing-pods> | grep -A5 Conditions

# Step 7: Check readiness probe results
kubectl describe pod <failing-pod> | grep -A10 "Readiness"
```

```promql
# PromQL queries to correlate

# Per-pod error rate (identify the 20% that are failing)
sum by (pod) (rate(http_requests_total{app="my-app", status=~"5.."}[5m]))
/
sum by (pod) (rate(http_requests_total{app="my-app"}[5m]))

# Per-node error rate (check if issue is node-specific)
sum by (node) (rate(http_requests_total{app="my-app", status=~"5.."}[5m]))

# Memory usage approaching limits
container_memory_working_set_bytes{container="my-app"}
/
container_spec_memory_limit_bytes{container="my-app"}

# CPU throttling
rate(container_cpu_cfs_throttled_seconds_total{container="my-app"}[5m])

# Network errors
rate(container_network_receive_errors_total{pod=~"my-app.*"}[5m])
```

```logql
# Loki queries for log correlation

# Errors from the specific failing pods
{app="my-app", pod=~"my-app-pod-abc.*"} |= "error"

# Correlate with trace ID from a failed request
{app="my-app"} | json | status >= 500 | line_format "trace={{.trace_id}} status={{.status}} msg={{.msg}}"
```

**Common root causes for 20% failure rate:**
1. **2 of 10 pods are on a degraded node** -- check per-node error rate
2. **Pods hitting memory limits** -- check OOM events and memory usage ratio
3. **Readiness probe passing but app partially degraded** -- check probe endpoints vs actual traffic paths
4. **Connection pool exhaustion** -- check active connections metric and database logs

</details>

---

**Previous**: [Autoscaling](./13_Autoscaling.md) | **Next**: [Multi-Cluster](./15_Multi_Cluster.md)

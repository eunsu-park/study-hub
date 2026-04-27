# 13. Autoscaling

**Previous**: [Admission Controllers](./12_Admission_Controllers.md) | **Next**: [Observability](./14_Observability.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Configure Horizontal Pod Autoscaler (HPA) v2 with built-in, custom, and external metrics
2. Deploy and tune the Vertical Pod Autoscaler (VPA) for right-sizing workloads
3. Set up the Cluster Autoscaler and understand its decision-making process
4. Use KEDA to scale workloads based on event sources (queues, streams, databases)
5. Implement cost-aware and predictive scaling strategies for production environments

---

One of the core promises of Kubernetes is elastic scaling -- the ability to automatically adjust compute resources based on demand. But autoscaling in Kubernetes is not a single feature; it is a layered system with three distinct components that operate at different levels. Horizontal Pod Autoscaling adjusts the number of pod replicas, Vertical Pod Autoscaling adjusts resource requests and limits per container, and Cluster Autoscaling adjusts the number of nodes. Getting these layers to work together smoothly is essential for both cost efficiency and reliability.

## Table of Contents

- [Theory & Principles](#theory--principles)
- [1. Horizontal Pod Autoscaler (HPA) v2](#1-horizontal-pod-autoscaler-hpa-v2)
- [2. Custom Metrics and External Metrics](#2-custom-metrics-and-external-metrics)
- [3. Vertical Pod Autoscaler (VPA)](#3-vertical-pod-autoscaler-vpa)
- [4. Cluster Autoscaler](#4-cluster-autoscaler)
- [5. KEDA (Kubernetes Event-Driven Autoscaling)](#5-keda-kubernetes-event-driven-autoscaling)
- [6. Scaling with Prometheus Metrics](#6-scaling-with-prometheus-metrics)
- [7. Predictive Autoscaling](#7-predictive-autoscaling)
- [8. Cost-Aware Scaling](#8-cost-aware-scaling)
- [9. Scaling Best Practices](#9-scaling-best-practices)
- [Exercises](#exercises)

---

## 1. Horizontal Pod Autoscaler (HPA) v2

### Theory: HPA: A Closed-Loop Controller With a Specific Formula

HPA reconciles every 15 seconds (default `--horizontal-pod-autoscaler-sync-period`). Each cycle:

1. Read the current metric values for all pods backing the target Deployment via the Metrics API (default: `metrics.k8s.io` for CPU, `custom.metrics.k8s.io` for app metrics, `external.metrics.k8s.io` for queue length, etc.).
2. For each metric source, compute:
   ```
   desiredReplicas = ceil(currentReplicas × currentMetric / targetMetric)
   ```
   For example, if you target 50% CPU, current is 80% across 4 pods: `ceil(4 × 80 / 50) = 7`.
3. If multiple metrics are configured, take the **maximum** of the per-metric desired replicas (any metric over target → scale up).
4. Apply tolerance band: if the change is within ±10% of current, do nothing (avoid flapping).
5. Apply behavior policy: max scale-up rate per minute, stabilization windows, etc.
6. Write the new replica count to the Deployment.

Two consequences:

- **Scale-up is eager, scale-down is conservative.** The default scale-down stabilization window is 5 minutes — HPA waits to make sure the dip isn't transient before removing pods. Scale-up has a 0-second default — react immediately to load.
- **The formula is a P-controller (proportional only).** It cannot anticipate or smooth oscillating workloads on its own. Workloads with sharp daily patterns benefit from the `behavior` configuration (introduced in HPA v2) to bound rate of change.

A common gotcha: scaling on CPU when your bottleneck is something else (DB connections, queue depth) means HPA scales pods that are not actually CPU-bound, wasting capacity. Custom metrics (§A continued) or KEDA (§D) address this.

### 1.1 How HPA Works

The HPA controller runs a control loop (default every 15 seconds) that:

1. Queries metrics from the metrics API
2. Computes the desired replica count
3. Scales the target workload

```
                          ┌─────────────────────┐
                          │    Metrics Server    │
                          │  (or custom adapter) │
                          └──────────┬──────────┘
                                     │ metrics
                                     ▼
┌─────────────┐          ┌─────────────────────┐         ┌─────────────────┐
│  Target      │◀─────────│   HPA Controller    │────────▶│  Scale          │
│  Workload    │  observe  │                     │  scale   │  Subresource    │
│  (Deployment)│          │  desiredReplicas =   │         │  /scale         │
└─────────────┘          │  ceil(current *      │         └─────────────────┘
                          │   currentMetric /    │
                          │   desiredMetric)     │
                          └─────────────────────┘
```

### 1.2 HPA v2 API

The `autoscaling/v2` API supports multiple metric types simultaneously:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: web-app-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-app
  minReplicas: 3
  maxReplicas: 50
  metrics:
  # Resource metric (CPU)
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  # Resource metric (memory) -- absolute value
  - type: Resource
    resource:
      name: memory
      target:
        type: AverageValue
        averageValue: 500Mi
  # Pod metric (custom metric from application)
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
  # Object metric (from another Kubernetes object)
  - type: Object
    object:
      describedObject:
        apiVersion: networking.k8s.io/v1
        kind: Ingress
        name: web-app-ingress
      metric:
        name: requests_per_second
      target:
        type: Value
        value: "10000"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Pods
        value: 4
        periodSeconds: 60
      - type: Percent
        value: 100
        periodSeconds: 60
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 2
        periodSeconds: 120
      selectPolicy: Min
```

### 1.3 The Scaling Formula

```
desiredReplicas = ceil[currentReplicas * (currentMetricValue / desiredMetricValue)]
```

When multiple metrics are specified, the HPA calculates the desired replica count for each metric and takes the **maximum**:

```
finalReplicas = max(desiredFromCPU, desiredFromMemory, desiredFromCustom)
```

Example:
- Current replicas: 5
- Current CPU utilization: 90%
- Target CPU utilization: 70%
- Desired = ceil(5 * 90/70) = ceil(6.43) = 7

### 1.4 Scaling Behavior

The `behavior` field controls the rate and stability of scaling:

```yaml
behavior:
  scaleUp:
    stabilizationWindowSeconds: 0     # Scale up immediately
    policies:
    - type: Pods
      value: 10                        # Add max 10 pods per period
      periodSeconds: 60
    - type: Percent
      value: 200                       # Or double pods per period
      periodSeconds: 60
    selectPolicy: Max                  # Use the larger of the two

  scaleDown:
    stabilizationWindowSeconds: 300    # Wait 5 min of sustained low metrics
    policies:
    - type: Pods
      value: 2                         # Remove max 2 pods per period
      periodSeconds: 120
    selectPolicy: Min                  # Use the smaller value (conservative)
```

### 1.5 Prerequisites

```bash
# HPA requires metrics-server for resource metrics (CPU/memory)
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Verify metrics-server is running
kubectl get deployment metrics-server -n kube-system
kubectl top pods
kubectl top nodes
```

### 1.6 HPA Commands

```bash
# Create a simple HPA
kubectl autoscale deployment web-app --cpu-percent=70 --min=3 --max=50

# View HPA status
kubectl get hpa web-app-hpa
# NAME          REFERENCE            TARGETS          MINPODS   MAXPODS   REPLICAS   AGE
# web-app-hpa   Deployment/web-app   45%/70%, 350Mi/500Mi   3         50        5          2h

# Describe HPA for detailed status and events
kubectl describe hpa web-app-hpa

# Check HPA conditions
kubectl get hpa web-app-hpa -o jsonpath='{.status.conditions[*].type}'
# AbleToScale ScalingActive ScalingLimited
```

---

## 2. Custom Metrics and External Metrics

### Theory: Metrics Pipeline: The Latency That Bounds Reactivity

HPA is only as fast as the metric it watches. The default pipeline:

```
Pod cgroup → kubelet (every 10s, default --housekeeping-interval)
          → metrics-server (scrapes kubelets every 60s by default)
          → Metrics API (HPA reads via aggregated API)
```

So even with HPA's 15s reconcile, the freshest CPU measurement HPA sees can be **60+ seconds old**. The end-to-end "load arrives → pods scale up" lag in default config is typically 60–120 seconds, not 15.

For tighter loops, you reduce the metrics-server scrape interval (`--metric-resolution=15s`), but at the cost of more API server load. Alternatively, custom metrics adapters (Prometheus Adapter, Datadog) read directly from your monitoring system, which may already have shorter scrape intervals.

For external sources (queue length, DB connection count), the Custom Metrics API or External Metrics API exposes them to HPA. Prometheus Adapter is the most common bridge: write your metric to Prometheus, configure the adapter to expose it as `custom.metrics.k8s.io/v1beta1/<resource>/<metric>`, reference it in HPA. This is how you scale on requests-per-second-per-pod, queue depth, P99 latency, or any business KPI.

### 2.1 Metrics API Architecture

```
                    ┌──────────────────────┐
                    │    HPA Controller     │
                    └────────┬─────────────┘
                             │ queries
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
    ┌──────────────┐ ┌──────────────┐ ┌────────────────┐
    │ metrics.k8s.io│ │custom.metrics│ │external.metrics│
    │ (Resource)    │ │.k8s.io      │ │.k8s.io         │
    └──────┬───────┘ └──────┬───────┘ └──────┬─────────┘
           │                │                │
           ▼                ▼                ▼
    ┌──────────────┐ ┌──────────────┐ ┌────────────────┐
    │metrics-server│ │  Prometheus  │ │  External API  │
    │              │ │  Adapter     │ │  Adapter       │
    └──────────────┘ └──────────────┘ └────────────────┘
```

### 2.2 Metric Types

| Type | API Group | Source | Example |
|---|---|---|---|
| Resource | `metrics.k8s.io` | kubelet cAdvisor | CPU, memory per pod |
| Pod | `custom.metrics.k8s.io` | Application metrics via adapter | `http_requests_per_second` |
| Object | `custom.metrics.k8s.io` | Metrics on K8s objects | Ingress request rate |
| External | `external.metrics.k8s.io` | Metrics outside the cluster | SQS queue depth, Pub/Sub backlog |

### 2.3 Prometheus Adapter

The Prometheus adapter bridges Prometheus metrics to the Kubernetes custom metrics API:

```bash
# Install Prometheus adapter
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus-adapter prometheus-community/prometheus-adapter \
  --namespace monitoring \
  --set prometheus.url=http://prometheus-server.monitoring.svc \
  --set prometheus.port=9090
```

Adapter configuration to expose application metrics:

```yaml
# prometheus-adapter-config.yaml
rules:
# Map Prometheus http_requests_total to custom.metrics.k8s.io
- seriesQuery: 'http_requests_total{namespace!="",pod!=""}'
  resources:
    overrides:
      namespace: {resource: "namespace"}
      pod: {resource: "pod"}
  name:
    matches: "^(.*)_total$"
    as: "${1}_per_second"
  metricsQuery: 'rate(<<.Series>>{<<.LabelMatchers>>}[2m])'

# Map queue depth
- seriesQuery: 'rabbitmq_queue_messages{namespace!="",service!=""}'
  resources:
    overrides:
      namespace: {resource: "namespace"}
      service: {resource: "service"}
  name:
    as: "queue_messages"
  metricsQuery: '<<.Series>>{<<.LabelMatchers>>}'
```

### 2.4 Verifying Custom Metrics

```bash
# List available custom metrics
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1 | jq '.resources[].name'

# Query a specific metric
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1/namespaces/production/pods/*/http_requests_per_second" | jq .

# List external metrics
kubectl get --raw /apis/external.metrics.k8s.io/v1beta1 | jq '.resources[].name'
```

---

## 3. Vertical Pod Autoscaler (VPA)

### Theory: VPA and Cluster Autoscaler: The Other Two Layers

**VPA (Vertical Pod Autoscaler)** observes historical resource usage and recommends `requests` and `limits` per container. Three modes:

- `Off`: only computes recommendations (visible via `vpa.status.recommendation`); a human reads them.
- `Initial`: applies recommendations only at pod creation; existing pods keep their settings.
- `Auto`: evicts pods to apply new recommendations (with PDB respect). Disruptive but fully automated.

VPA and HPA on the same workload using the same metric (CPU) is a known footgun — they fight each other. Use VPA for memory + HPA for CPU, or use VPA in `Off` mode to inform manual sizing while HPA scales horizontally.

VPA's other role is **right-sizing batch and stateful workloads** that can't horizontally scale. A database that needs 8 GB sometimes and 32 GB during nightly batch is a perfect VPA candidate (in Off + manual mode for prod, or Initial mode if restarts are tolerable).

**Cluster Autoscaler (CA)** watches for unschedulable pods. When the scheduler reports `Pending` because no node has room, CA simulates: "if I added a node from group X, would this pod fit?" If yes, CA asks the cloud provider to provision the node (via the cloud's auto-scaling group / managed node group / VM scale set). When nodes have low utilization for `--scale-down-unneeded-time` (default 10 min) and their pods can fit elsewhere, CA cordons and drains them, then asks the cloud to remove them.

CA does *not* look at CPU or memory directly; it looks at *requests vs allocatable*. So if your pods have low CPU requests but high actual usage, CA won't add nodes — but pods will get throttled because they're pinned to overcommitted nodes. **Right-sizing requests is what makes CA work.**

### 3.1 What is VPA?

VPA automatically adjusts the CPU and memory **requests and limits** of containers based on observed usage. Instead of adding more pods, it right-sizes existing pods.

### 3.2 VPA Components

```
┌─────────────────────────────────────────────────┐
│                VPA System                        │
│                                                  │
│  ┌──────────────┐  ┌──────────────────────────┐  │
│  │   Recommender │  │     Admission Controller │  │
│  │  (analyzes    │  │  (applies recommendations│  │
│  │   metrics)    │  │   at pod creation)       │  │
│  └──────┬───────┘  └──────────┬───────────────┘  │
│         │                      │                  │
│         ▼                      ▼                  │
│  ┌──────────────┐  ┌──────────────────────────┐  │
│  │   Updater    │  │     VPA Object           │  │
│  │  (evicts pods │  │  (stores recommendations │  │
│  │   for resize) │  │   and policy)            │  │
│  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### 3.3 Installation

```bash
# Clone and install VPA
git clone https://github.com/kubernetes/autoscaler.git
cd autoscaler/vertical-pod-autoscaler
./hack/vpa-up.sh

# Verify
kubectl get pods -n kube-system | grep vpa
```

### 3.4 VPA Configuration

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: web-app-vpa
  namespace: production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-app
  updatePolicy:
    updateMode: Auto   # Off, Initial, Recreate, or Auto
  resourcePolicy:
    containerPolicies:
    - containerName: web-app
      minAllowed:
        cpu: 100m
        memory: 128Mi
      maxAllowed:
        cpu: 4
        memory: 8Gi
      controlledResources: ["cpu", "memory"]
      controlledValues: RequestsAndLimits  # or RequestsOnly
    - containerName: sidecar
      mode: "Off"  # Do not adjust sidecar resources
```

### 3.5 Update Modes

| Mode | Behavior | Use Case |
|---|---|---|
| `Off` | Only provides recommendations, no changes | Observation phase |
| `Initial` | Sets resources only at pod creation | Stable workloads, avoid restarts |
| `Recreate` | Evicts and recreates pods to apply changes | General use |
| `Auto` | Currently same as Recreate, may support in-place in future | Recommended default |

### 3.6 Reading VPA Recommendations

```bash
# Get VPA recommendations
kubectl describe vpa web-app-vpa

# Example output:
# Recommendation:
#   Container Recommendations:
#     Container Name: web-app
#     Lower Bound:
#       Cpu:     100m
#       Memory:  256Mi
#     Target:
#       Cpu:     350m
#       Memory:  512Mi
#     Uncapped Target:
#       Cpu:     350m
#       Memory:  512Mi
#     Upper Bound:
#       Cpu:     2
#       Memory:  2Gi
```

### 3.7 VPA and HPA Interaction

**Do not use VPA and HPA on the same metric (CPU).** They will conflict:

| Combination | Works? | Notes |
|---|---|---|
| HPA on CPU + VPA on CPU | No | Both try to control CPU, creates oscillation |
| HPA on custom metric + VPA on CPU/memory | Yes | Different control dimensions |
| HPA + VPA in `Off` mode | Yes | VPA only recommends, does not act |

---

## 4. Cluster Autoscaler

### 4.1 What is Cluster Autoscaler?

While HPA and VPA adjust workloads, the Cluster Autoscaler adjusts the number of **nodes** in the cluster. It adds nodes when pods are unschedulable and removes underutilized nodes.

### 4.2 Scale-Up Decision

```
Pod pending (unschedulable)
    │
    ▼
Cluster Autoscaler checks:
    1. Is there a node group that could fit this pod?
    2. Would adding a node make the pod schedulable?
    3. Is the node group below its max size?
    │
    ▼ (all yes)
Request new node from cloud provider
    │
    ▼
Wait for node to join the cluster (1-5 min)
    │
    ▼
Scheduler places pending pod on new node
```

### 4.3 Scale-Down Decision

```
Node utilization check (every 10s)
    │
    ▼
Is node utilization < 50% (default)?
    │ yes
    ▼
Can all pods be moved to other nodes?
    │ yes
    ▼
Are there any blocking conditions?
    - PDB would be violated?
    - Pod with local storage?
    - Pod without controller (bare pod)?
    - Pod with "cluster-autoscaler.kubernetes.io/safe-to-evict: false"?
    │ no blockers
    ▼
Wait 10 minutes (scale-down-unneeded-time)
    │
    ▼
Drain and delete node
```

### 4.4 Installation (AWS EKS Example)

```bash
# Install Cluster Autoscaler on EKS
helm repo add autoscaler https://kubernetes.github.io/autoscaler
helm install cluster-autoscaler autoscaler/cluster-autoscaler \
  --namespace kube-system \
  --set autoDiscovery.clusterName=my-cluster \
  --set awsRegion=us-west-2 \
  --set extraArgs.balance-similar-node-groups=true \
  --set extraArgs.skip-nodes-with-local-storage=false \
  --set extraArgs.expander=least-waste \
  --set extraArgs.scale-down-utilization-threshold=0.5 \
  --set extraArgs.scale-down-unneeded-time=10m \
  --set extraArgs.scale-down-delay-after-add=10m
```

### 4.5 Node Group Configuration

```yaml
# AWS Auto Scaling Group tags for auto-discovery
# k8s.io/cluster-autoscaler/enabled: true
# k8s.io/cluster-autoscaler/my-cluster: owned

# Priority-based expander configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-priority-expander
  namespace: kube-system
data:
  priorities: |
    50:
    - name: spot-node-group.*
    30:
    - name: on-demand-node-group.*
    10:
    - name: gpu-node-group.*
```

### 4.6 Pod Annotations for Cluster Autoscaler

```yaml
metadata:
  annotations:
    # Tell CA this pod is safe to evict (for scale-down)
    cluster-autoscaler.kubernetes.io/safe-to-evict: "true"

    # Tell CA this pod is NOT safe to evict
    cluster-autoscaler.kubernetes.io/safe-to-evict: "false"
```

---

## 5. KEDA (Kubernetes Event-Driven Autoscaling)

### Theory: KEDA: Event-Driven Scaling for the Cases HPA Misses

HPA scales based on metrics that pods *expose* (CPU, memory, custom). But many workloads should scale based on **external events** the pods don't know about:

- A queue has 10,000 messages → spin up consumers.
- A Kafka topic has consumer lag → add more partitioned consumers.
- A scheduled batch job needs 100 pods at 02:00 → scale before the work arrives.

KEDA (Kubernetes Event-Driven Autoscaling) is a CRD + controller that bridges these external sources to HPA. You define a `ScaledObject`:

```yaml
kind: ScaledObject
metadata: { name: rabbitmq-consumer }
spec:
  scaleTargetRef: { name: my-consumer }
  minReplicaCount: 0
  maxReplicaCount: 100
  triggers:
    - type: rabbitmq
      metadata:
        host: amqp://...
        queueName: jobs
        queueLength: "5"     # 1 pod per 5 messages
```

KEDA polls RabbitMQ, computes `desiredReplicas = ceil(queueLength / 5)`, and exposes that to a generated HPA. The crucial extra capability: **scale to zero**. HPA cannot scale below 1; KEDA can scale to 0 when the queue is empty and back up when messages arrive (creating the first pod via the operator pattern). This is huge for cost on bursty workloads — pay only for the time work is actually running.

KEDA has 60+ scalers (RabbitMQ, Kafka, AWS SQS, Postgres queries, Cron, Prometheus, ...). For workloads where load is event-shaped rather than CPU-shaped, KEDA replaces HPA-on-custom-metrics with a vastly simpler config.

### 5.1 What is KEDA?

KEDA extends Kubernetes autoscaling to event-driven workloads. It can scale based on event sources like message queues, databases, cron schedules, and cloud services.

### 5.2 Architecture

```
              ┌──────────────────────────┐
              │     Event Sources         │
              │  (RabbitMQ, Kafka, SQS,  │
              │   Redis, Prometheus...)   │
              └────────────┬─────────────┘
                           │ poll
                           ▼
              ┌──────────────────────────┐
              │      KEDA Operator       │
              │  ┌────────────────────┐  │
              │  │   Metrics Server   │  │  expose metrics
              │  │   (custom metrics) │──────────────┐
              │  └────────────────────┘  │            │
              │  ┌────────────────────┐  │            ▼
              │  │   Controller       │  │     ┌─────────────┐
              │  │  (scale to/from 0) │──────▶│     HPA     │
              │  └────────────────────┘  │     └─────────────┘
              └──────────────────────────┘            │
                                                      ▼
                                              ┌───────────────┐
                                              │  Deployment   │
                                              │  (0 → N pods) │
                                              └───────────────┘
```

### 5.3 Installation

```bash
# Install KEDA using Helm
helm repo add kedacore https://kedacore.github.io/charts
helm install keda kedacore/keda \
  --namespace keda \
  --create-namespace \
  --set watchNamespace="" \
  --set operator.replicaCount=2
```

### 5.4 ScaledObject for Queue-Based Scaling

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: order-processor
  namespace: production
spec:
  scaleTargetRef:
    name: order-processor
  pollingInterval: 15           # Check every 15 seconds
  cooldownPeriod: 300           # Wait 5 min before scale-to-zero
  idleReplicaCount: 0           # Scale to zero when idle
  minReplicaCount: 0            # Minimum replicas
  maxReplicaCount: 100          # Maximum replicas
  fallback:
    failureThreshold: 3
    replicas: 5                  # Fallback replica count if scaler fails
  triggers:
  - type: rabbitmq
    metadata:
      host: amqp://guest:guest@rabbitmq.default.svc:5672/
      queueName: orders
      queueLength: "10"          # 1 pod per 10 messages
    authenticationRef:
      name: rabbitmq-credentials
```

### 5.5 ScaledObject for Kafka

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: kafka-consumer
  namespace: production
spec:
  scaleTargetRef:
    name: kafka-consumer
  minReplicaCount: 1
  maxReplicaCount: 30
  triggers:
  - type: kafka
    metadata:
      bootstrapServers: kafka-broker.kafka.svc:9092
      consumerGroup: order-group
      topic: orders
      lagThreshold: "100"         # Scale when lag > 100 per partition
      offsetResetPolicy: latest
      allowIdleConsumers: "false"
      scaleToZeroOnInvalidOffset: "false"
```

### 5.6 ScaledObject with Cron Trigger

```yaml
# Predictive scaling based on schedule
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: web-frontend
  namespace: production
spec:
  scaleTargetRef:
    name: web-frontend
  minReplicaCount: 3
  maxReplicaCount: 100
  triggers:
  # Prometheus-based reactive scaling
  - type: prometheus
    metadata:
      serverAddress: http://prometheus-server.monitoring.svc:9090
      metricName: http_requests_per_second
      query: sum(rate(http_requests_total{deployment="web-frontend"}[2m]))
      threshold: "500"

  # Cron-based proactive scaling for known traffic patterns
  - type: cron
    metadata:
      timezone: America/New_York
      start: 0 8 * * 1-5          # Mon-Fri 8:00 AM
      end: 0 20 * * 1-5           # Mon-Fri 8:00 PM
      desiredReplicas: "20"        # Pre-scale for business hours
```

### 5.7 ScaledJob for Batch Processing

```yaml
# KEDA can also scale Jobs (not just Deployments)
apiVersion: keda.sh/v1alpha1
kind: ScaledJob
metadata:
  name: image-processor
  namespace: production
spec:
  jobTargetRef:
    template:
      spec:
        containers:
        - name: processor
          image: example.com/image-processor:v1
          env:
          - name: QUEUE_URL
            value: "https://sqs.us-west-2.amazonaws.com/123456/images"
        restartPolicy: Never
    backoffLimit: 3
  pollingInterval: 10
  maxReplicaCount: 50
  successfulJobsHistoryLimit: 10
  failedJobsHistoryLimit: 5
  scalingStrategy:
    strategy: accurate
  triggers:
  - type: aws-sqs-queue
    metadata:
      queueURL: https://sqs.us-west-2.amazonaws.com/123456/images
      queueLength: "5"
      awsRegion: us-west-2
    authenticationRef:
      name: aws-credentials
```

---

## 6. Scaling with Prometheus Metrics

### 6.1 Application Metrics for HPA

Expose metrics from your application for HPA consumption:

```go
// Go application with Prometheus metrics
package main

import (
    "net/http"

    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
    httpRequestsTotal = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "http_requests_total",
            Help: "Total number of HTTP requests",
        },
        []string{"method", "path", "status"},
    )
    httpRequestDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "http_request_duration_seconds",
            Help:    "HTTP request duration in seconds",
            Buckets: prometheus.DefBuckets,
        },
        []string{"method", "path"},
    )
    activeConnections = promauto.NewGauge(
        prometheus.GaugeOpts{
            Name: "active_connections",
            Help: "Number of active connections",
        },
    )
    queueDepth = promauto.NewGauge(
        prometheus.GaugeOpts{
            Name: "queue_depth",
            Help: "Number of items in the processing queue",
        },
    )
)

func main() {
    http.Handle("/metrics", promhttp.Handler())
    http.HandleFunc("/api/", func(w http.ResponseWriter, r *http.Request) {
        activeConnections.Inc()
        defer activeConnections.Dec()
        // handler logic
        httpRequestsTotal.WithLabelValues(r.Method, r.URL.Path, "200").Inc()
    })
    http.ListenAndServe(":8080", nil)
}
```

### 6.2 Prometheus Adapter Rules

```yaml
# Map application metrics to HPA-consumable metrics
rules:
- seriesQuery: 'http_requests_total{namespace!="",pod!=""}'
  resources:
    overrides:
      namespace: {resource: "namespace"}
      pod: {resource: "pod"}
  name:
    matches: "^(.*)_total$"
    as: "${1}_per_second"
  metricsQuery: 'sum(rate(<<.Series>>{<<.LabelMatchers>>}[2m])) by (<<.GroupBy>>)'

- seriesQuery: 'active_connections{namespace!="",pod!=""}'
  resources:
    overrides:
      namespace: {resource: "namespace"}
      pod: {resource: "pod"}
  name:
    as: "active_connections"
  metricsQuery: '<<.Series>>{<<.LabelMatchers>>}'

- seriesQuery: 'queue_depth{namespace!="",pod!=""}'
  resources:
    overrides:
      namespace: {resource: "namespace"}
      pod: {resource: "pod"}
  name:
    as: "queue_depth"
  metricsQuery: 'avg(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)'
```

### 6.3 HPA Using Prometheus Metrics

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: web-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-app
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "500"
  - type: Pods
    pods:
      metric:
        name: active_connections
      target:
        type: AverageValue
        averageValue: "100"
```

---

## 7. Predictive Autoscaling

### 7.1 The Problem with Reactive Scaling

Reactive autoscaling (traditional HPA) has an inherent latency:

```
Traffic spike arrives
    │
    ▼  (15s) HPA scrape interval
HPA detects increased metric
    │
    ▼  (seconds) HPA computation
Scale decision made
    │
    ▼  (30-120s) Pod scheduling + startup
New pods ready to serve
    │
    Total delay: 45s - 3 min
```

For workloads with predictable traffic patterns, this delay causes degraded performance during ramp-up.

### 7.2 KEDA Cron-Based Pre-Scaling

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: predictive-web
spec:
  scaleTargetRef:
    name: web-frontend
  minReplicaCount: 3
  maxReplicaCount: 100
  triggers:
  # Reactive: handle unexpected spikes
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring.svc:9090
      metricName: http_rps
      query: sum(rate(http_requests_total{app="web-frontend"}[2m]))
      threshold: "500"

  # Predictive: pre-scale for known patterns
  - type: cron
    metadata:
      timezone: UTC
      start: 30 7 * * 1-5      # Pre-scale at 7:30 AM weekdays
      end: 0 9 * * 1-5          # Hold until 9 AM
      desiredReplicas: "30"

  - type: cron
    metadata:
      timezone: UTC
      start: 30 11 * * 1-5     # Lunch spike
      end: 0 14 * * 1-5
      desiredReplicas: "25"

  - type: cron
    metadata:
      timezone: UTC
      start: 0 0 25 11 *       # Black Friday
      end: 0 0 27 11 *
      desiredReplicas: "80"
```

### 7.3 Combining HPA with PodDisruptionBudget

Prevent scale-down from causing service disruption:

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: web-app-pdb
spec:
  minAvailable: "80%"
  selector:
    matchLabels:
      app: web-app
```

---

## 8. Cost-Aware Scaling

### 8.1 Right-Sizing with VPA Recommendations

```bash
# Use VPA in Off mode to collect recommendations without applying them
kubectl get vpa web-app-vpa -o jsonpath='{.status.recommendation.containerRecommendations[0]}' | jq .

# Use goldilocks to get VPA recommendations for all workloads
# https://github.com/FairwindsOps/goldilocks
helm install goldilocks fairwinds-stable/goldilocks --namespace goldilocks --create-namespace
```

### 8.2 Spot/Preemptible Instances

```yaml
# Node affinity for cost-aware scheduling
apiVersion: apps/v1
kind: Deployment
metadata:
  name: batch-processor
spec:
  replicas: 10
  template:
    spec:
      affinity:
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 90
            preference:
              matchExpressions:
              - key: node.kubernetes.io/instance-type
                operator: In
                values: ["spot"]
          - weight: 10
            preference:
              matchExpressions:
              - key: node.kubernetes.io/instance-type
                operator: In
                values: ["on-demand"]
      tolerations:
      - key: "kubernetes.io/spot"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
      containers:
      - name: processor
        image: example.com/batch-processor:v1
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: "1"
            memory: 1Gi
```

### 8.3 Cluster Autoscaler Priority Expander

Configure the Cluster Autoscaler to prefer cheaper node groups:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-priority-expander
  namespace: kube-system
data:
  priorities: |
    100:
    - name: spot-.*          # Highest priority: spot instances
    50:
    - name: on-demand-small-.*  # Medium: small on-demand
    10:
    - name: on-demand-large-.*  # Last resort: large on-demand
```

### 8.4 Scale-Down Optimization

```bash
# Aggressive scale-down for non-production
--scale-down-unneeded-time=3m
--scale-down-utilization-threshold=0.3
--scale-down-delay-after-add=3m

# Conservative scale-down for production
--scale-down-unneeded-time=15m
--scale-down-utilization-threshold=0.5
--scale-down-delay-after-add=15m
--scale-down-delay-after-failure=5m
```

---

## 9. Scaling Best Practices

### 9.1 Resource Requests Are Critical

HPA calculates utilization as a percentage of resource **requests**, not limits. Setting requests too high leads to under-scaling; too low leads to over-scaling.

```yaml
# BAD: requests too high, HPA will never trigger
resources:
  requests:
    cpu: "4"        # Using 300m of 4 cores = 7.5% utilization
  limits:
    cpu: "4"

# GOOD: requests reflect actual steady-state usage
resources:
  requests:
    cpu: 500m       # Using 300m of 500m = 60% utilization
  limits:
    cpu: "2"
```

### 9.2 Scaling Checklist

| Practice | Recommendation |
|---|---|
| Set resource requests on all containers | Required for HPA CPU/memory metrics |
| Use readiness probes | Prevents routing traffic to unready pods |
| Use PodDisruptionBudgets | Prevents scale-down from disrupting service |
| Start with conservative behavior | Long stabilization windows, slow scale-down |
| Monitor HPA decisions | Check `kubectl describe hpa` for scaling events |
| Avoid HPA + VPA on same metric | Use HPA on custom metrics + VPA on resources |
| Set maxReplicas thoughtfully | Consider node capacity and cost budget |
| Test scaling under load | Use tools like k6 or Locust for load testing |

### 9.3 Load Testing Autoscaling

```bash
# Use k6 to test autoscaling behavior
cat > load-test.js <<'EOF'
import http from 'k6/http';
import { sleep } from 'k6';

export const options = {
  stages: [
    { duration: '2m', target: 100 },   // Ramp up to 100 users
    { duration: '5m', target: 100 },   // Stay at 100 users
    { duration: '2m', target: 500 },   // Spike to 500 users
    { duration: '5m', target: 500 },   // Stay at 500 users
    { duration: '5m', target: 0 },     // Ramp down
  ],
};

export default function () {
  http.get('http://web-app.production.svc/api/health');
  sleep(1);
}
EOF

kubectl run k6 --image=grafana/k6 --rm -it --restart=Never -- run - < load-test.js
```

Monitor during the load test:

```bash
# Watch HPA in real-time
kubectl get hpa -w

# Watch pod count
kubectl get pods -l app=web-app -w

# Watch node count
kubectl get nodes -w
```

---

## Exercises

### Exercise 1: Multi-Metric HPA

Create an HPA for a web application Deployment named `api-server` that scales based on three metrics simultaneously: (a) CPU utilization target of 60%, (b) memory utilization target of 75%, (c) a custom metric `http_requests_per_second` with a target average value of 200 per pod. Set minimum replicas to 2 and maximum to 30. Configure behavior so that scale-up can add at most 5 pods per minute, and scale-down removes at most 1 pod every 2 minutes with a 5-minute stabilization window.

<details>
<summary>Show Answer</summary>

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-server-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-server
  minReplicas: 2
  maxReplicas: 30
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 75
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "200"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
      - type: Pods
        value: 5
        periodSeconds: 60
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 120
      selectPolicy: Min
```

Verification:

```bash
kubectl apply -f api-server-hpa.yaml
kubectl get hpa api-server-hpa
kubectl describe hpa api-server-hpa
```

</details>

### Exercise 2: VPA Configuration

Configure VPA for a Java application that has unpredictable memory usage. The Deployment has two containers: `app` (Java) and `envoy-proxy` (sidecar). Requirements: (a) VPA should only manage the `app` container, (b) CPU should be between 250m and 4 cores, (c) memory between 512Mi and 8Gi, (d) the sidecar should not be modified, (e) use `Initial` mode so existing pods are not restarted. Also write the command to check the current recommendation.

<details>
<summary>Show Answer</summary>

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: java-app-vpa
  namespace: production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: java-app
  updatePolicy:
    updateMode: Initial
  resourcePolicy:
    containerPolicies:
    - containerName: app
      minAllowed:
        cpu: 250m
        memory: 512Mi
      maxAllowed:
        cpu: "4"
        memory: 8Gi
      controlledResources: ["cpu", "memory"]
      controlledValues: RequestsAndLimits
    - containerName: envoy-proxy
      mode: "Off"
```

Check the recommendation:

```bash
# View current VPA recommendation
kubectl get vpa java-app-vpa -n production -o yaml | \
  yq '.status.recommendation.containerRecommendations'

# Or with kubectl describe
kubectl describe vpa java-app-vpa -n production

# Check that only the 'app' container is being recommended
kubectl get vpa java-app-vpa -n production \
  -o jsonpath='{range .status.recommendation.containerRecommendations[*]}{.containerName}: CPU={.target.cpu}, Memory={.target.memory}{"\n"}{end}'
```

</details>

### Exercise 3: KEDA with Multiple Triggers

Create a KEDA ScaledObject for an `order-processor` Deployment that scales based on: (a) RabbitMQ queue depth (1 pod per 5 messages in the `orders` queue), (b) a cron schedule that pre-scales to 10 replicas during business hours (9 AM - 6 PM EST, weekdays), (c) a Prometheus metric fallback that scales when processing latency exceeds 500ms. The system should scale to zero during off-hours with a 5-minute cooldown. Include the TriggerAuthentication for RabbitMQ.

<details>
<summary>Show Answer</summary>

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: rabbitmq-secret
  namespace: production
data:
  host: YW1xcDovL3VzZXI6cGFzc0ByYWJiaXRtcS5wcm9kdWN0aW9uLnN2Yzo1NjcyLw==
---
apiVersion: keda.sh/v1alpha1
kind: TriggerAuthentication
metadata:
  name: rabbitmq-auth
  namespace: production
spec:
  secretTargetRef:
  - parameter: host
    name: rabbitmq-secret
    key: host
---
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: order-processor
  namespace: production
spec:
  scaleTargetRef:
    name: order-processor
  pollingInterval: 10
  cooldownPeriod: 300
  idleReplicaCount: 0
  minReplicaCount: 0
  maxReplicaCount: 50
  fallback:
    failureThreshold: 3
    replicas: 5
  triggers:
  # RabbitMQ queue depth
  - type: rabbitmq
    metadata:
      queueName: orders
      queueLength: "5"
      protocol: amqp
    authenticationRef:
      name: rabbitmq-auth

  # Cron: pre-scale during business hours
  - type: cron
    metadata:
      timezone: America/New_York
      start: 0 9 * * 1-5
      end: 0 18 * * 1-5
      desiredReplicas: "10"

  # Prometheus: scale on processing latency
  - type: prometheus
    metadata:
      serverAddress: http://prometheus-server.monitoring.svc:9090
      metricName: order_processing_latency
      query: |
        histogram_quantile(0.95,
          rate(order_processing_duration_seconds_bucket{deployment="order-processor"}[5m])
        )
      threshold: "0.5"
      activationThreshold: "0.3"
```

Verification:

```bash
kubectl get scaledobject order-processor -n production
kubectl get hpa -n production  # KEDA creates an HPA under the hood
kubectl describe scaledobject order-processor -n production
```

</details>

### Exercise 4: Cluster Autoscaler Troubleshooting

A cluster has 3 node groups (spot-small, spot-large, on-demand). Pods are pending but the Cluster Autoscaler is not scaling up. Write the commands to: (a) check the Cluster Autoscaler status ConfigMap, (b) view the CA logs for scale-up decisions, (c) check if the pending pod has scheduling constraints that prevent placement, (d) verify node group configuration. Then describe three common reasons CA fails to scale up and their solutions.

<details>
<summary>Show Answer</summary>

```bash
# (a) Check CA status ConfigMap
kubectl get configmap cluster-autoscaler-status -n kube-system -o yaml

# (b) View CA logs for scale-up decisions
kubectl logs -n kube-system -l app.kubernetes.io/name=cluster-autoscaler --tail=200 | grep -E "Scale|scale|pending|unschedulable"

# (c) Check pending pod details
kubectl get pods --field-selector=status.phase=Pending -A
kubectl describe pod <pending-pod> -n <namespace>
# Look for Events section, especially "FailedScheduling" with reason

# (d) Verify node group configuration
kubectl get nodes --show-labels | grep node.kubernetes.io/instance-type
kubectl get nodes -o custom-columns="NAME:.metadata.name,CAPACITY_CPU:.status.capacity.cpu,CAPACITY_MEM:.status.capacity.memory,ALLOCATABLE_CPU:.status.allocatable.cpu"
```

**Three common reasons CA fails to scale up:**

1. **Resource request exceeds node capacity**: The pending pod requests more CPU/memory than any node in any node group can provide. Solution: Create a node group with larger instance types, or reduce the pod's resource requests.

2. **Node group at maximum size**: The node group that could fit the pod has already reached its configured maximum. Solution: Increase the `maxSize` of the node group, or add additional node groups.

3. **Pod has unsatisfiable constraints**: The pod specifies `nodeSelector`, `nodeAffinity`, or `tolerations` that no node group can satisfy. Solution: Add labels to node group launch templates, or add taints that match the pod's tolerations. Check with:

```bash
# Check what affinity/selectors the pod requires
kubectl get pod <pending-pod> -o jsonpath='{.spec.nodeSelector}'
kubectl get pod <pending-pod> -o jsonpath='{.spec.affinity}'
kubectl get pod <pending-pod> -o jsonpath='{.spec.tolerations}'

# Compare with available node labels
kubectl get nodes --show-labels
```

</details>

### Exercise 5: End-to-End Autoscaling Design

Design an autoscaling strategy for a microservices application with three tiers: (a) an API gateway (latency-sensitive, must respond in <100ms), (b) a background job processor (consumes from an SQS queue, can tolerate 30s delays), (c) a data pipeline (runs overnight, cost-sensitive). For each tier, specify: the autoscaler type (HPA/VPA/KEDA/CA), the metrics to scale on, min/max replicas, scaling behavior, and node type (on-demand vs spot). Write the HPA/KEDA manifests for all three tiers.

<details>
<summary>Show Answer</summary>

**Design:**

| Tier | Autoscaler | Metrics | Min/Max | Node Type |
|---|---|---|---|---|
| API Gateway | HPA v2 | CPU (60%), p99 latency (<80ms) | 5/100 | On-demand |
| Job Processor | KEDA | SQS queue depth | 0/50 | Spot |
| Data Pipeline | KEDA (cron + ScaledJob) | Cron schedule | 0/30 | Spot |

```yaml
# (a) API Gateway - HPA with aggressive scale-up
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-gateway-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-gateway
  minReplicas: 5
  maxReplicas: 100
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
  - type: Pods
    pods:
      metric:
        name: http_request_duration_p99
      target:
        type: AverageValue
        averageValue: "80m"  # 80ms target, scale before hitting 100ms SLO
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0  # Scale up immediately
      policies:
      - type: Percent
        value: 100                    # Double pods instantly if needed
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60
---
# (b) Job Processor - KEDA with SQS
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: job-processor
  namespace: production
spec:
  scaleTargetRef:
    name: job-processor
  pollingInterval: 15
  cooldownPeriod: 300
  idleReplicaCount: 0
  minReplicaCount: 0
  maxReplicaCount: 50
  triggers:
  - type: aws-sqs-queue
    metadata:
      queueURL: https://sqs.us-west-2.amazonaws.com/123456/jobs
      queueLength: "10"
      awsRegion: us-west-2
    authenticationRef:
      name: aws-credentials
---
# (c) Data Pipeline - KEDA ScaledJob with cron
apiVersion: keda.sh/v1alpha1
kind: ScaledJob
metadata:
  name: data-pipeline
  namespace: batch
spec:
  jobTargetRef:
    template:
      spec:
        containers:
        - name: pipeline
          image: example.com/data-pipeline:v1
          resources:
            requests:
              cpu: "2"
              memory: 4Gi
        restartPolicy: Never
        tolerations:
        - key: "kubernetes.io/spot"
          operator: "Equal"
          value: "true"
          effect: "NoSchedule"
        nodeSelector:
          node.kubernetes.io/instance-type: spot
    backoffLimit: 2
  pollingInterval: 30
  maxReplicaCount: 30
  successfulJobsHistoryLimit: 5
  failedJobsHistoryLimit: 3
  triggers:
  - type: cron
    metadata:
      timezone: UTC
      start: 0 2 * * *      # Start at 2 AM UTC
      end: 0 6 * * *         # End at 6 AM UTC
      desiredReplicas: "20"
```

</details>

---

**Previous**: [Admission Controllers](./12_Admission_Controllers.md) | **Next**: [Observability](./14_Observability.md)

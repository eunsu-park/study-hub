# 02. Workload Resources

**Previous**: [Architecture Deep Dive](./01_Architecture_Deep_Dive.md) | **Next**: [Networking Fundamentals](./03_Networking_Fundamentals.md)

## Learning Objectives
- Master Pod lifecycle, init containers, and multi-container patterns
- Deploy and manage ReplicaSets, Deployments, and StatefulSets
- Configure rolling updates, rollbacks, and deployment strategies
- Use Jobs and CronJobs for batch workloads
- Understand resource management including requests, limits, and QoS classes

---

Workload resources are the building blocks that run your application code on
Kubernetes. From the atomic Pod to sophisticated StatefulSets, each resource
type is designed for a specific operational pattern. This lesson covers every
workload resource in detail with production-ready examples.

Before the workload tour, read [**Theory & Principles**](#theory--principles) — why each workload kind exists as a different reconciler over the same Pod primitive, the rolling-update math behind Deployments, the ordered-stable-network identity guarantee of StatefulSets, and how requests/limits/QoS map to the Linux kernel's cgroup knobs.

## Table of Contents
0. [Theory & Principles](#theory--principles)
1. [Pods](#1-pods)
2. [ReplicaSets](#2-replicasets)
3. [Deployments](#3-deployments)
4. [StatefulSets](#4-statefulsets)
5. [DaemonSets](#5-daemonsets)
6. [Jobs and CronJobs](#6-jobs-and-cronjobs)
7. [Pod Disruption Budgets](#7-pod-disruption-budgets)
8. [Resource Requests and Limits](#8-resource-requests-and-limits)
9. [QoS Classes](#9-qos-classes)
10. [Exercises](#exercises)

---

## Theory & Principles

Workload resources look like a long menu — Pod, ReplicaSet, Deployment, StatefulSet, DaemonSet, Job, CronJob — but they are all built from one primitive (the Pod) plus a single design pattern (a controller that watches a desired-state object and reconciles a set of child Pods). What changes between them is the *reconciliation policy*: how many replicas, in what order, with what identity guarantees, and what to do on failure. This section covers the four ideas that explain every workload kind plus the resource-management semantics that determine whether your pod gets killed under memory pressure.

### A. Pod as the Atomic Scheduling Unit

A Pod is one or more containers that **share fate** — same node, same network namespace (same IP, same `localhost`, same port space), optional shared volumes, scheduled together, killed together. The decision to make Pods (not containers) the atom enables three patterns that bare containers cannot model cleanly:

- **Sidecar**: a logging or proxy container that needs to share volumes and localhost with the main app, but should be upgraded independently.
- **Init container**: a sequenced setup step (DB migration, config fetch) that runs to completion before the main container starts; failure blocks startup.
- **Adapter**: a translation container that exposes the main app's metrics or API in a different protocol over localhost.

For the common single-container case, the Pod adds essentially zero overhead. Crucially, **Pods are ephemeral and have ephemeral IPs.** They are not pets. A Pod's IP changes on restart, and a Pod that fails its liveness probe is replaced by a new Pod with a new IP. Anything that needs a stable network identity must use a Service (or a Headless Service + StatefulSet for per-pod identity).

This ephemerality is what enables every higher-level workload: if Pods were precious, you could not safely roll, scale, or evict them. Because they are disposable, the controller above can always create more.

### B. ReplicaSet → Deployment: The Rolling-Update Algorithm

A **ReplicaSet** is the simplest reconciler over Pods: "I want N Pods matching this selector. Right now I see M. If M < N, create. If M > N, delete." That is the entire algorithm, expressed as a level-triggered loop. ReplicaSets handle scaling and self-healing, but they do *not* handle template changes — modifying the pod template does not update existing Pods.

A **Deployment** wraps a ReplicaSet (actually two of them during a rollout) and adds the rolling-update algorithm. Given a new pod template:

1. Create a new ReplicaSet RS-new with `replicas=0` and the new template.
2. Repeat until RS-new has the target count and RS-old has 0:
   - Scale RS-new up by `maxSurge` (default 25%).
   - Wait for the new pods to become Ready.
   - Scale RS-old down by `maxUnavailable` (default 25%).
3. Keep RS-old around (with 0 replicas) for one-command rollback.

The math: with `replicas=10`, `maxSurge=25%`, `maxUnavailable=25%`, you can have at most 13 pods total at any moment, and at least 7 must be Ready. This bounds both extra cost (the surge) and capacity loss (the unavailability) during the rollout. `recreate` strategy skips this and just kills everything before recreating — used when two versions cannot coexist (e.g., schema migrations).

Rollback is just "set the deployment template back to the old ReplicaSet's pod-template-hash." The deployment controller then runs the same algorithm in reverse, scaling RS-old back up and RS-new down. This is why kept history (`revisionHistoryLimit`) matters.

### C. StatefulSet: Order, Identity, Storage

StatefulSets exist because some workloads (databases, message brokers, distributed consensus systems) need **stable identity per replica**. Three guarantees differ from Deployments:

- **Ordered, stable network identity.** Pods are named `<set>-0`, `<set>-1`, ..., `<set>-(N-1)`. With a Headless Service, each pod gets a DNS name like `mysql-0.mysql.default.svc.cluster.local` that persists across restarts and reschedules.
- **Ordered deployment and termination.** Pods come up one at a time in order (`-0`, then `-1`, then `-2`), each waiting for the previous to be Ready. Termination is reverse order. This matters for cluster bootstrap (e.g., the first replica is the seed; replica 1 joins by referencing replica 0's stable DNS name).
- **Stable, per-replica storage via VolumeClaimTemplates.** Each pod gets its own PVC, with a name derived from the pod's ordinal. When pod `mysql-1` reschedules, it reattaches to the same PVC — same data. A Deployment cannot do this safely because it has no notion of "which pod is which."

The cost of these guarantees is reduced agility: scaling a StatefulSet from 3 to 6 takes longer than scaling a Deployment because pods come up sequentially. Updates use either `RollingUpdate` (one pod at a time, reverse-ordinal) or `OnDelete` (operator-driven, when ordering is too dangerous to automate).

### D. DaemonSet, Job, CronJob: Specialized Reconcilers

Each of these is the same controller pattern with a different "what does desired state mean?":

- **DaemonSet**: "one Pod per matching node." Reconciler watches both the DaemonSet and the Node list; it creates a Pod on every new node that matches the node selector and removes Pods from drained or deleted nodes. Used for log collectors, node exporters, CNI agents — anything that must run cluster-wide.
- **Job**: "run N successful completions, with up to P running in parallel." Reconciler creates Pods, watches for completion, and stops when the success count is reached. `backoffLimit` caps retries on failure; `activeDeadlineSeconds` caps total wall-clock time. Critical for batch workloads.
- **CronJob**: a Job factory. Reconciler reads the crontab-style schedule, and at each fire time creates a fresh Job from a template. `concurrencyPolicy` decides what happens if the previous Job is still running (`Allow`, `Forbid`, `Replace`).

You write a Pod template; the controller decides how many, in what order, on which nodes, with what success criteria.

### E. Requests, Limits, QoS: From YAML to cgroups

Resource management has two numbers per container per resource (CPU, memory):

- **`requests`**: the scheduler treats this as the pod's reservation. Sum of all pod requests on a node cannot exceed node allocatable. This is what determines *whether* the pod fits on a node.
- **`limits`**: the kernel-enforced ceiling. CPU above the limit is throttled (cgroup `cpu.cfs_quota_us`); memory above the limit triggers OOM kill (cgroup `memory.limit_in_bytes`).

Three QoS classes emerge automatically from the requests/limits combination:

| Class | Condition | Eviction priority |
|-------|-----------|-------------------|
| `Guaranteed` | every container has `requests == limits` for both CPU and memory | last to be evicted |
| `Burstable` | at least one request set, but not all `requests == limits` | middle |
| `BestEffort` | no requests or limits set anywhere | first to be evicted |

Under node memory pressure, the kubelet evicts BestEffort first, then Burstable in order of how far they exceed their requests, then Guaranteed only if absolutely necessary. So **`Guaranteed` is not just for resource budgeting — it is your eviction insurance.**

CPU vs memory differ critically: CPU is **compressible** (you throttle the slow tenant and everyone keeps running), memory is **incompressible** (when it's gone, someone must die). This is why memory-OOM is sudden and fatal while CPU-throttle is gradual and recoverable.

### From Theory to the YAML Below

The lesson now walks you through these abstractions as concrete manifests:

- **Section 1 (Pods)** demonstrates the atomic unit from §A — multi-container patterns, init containers, sidecars, lifecycle hooks.
- **Section 2 (ReplicaSets)** shows the simplest reconciler from §B in isolation, before Deployments take over.
- **Section 3 (Deployments)** is the rolling-update algorithm from §B applied to real services, with `maxSurge`/`maxUnavailable` knobs to tune.
- **Section 4 (StatefulSets)** is §C, with VolumeClaimTemplates and Headless Services that give per-pod identity.
- **Sections 5–6 (DaemonSets, Jobs, CronJobs)** are the specialized reconcilers from §D.
- **Section 7 (PDB)** is a guardrail against the rolling-update algorithm — "don't ever take more than X pods down at once, even during voluntary disruption."
- **Sections 8–9 (Requests/Limits, QoS)** translate §E into the YAML fields the kubelet feeds to cgroups.

Once you see Pod-as-atom + reconciler-as-pattern, the difference between a DaemonSet and a Job is just two lines of pseudocode.

---

## 1. Pods

### 1.1 Pod Fundamentals

A Pod is the smallest deployable unit in Kubernetes. It encapsulates one or more
containers that share network and storage, and a specification for how to run them.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: simple-pod
  labels:
    app: demo
    version: v1
spec:
  containers:
    - name: web
      image: nginx:1.25
      ports:
        - containerPort: 80
          name: http
          protocol: TCP
```

Key characteristics:
- All containers in a Pod share the same network namespace (localhost)
- All containers share the same IPC namespace
- Volumes can be shared between containers
- Pods are ephemeral—they are never "repaired," only replaced

### 1.2 Pod Lifecycle

```
Pending → Running → Succeeded / Failed
              │
              └──→ Unknown (node lost contact)
```

| Phase | Description |
|-------|-------------|
| Pending | Accepted by the cluster but containers not yet running |
| Running | At least one container is running |
| Succeeded | All containers terminated with exit code 0 |
| Failed | At least one container terminated with non-zero exit code |
| Unknown | Node communication failure |

```bash
# Watch pod phase transitions
kubectl get pod simple-pod -w

# Detailed phase and condition information
kubectl get pod simple-pod -o jsonpath='{.status.phase}'
kubectl get pod simple-pod -o jsonpath='{.status.conditions}' | python3 -m json.tool
```

### 1.3 Container States

Each container within a pod has its own state:

| State | Description |
|-------|-------------|
| Waiting | Container is not yet running (pulling image, etc.) |
| Running | Container is executing |
| Terminated | Container finished execution |

### 1.4 Init Containers

Init containers run sequentially before app containers start. They are useful for
setup tasks like database migrations, configuration generation, or waiting for
dependent services.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: init-demo
spec:
  initContainers:
    # Init container 1: Wait for a service to be available
    - name: wait-for-db
      image: busybox:1.36
      command:
        - sh
        - -c
        - |
          echo "Waiting for database..."
          until nslookup postgres-service.default.svc.cluster.local; do
            echo "Database not ready, retrying in 2s..."
            sleep 2
          done
          echo "Database is available"

    # Init container 2: Run a database migration
    - name: run-migration
      image: my-app:v1.0
      command: ["./migrate", "--target=latest"]
      env:
        - name: DB_HOST
          value: "postgres-service"

  containers:
    - name: app
      image: my-app:v1.0
      ports:
        - containerPort: 8080
```

Properties of init containers:
- Run one at a time, in order
- Each must complete successfully before the next starts
- If an init container fails, the kubelet restarts it (subject to `restartPolicy`)
- Init containers do not support `livenessProbe`, `readinessProbe`, or `startupProbe`
- They can have different resource limits than app containers

### 1.4.1 Native Sidecar Containers (Kubernetes 1.29+)

Kubernetes 1.29 introduced native sidecar support via `initContainers` with
`restartPolicy: Always`. Unlike regular init containers, native sidecars start
before app containers and remain running for the entire Pod lifetime—solving
the problem of log shippers or service-mesh proxies that need to outlive the
main app container.

```yaml
spec:
  initContainers:
    - name: log-collector          # Native sidecar: runs alongside app containers
      image: fluent/fluent-bit:2.2
      restartPolicy: Always        # Key field — makes this a sidecar, not a one-shot init
      volumeMounts:
        - name: shared-logs
          mountPath: /var/log/app
          readOnly: true
    - name: run-migration          # Regular init container (no restartPolicy)
      image: my-app:v1.0
      command: ["./migrate", "--target=latest"]
  containers:
    - name: app
      image: my-app:v1.0
```

Ordering: native sidecars start after the regular init containers that precede
them complete, and they are guaranteed to be Running before subsequent init
containers or app containers start.

### 1.5 Multi-Container Patterns

#### Sidecar Pattern

A helper container augments the main application:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: sidecar-logging
spec:
  volumes:
    - name: shared-logs
      emptyDir: {}
  containers:
    # Main application
    - name: app
      image: my-app:v1.0
      volumeMounts:
        - name: shared-logs
          mountPath: /var/log/app

    # Sidecar: log shipper
    - name: log-shipper
      image: fluent/fluent-bit:2.2
      volumeMounts:
        - name: shared-logs
          mountPath: /var/log/app
          readOnly: true
      env:
        - name: FLUENT_ELASTICSEARCH_HOST
          value: "elasticsearch.logging.svc"
```

#### Ambassador Pattern

A proxy container handles outbound connections:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ambassador-pattern
spec:
  containers:
    - name: app
      image: my-app:v1.0
      env:
        # App connects to localhost, ambassador proxies to the real service
        - name: DB_HOST
          value: "localhost"
        - name: DB_PORT
          value: "5432"

    - name: ambassador
      image: haproxy:2.9
      ports:
        - containerPort: 5432
      volumeMounts:
        - name: haproxy-config
          mountPath: /usr/local/etc/haproxy
  volumes:
    - name: haproxy-config
      configMap:
        name: ambassador-haproxy-config
```

#### Adapter Pattern

A container transforms the output of the main container:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: adapter-pattern
spec:
  volumes:
    - name: shared-data
      emptyDir: {}
  containers:
    # Main app writes metrics in a custom format
    - name: app
      image: legacy-app:v2.0
      volumeMounts:
        - name: shared-data
          mountPath: /metrics

    # Adapter converts metrics to Prometheus format
    - name: prometheus-adapter
      image: prom/statsd-exporter:v0.26.0
      ports:
        - containerPort: 9102
          name: metrics
      volumeMounts:
        - name: shared-data
          mountPath: /metrics
          readOnly: true
```

### 1.6 Probes

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: probes-demo
spec:
  containers:
    - name: app
      image: my-app:v1.0
      ports:
        - containerPort: 8080

      # Startup probe: slow-starting apps
      startupProbe:
        httpGet:
          path: /healthz
          port: 8080
        failureThreshold: 30
        periodSeconds: 10
        # Allows up to 300s (30 * 10) for startup

      # Liveness probe: restart if unhealthy
      livenessProbe:
        httpGet:
          path: /healthz
          port: 8080
        initialDelaySeconds: 0
        periodSeconds: 10
        timeoutSeconds: 5
        failureThreshold: 3

      # Readiness probe: remove from service if not ready
      readinessProbe:
        httpGet:
          path: /ready
          port: 8080
        initialDelaySeconds: 5
        periodSeconds: 5
        failureThreshold: 1
```

| Probe Type | Action on Failure | Use Case |
|-----------|-------------------|----------|
| Startup | Keep retrying | Slow-starting apps (DB migration) |
| Liveness | Restart container | Deadlock detection |
| Readiness | Remove from Service endpoints | Temporary unavailability |

### 1.7 Ephemeral Containers (kubectl debug)

Ephemeral containers are temporary containers injected into a running Pod for
debugging. They are useful when the main container image is minimal or
distroless and lacks a shell or debugging tools.

```bash
# Inject a debug container into a running pod (shares the pod's namespaces)
kubectl debug -it my-pod --image=busybox:1.36 --target=app

# Use a richer image for more tools
kubectl debug -it my-pod --image=nicolaka/netshoot --target=app

# Copy a pod (create a new pod with a modified spec for deeper debugging)
kubectl debug my-pod --copy-to=debug-pod --set-image=app=busybox:1.36 -it
```

Key properties of ephemeral containers:
- Cannot be restarted; removed when the Pod is deleted or restarted
- Cannot have probes or resource requests/limits
- Share the target container's process namespace when `--target` is specified
- Ideal for debugging distroless images (e.g., `gcr.io/distroless/static`) that
  contain no shell

---

## 2. ReplicaSets

A ReplicaSet ensures a specified number of pod replicas are running at any time.

```yaml
apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: web-rs
  labels:
    app: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
      tier: frontend
  template:
    metadata:
      labels:
        app: web
        tier: frontend
    spec:
      containers:
        - name: nginx
          image: nginx:1.25
          ports:
            - containerPort: 80
          resources:
            requests:
              cpu: "100m"
              memory: "64Mi"
            limits:
              cpu: "200m"
              memory: "128Mi"
```

```bash
# Create the ReplicaSet
kubectl apply -f web-rs.yaml

# Check status
kubectl get rs web-rs

# Scale manually
kubectl scale rs web-rs --replicas=5

# View which pods belong to this RS
kubectl get pods -l app=web,tier=frontend --show-labels
```

> **Note**: You rarely create ReplicaSets directly. Use Deployments instead—they
> manage ReplicaSets for you and add rollout/rollback capabilities.

---

## 3. Deployments

Deployments manage ReplicaSets, which in turn manage Pods. When you perform a
rolling update, the Deployment creates a new ReplicaSet and gradually shifts
Pods from the old RS to the new one. Old ReplicaSets (scaled to 0) are kept for
rollback, controlled by `revisionHistoryLimit`.

```
Deployment
├── ReplicaSet (v2 — current, replicas=3)
│   ├── Pod web-deploy-6d7f9b-abc
│   ├── Pod web-deploy-6d7f9b-def
│   └── Pod web-deploy-6d7f9b-ghi
└── ReplicaSet (v1 — previous, replicas=0)  ← kept for rollback
    └── (no pods)
```

### 3.1 Basic Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-deploy
  labels:
    app: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
        - name: nginx
          image: nginx:1.25
          ports:
            - containerPort: 80
          readinessProbe:
            httpGet:
              path: /
              port: 80
            periodSeconds: 5
          resources:
            requests:
              cpu: "100m"
              memory: "64Mi"
            limits:
              cpu: "200m"
              memory: "128Mi"
```

### 3.2 Rolling Update Strategy

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-deploy
spec:
  replicas: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 2     # At most 2 pods can be unavailable during update
      maxSurge: 3           # At most 3 extra pods can exist during update
  minReadySeconds: 10       # Wait 10s after a pod is ready before proceeding
  revisionHistoryLimit: 5   # Keep 5 old ReplicaSets for rollback
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
        - name: nginx
          image: nginx:1.26
          ports:
            - containerPort: 80
```

Rolling update sequence (with 10 replicas, maxUnavailable=2, maxSurge=3):
1. Create new RS, scale to 3 (surge)
2. Scale old RS down to 8 (2 unavailable)
3. As new pods become ready, continue scaling new RS up and old RS down
4. Eventually: new RS at 10, old RS at 0

```bash
# Trigger a rolling update by changing the image
kubectl set image deployment/web-deploy nginx=nginx:1.26

# Watch the rollout
kubectl rollout status deployment/web-deploy

# View rollout history
kubectl rollout history deployment/web-deploy

# View a specific revision
kubectl rollout history deployment/web-deploy --revision=2
```

### 3.3 Recreate Strategy

All existing pods are killed before new ones are created:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: legacy-app
spec:
  replicas: 3
  strategy:
    type: Recreate    # Kill all, then create all (downtime!)
  selector:
    matchLabels:
      app: legacy
  template:
    metadata:
      labels:
        app: legacy
    spec:
      containers:
        - name: app
          image: legacy-app:v2.0
```

Use Recreate when:
- The app cannot tolerate running two versions simultaneously
- The app uses a volume with `ReadWriteOnce` access mode
- Database schema changes are incompatible with the old version

### 3.4 Rollbacks

```bash
# Rollback to the previous revision
kubectl rollout undo deployment/web-deploy

# Rollback to a specific revision
kubectl rollout undo deployment/web-deploy --to-revision=2

# Pause a rollout (for canary-style testing)
kubectl rollout pause deployment/web-deploy

# Resume after pausing
kubectl rollout resume deployment/web-deploy
```

### 3.5 Blue-Green Deployment Pattern

Kubernetes does not have a built-in blue-green resource, but you can implement it
with two Deployments and a Service:

```yaml
# Blue deployment (current production)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
      version: blue
  template:
    metadata:
      labels:
        app: web
        version: blue
    spec:
      containers:
        - name: app
          image: my-app:v1.0
---
# Green deployment (new version)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
      version: green
  template:
    metadata:
      labels:
        app: web
        version: green
    spec:
      containers:
        - name: app
          image: my-app:v2.0
---
# Service: switch by changing the selector
apiVersion: v1
kind: Service
metadata:
  name: web-svc
spec:
  selector:
    app: web
    version: blue    # Change to "green" to switch traffic
  ports:
    - port: 80
      targetPort: 8080
```

```bash
# Switch traffic from blue to green
kubectl patch service web-svc -p '{"spec":{"selector":{"version":"green"}}}'

# Rollback: switch back to blue
kubectl patch service web-svc -p '{"spec":{"selector":{"version":"blue"}}}'
```

---

## 4. StatefulSets

StatefulSets manage stateful applications with guarantees about ordering, stable
network identity, and persistent storage.

### 4.1 Key Properties

| Feature | Deployment | StatefulSet |
|---------|-----------|-------------|
| Pod names | Random hash suffix | Ordered index (0, 1, 2...) |
| Scaling | Parallel | Sequential (by default) |
| Storage | Shared or none | Per-pod PVC |
| Network identity | Random | Stable DNS (`pod-0.svc`) |
| Update order | Any | Reverse ordinal (N-1 → 0) |

### 4.2 StatefulSet Manifest

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: postgres-headless   # Required: headless service name
  replicas: 3
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
        - name: postgres
          image: postgres:16
          ports:
            - containerPort: 5432
              name: postgres
          env:
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: postgres-secret
                  key: password
            - name: PGDATA
              value: /var/lib/postgresql/data/pgdata
          volumeMounts:
            - name: data
              mountPath: /var/lib/postgresql/data
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "1"
              memory: "1Gi"
          readinessProbe:
            exec:
              command:
                - pg_isready
                - -U
                - postgres
            periodSeconds: 10

  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: standard
        resources:
          requests:
            storage: 10Gi

  # Pod management policy
  podManagementPolicy: OrderedReady   # Default: scale one at a time
  # podManagementPolicy: Parallel     # Scale all at once (for non-ordered workloads)
---
# Headless service for stable DNS
apiVersion: v1
kind: Service
metadata:
  name: postgres-headless
spec:
  clusterIP: None            # Headless service
  selector:
    app: postgres
  ports:
    - port: 5432
      targetPort: 5432
```

### 4.3 Stable Network Identity

StatefulSets provide three stable guarantees that Deployments do not:

```
StatefulSet: web (replicas=3)
│
├── Stable pod names (predictable, not random)
│   ├── web-0
│   ├── web-1
│   └── web-2
│
├── Stable DNS via headless Service (clusterIP: None)
│   ├── web-0.web-headless.default.svc.cluster.local
│   ├── web-1.web-headless.default.svc.cluster.local
│   └── web-2.web-headless.default.svc.cluster.local
│
└── Per-pod PersistentVolumeClaims (survive pod restarts)
    ├── data-web-0  (bound to PV)
    ├── data-web-1  (bound to PV)
    └── data-web-2  (bound to PV)
```

With the above StatefulSet, pods get predictable DNS names:

```
postgres-0.postgres-headless.default.svc.cluster.local
postgres-1.postgres-headless.default.svc.cluster.local
postgres-2.postgres-headless.default.svc.cluster.local
```

```bash
# Verify DNS resolution from within the cluster
kubectl run dns-test --rm -it --image=busybox:1.36 --restart=Never -- \
  nslookup postgres-0.postgres-headless.default.svc.cluster.local
```

### 4.4 Update Strategies

```yaml
spec:
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      partition: 2    # Only update pods with ordinal >= 2
      # Useful for canary testing: update pod-2 first, then lower partition
```

```bash
# Update the image
kubectl set image statefulset/postgres postgres=postgres:17

# With partition=2, only postgres-2 gets updated
# Lower the partition to roll out to more pods
kubectl patch statefulset postgres -p '{"spec":{"updateStrategy":{"rollingUpdate":{"partition":1}}}}'
# Now postgres-1 and postgres-2 are updated

kubectl patch statefulset postgres -p '{"spec":{"updateStrategy":{"rollingUpdate":{"partition":0}}}}'
# Full rollout complete
```

### 4.5 Ordered Deployment and Scaling

- **Scale up**: pods are created in order: 0, 1, 2, ...
- **Scale down**: pods are terminated in reverse order: 2, 1, 0
- Each pod must be Running and Ready before the next is created
- This is critical for databases that require primary-first initialization

---

## 5. DaemonSets

A DaemonSet ensures a copy of a pod runs on every (or selected) node.

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: node-exporter
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: node-exporter
  template:
    metadata:
      labels:
        app: node-exporter
    spec:
      # Tolerate all taints to run on every node including control plane
      tolerations:
        - operator: Exists
      hostNetwork: true     # Use the node's network namespace
      hostPID: true         # Access host processes for metrics
      containers:
        - name: node-exporter
          image: prom/node-exporter:v1.7.0
          ports:
            - containerPort: 9100
              hostPort: 9100
              name: metrics
          args:
            - --path.procfs=/host/proc
            - --path.sysfs=/host/sys
            - --path.rootfs=/host/root
          volumeMounts:
            - name: proc
              mountPath: /host/proc
              readOnly: true
            - name: sys
              mountPath: /host/sys
              readOnly: true
            - name: root
              mountPath: /host/root
              readOnly: true
              mountPropagation: HostToContainer
          resources:
            requests:
              cpu: "50m"
              memory: "64Mi"
            limits:
              cpu: "200m"
              memory: "128Mi"
      volumes:
        - name: proc
          hostPath:
            path: /proc
        - name: sys
          hostPath:
            path: /sys
        - name: root
          hostPath:
            path: /
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1    # Update one node at a time
```

Common DaemonSet use cases:
- Node monitoring (node-exporter, Datadog agent)
- Log collection (Fluentd, Fluent Bit)
- Network plugins (Calico, Cilium)
- Storage plugins (CSI node drivers)

```bash
# Check DaemonSet status
kubectl -n monitoring get ds node-exporter

# See which nodes have the pod
kubectl -n monitoring get pods -l app=node-exporter -o wide
```

### 5.1 Targeting Specific Nodes

```yaml
spec:
  template:
    spec:
      nodeSelector:
        gpu: "true"      # Only run on GPU nodes
      # Or use affinity for more complex rules
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: node.kubernetes.io/instance-type
                    operator: In
                    values: ["p4d.24xlarge", "g5.xlarge"]
```

---

## 6. Jobs and CronJobs

### 6.1 Jobs

A Job creates one or more Pods and ensures a specified number of them successfully
terminate.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: data-migration
spec:
  completions: 1           # Number of successful completions needed
  parallelism: 1           # Number of pods running in parallel
  backoffLimit: 3           # Number of retries before marking as failed
  activeDeadlineSeconds: 600  # Timeout: kill after 10 minutes
  ttlSecondsAfterFinished: 3600  # Auto-delete after 1 hour

  template:
    spec:
      restartPolicy: Never   # Required: Never or OnFailure
      containers:
        - name: migrate
          image: my-app:v1.0
          command: ["./migrate", "--target=latest"]
          env:
            - name: DB_URL
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: url
          resources:
            requests:
              cpu: "500m"
              memory: "256Mi"
```

### 6.2 Parallel Jobs

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: batch-processor
spec:
  completions: 10          # Process 10 items total
  parallelism: 3           # Run 3 pods at a time
  completionMode: Indexed  # Each pod gets a unique index (0-9)
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: processor
          image: batch-processor:v1.0
          command:
            - sh
            - -c
            - |
              echo "Processing item $JOB_COMPLETION_INDEX"
              # Use the index to partition work
              ./process --partition=$JOB_COMPLETION_INDEX --total=10
          env:
            - name: JOB_COMPLETION_INDEX
              valueFrom:
                fieldRef:
                  fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
```

```bash
# Monitor job progress
kubectl get job batch-processor -w

# View completed/failed pods
kubectl get pods -l job-name=batch-processor

# View logs from a specific indexed pod
kubectl logs batch-processor-2-xxxxx
```

### 6.3 CronJobs

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: nightly-backup
spec:
  schedule: "0 2 * * *"              # Every day at 2 AM
  timeZone: "America/New_York"       # Timezone-aware (v1.27+)
  concurrencyPolicy: Forbid          # Skip if previous job still running
  successfulJobsHistoryLimit: 3      # Keep 3 successful jobs
  failedJobsHistoryLimit: 5          # Keep 5 failed jobs
  startingDeadlineSeconds: 300       # Skip if more than 5 min late
  suspend: false                     # Set true to pause scheduling

  jobTemplate:
    spec:
      backoffLimit: 2
      activeDeadlineSeconds: 3600    # 1-hour timeout
      template:
        spec:
          restartPolicy: OnFailure
          containers:
            - name: backup
              image: postgres:16
              command:
                - sh
                - -c
                - |
                  pg_dump -h $DB_HOST -U $DB_USER $DB_NAME | \
                    gzip > /backup/db-$(date +%Y%m%d).sql.gz
              env:
                - name: DB_HOST
                  value: "postgres-headless"
                - name: DB_USER
                  valueFrom:
                    secretKeyRef:
                      name: db-credentials
                      key: username
                - name: DB_NAME
                  value: "production"
                - name: PGPASSWORD
                  valueFrom:
                    secretKeyRef:
                      name: db-credentials
                      key: password
              volumeMounts:
                - name: backup-storage
                  mountPath: /backup
          volumes:
            - name: backup-storage
              persistentVolumeClaim:
                claimName: backup-pvc
```

CronJob concurrency policies:
| Policy | Behavior |
|--------|----------|
| Allow | Multiple jobs can run simultaneously (default) |
| Forbid | Skip new run if previous is still active |
| Replace | Kill the running job and start a new one |

```bash
# List CronJobs
kubectl get cronjobs

# Manually trigger a CronJob
kubectl create job --from=cronjob/nightly-backup manual-backup

# Suspend a CronJob
kubectl patch cronjob nightly-backup -p '{"spec":{"suspend":true}}'
```

---

## 7. Pod Disruption Budgets

PDBs protect applications during voluntary disruptions (node drains, cluster
upgrades) by ensuring a minimum number of pods remain available.

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: web-pdb
spec:
  # Use ONE of these:
  minAvailable: 2          # At least 2 pods must remain
  # maxUnavailable: 1      # At most 1 pod can be down
  # minAvailable: "50%"    # Percentage also works

  selector:
    matchLabels:
      app: web

  # Unhealthy pod eviction policy (v1.31+)
  unhealthyPodEvictionPolicy: AlwaysAllow
  # IfHealthy (default): only evict unhealthy pods if all others are healthy
  # AlwaysAllow: always allow eviction of unhealthy pods
```

```bash
# View PDB status
kubectl get pdb web-pdb

# Output:
# NAME      MIN AVAILABLE   MAX UNAVAILABLE   ALLOWED DISRUPTIONS   AGE
# web-pdb   2               N/A               1                     5m

# Drain a node (will respect PDBs)
kubectl drain node-1 --ignore-daemonsets --delete-emptydir-data
```

PDB interaction during node drain:
1. kubectl drain sends eviction requests
2. API server checks PDBs before allowing eviction
3. If eviction would violate PDB, the request is rejected (429)
4. kubectl drain retries until the PDB allows the eviction

---

## 8. Resource Requests and Limits

### 8.1 CPU and Memory

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: resource-demo
spec:
  containers:
    - name: app
      image: my-app:v1.0
      resources:
        requests:
          cpu: "250m"       # 0.25 CPU cores (guaranteed minimum)
          memory: "256Mi"   # 256 MiB (guaranteed minimum)
        limits:
          cpu: "500m"       # 0.5 CPU cores (maximum allowed)
          memory: "512Mi"   # 512 MiB (OOMKilled if exceeded)
```

Resource behavior:
| Resource | Request | Limit | Exceeding Limit |
|----------|---------|-------|-----------------|
| CPU | Guaranteed scheduling | Throttled via CFS | Throttled (not killed) |
| Memory | Guaranteed scheduling | Hard cap | OOMKilled |

### 8.2 Understanding CPU Units

```
1 CPU = 1000m (millicores)
1 CPU = 1 vCPU (AWS) = 1 Core (GCP) = 1 vCore (Azure)

Common values:
  100m = 0.1 CPU (10% of one core)
  250m = 0.25 CPU
  500m = 0.5 CPU
  1000m = 1.0 CPU = 1
```

### 8.3 Understanding Memory Units

```
Binary units (power of 2):
  Ki = 1024 bytes
  Mi = 1024 Ki = 1,048,576 bytes
  Gi = 1024 Mi = 1,073,741,824 bytes

Decimal units (power of 10):
  K = 1000 bytes
  M = 1000 K = 1,000,000 bytes
  G = 1000 M = 1,000,000,000 bytes

Always use binary units (Mi, Gi) for consistency.
```

### 8.4 LimitRange

Enforce defaults and constraints at the namespace level:

```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: default-limits
  namespace: production
spec:
  limits:
    - type: Container
      default:           # Default limits (if not specified)
        cpu: "500m"
        memory: "256Mi"
      defaultRequest:    # Default requests (if not specified)
        cpu: "100m"
        memory: "128Mi"
      min:               # Minimum allowed
        cpu: "50m"
        memory: "64Mi"
      max:               # Maximum allowed
        cpu: "2"
        memory: "2Gi"
    - type: Pod
      max:
        cpu: "4"
        memory: "4Gi"
```

### 8.5 ResourceQuota

Limit total resource consumption per namespace:

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: compute-quota
  namespace: production
spec:
  hard:
    requests.cpu: "20"
    requests.memory: "40Gi"
    limits.cpu: "40"
    limits.memory: "80Gi"
    pods: "50"
    persistentvolumeclaims: "10"
    services.loadbalancers: "2"
    count/deployments.apps: "20"
```

```bash
# View quota usage
kubectl describe resourcequota compute-quota -n production
```

---

## 9. QoS Classes

Kubernetes assigns one of three QoS classes to every pod based on its resource
configuration. The QoS class determines the order in which pods are killed during
node memory pressure.

### 9.1 Guaranteed

Every container has requests == limits for both CPU and memory.

```yaml
# QoS: Guaranteed
spec:
  containers:
    - name: app
      resources:
        requests:
          cpu: "500m"
          memory: "256Mi"
        limits:
          cpu: "500m"       # Same as request
          memory: "256Mi"   # Same as request
```

- Last to be evicted during memory pressure
- Gets dedicated CPU time (if CPU manager policy is `static`)

### 9.2 Burstable

At least one container has a request or limit set, but they are not equal.

```yaml
# QoS: Burstable
spec:
  containers:
    - name: app
      resources:
        requests:
          cpu: "100m"
          memory: "128Mi"
        limits:
          cpu: "500m"       # Different from request
          memory: "512Mi"   # Different from request
```

- Evicted after BestEffort pods
- Can burst above requests when resources are available

### 9.3 BestEffort

No resource requests or limits set at all.

```yaml
# QoS: BestEffort
spec:
  containers:
    - name: app
      image: my-app:v1.0
      # No resources specified
```

- First to be evicted during memory pressure
- Gets whatever resources are available
- Not recommended for production workloads

### 9.4 Checking QoS Class

```bash
# View the QoS class assigned to a pod
kubectl get pod resource-demo -o jsonpath='{.status.qosClass}'

# List all pods with their QoS class
kubectl get pods -o custom-columns='NAME:.metadata.name,QOS:.status.qosClass'
```

### 9.5 Eviction Order

During node memory pressure, kubelet evicts pods in this order:

1. **BestEffort** pods exceeding no requests (first evicted)
2. **Burstable** pods exceeding their memory requests
3. **Guaranteed** pods (only if they exceed their limits, which equals OOM)

Within the same QoS class, pods using more memory relative to their request
are evicted first.

```bash
# Check node memory pressure conditions
kubectl describe node minikube | grep -A 5 Conditions

# Check eviction thresholds
kubectl get --raw /api/v1/nodes/minikube/proxy/configz | python3 -m json.tool | grep eviction
```

---

## Exercises

### Exercise 1: Multi-Container Pod

Create a pod with an init container that writes a file, and a main container
that serves it. The init container should write "Hello from init!" to a shared
volume, and the main container should serve it on port 80.

<details>
<summary>Show Answer</summary>

```yaml
# Save as /tmp/multi-container.yaml
apiVersion: v1
kind: Pod
metadata:
  name: multi-container-demo
spec:
  volumes:
    - name: shared-data
      emptyDir: {}
  initContainers:
    - name: init-writer
      image: busybox:1.36
      command:
        - sh
        - -c
        - |
          echo "<html><body><h1>Hello from init!</h1></body></html>" \
            > /data/index.html
          echo "Init container completed"
      volumeMounts:
        - name: shared-data
          mountPath: /data
  containers:
    - name: web-server
      image: nginx:1.25
      ports:
        - containerPort: 80
      volumeMounts:
        - name: shared-data
          mountPath: /usr/share/nginx/html
          readOnly: true
      resources:
        requests:
          cpu: "50m"
          memory: "64Mi"
```

```bash
kubectl apply -f /tmp/multi-container.yaml
kubectl wait --for=condition=Ready pod/multi-container-demo --timeout=60s

# Verify the content
kubectl exec multi-container-demo -c web-server -- curl -s localhost

# Check init container logs
kubectl logs multi-container-demo -c init-writer

# Clean up
kubectl delete pod multi-container-demo
```

</details>

### Exercise 2: Deployment Rolling Update

Create a Deployment with 5 replicas running nginx:1.24. Perform a rolling update
to nginx:1.25 with maxUnavailable=1 and maxSurge=2. Then rollback to the
previous version.

<details>
<summary>Show Answer</summary>

```yaml
# Save as /tmp/rolling-update.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rolling-demo
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 2
  selector:
    matchLabels:
      app: rolling-demo
  template:
    metadata:
      labels:
        app: rolling-demo
    spec:
      containers:
        - name: nginx
          image: nginx:1.24
          ports:
            - containerPort: 80
          readinessProbe:
            httpGet:
              path: /
              port: 80
            periodSeconds: 3
          resources:
            requests:
              cpu: "50m"
              memory: "64Mi"
```

```bash
# Deploy initial version
kubectl apply -f /tmp/rolling-update.yaml
kubectl rollout status deployment/rolling-demo

# Record the initial state
kubectl rollout history deployment/rolling-demo

# Trigger rolling update
kubectl set image deployment/rolling-demo nginx=nginx:1.25

# Watch the rollout in real time
kubectl rollout status deployment/rolling-demo

# Verify new image
kubectl get pods -l app=rolling-demo -o jsonpath='{.items[0].spec.containers[0].image}'
# nginx:1.25

# Rollback to previous version
kubectl rollout undo deployment/rolling-demo

# Verify rollback
kubectl get pods -l app=rolling-demo -o jsonpath='{.items[0].spec.containers[0].image}'
# nginx:1.24

# Clean up
kubectl delete deployment rolling-demo
```

</details>

### Exercise 3: StatefulSet with Persistent Storage

Create a StatefulSet for a Redis cluster with 3 replicas, each with its own
1Gi PersistentVolumeClaim. Verify stable network identity and storage persistence.

<details>
<summary>Show Answer</summary>

```yaml
# Save as /tmp/statefulset-exercise.yaml
apiVersion: v1
kind: Service
metadata:
  name: redis-headless
spec:
  clusterIP: None
  selector:
    app: redis
  ports:
    - port: 6379
      targetPort: 6379
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
spec:
  serviceName: redis-headless
  replicas: 3
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
        - name: redis
          image: redis:7.2
          ports:
            - containerPort: 6379
          command:
            - redis-server
            - --appendonly
            - "yes"
            - --dir
            - /data
          volumeMounts:
            - name: redis-data
              mountPath: /data
          resources:
            requests:
              cpu: "100m"
              memory: "128Mi"
            limits:
              cpu: "200m"
              memory: "256Mi"
          readinessProbe:
            exec:
              command: ["redis-cli", "ping"]
            periodSeconds: 5
  volumeClaimTemplates:
    - metadata:
        name: redis-data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 1Gi
```

```bash
kubectl apply -f /tmp/statefulset-exercise.yaml

# Wait for all pods
kubectl rollout status statefulset/redis

# Verify ordered pod names
kubectl get pods -l app=redis
# redis-0   1/1   Running
# redis-1   1/1   Running
# redis-2   1/1   Running

# Verify stable DNS
kubectl run dns-check --rm -it --image=busybox:1.36 --restart=Never -- \
  nslookup redis-0.redis-headless.default.svc.cluster.local

# Write data to redis-0
kubectl exec redis-0 -- redis-cli set test-key "persisted-value"

# Delete the pod (StatefulSet recreates it with same name and PVC)
kubectl delete pod redis-0
kubectl wait --for=condition=Ready pod/redis-0 --timeout=60s

# Verify data persists
kubectl exec redis-0 -- redis-cli get test-key
# "persisted-value"

# Verify PVCs
kubectl get pvc -l app=redis
# redis-data-redis-0   Bound   1Gi
# redis-data-redis-1   Bound   1Gi
# redis-data-redis-2   Bound   1Gi

# Clean up
kubectl delete statefulset redis
kubectl delete svc redis-headless
kubectl delete pvc -l app=redis
```

</details>

### Exercise 4: Job with Indexed Completion

Create a Job that processes 5 items in parallel (2 at a time) using indexed
completion mode. Each pod should print its completion index.

<details>
<summary>Show Answer</summary>

```yaml
# Save as /tmp/indexed-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: indexed-processor
spec:
  completions: 5
  parallelism: 2
  completionMode: Indexed
  backoffLimit: 3
  ttlSecondsAfterFinished: 300
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: worker
          image: busybox:1.36
          command:
            - sh
            - -c
            - |
              INDEX=${JOB_COMPLETION_INDEX}
              echo "Worker processing item ${INDEX} of 5"
              echo "Start time: $(date)"
              # Simulate work proportional to index
              sleep $((INDEX + 1))
              echo "Item ${INDEX} completed at $(date)"
          resources:
            requests:
              cpu: "50m"
              memory: "32Mi"
```

```bash
kubectl apply -f /tmp/indexed-job.yaml

# Watch the job progress
kubectl get job indexed-processor -w

# See pods with their indexes
kubectl get pods -l job-name=indexed-processor \
  -o custom-columns='NAME:.metadata.name,INDEX:.metadata.annotations.batch\.kubernetes\.io/job-completion-index,STATUS:.status.phase'

# View logs from each worker
for i in $(seq 0 4); do
  echo "=== Worker $i ==="
  kubectl logs -l batch.kubernetes.io/job-completion-index=$i -l job-name=indexed-processor
done

# Check job status
kubectl describe job indexed-processor

# Clean up
kubectl delete job indexed-processor
```

</details>

### Exercise 5: Resource Limits and QoS

Create three pods demonstrating each QoS class (Guaranteed, Burstable, BestEffort).
Verify the QoS class assignment for each pod.

<details>
<summary>Show Answer</summary>

```yaml
# Save as /tmp/qos-exercise.yaml
# Pod 1: Guaranteed (requests == limits)
apiVersion: v1
kind: Pod
metadata:
  name: qos-guaranteed
spec:
  containers:
    - name: app
      image: nginx:1.25
      resources:
        requests:
          cpu: "200m"
          memory: "128Mi"
        limits:
          cpu: "200m"
          memory: "128Mi"
---
# Pod 2: Burstable (requests != limits)
apiVersion: v1
kind: Pod
metadata:
  name: qos-burstable
spec:
  containers:
    - name: app
      image: nginx:1.25
      resources:
        requests:
          cpu: "100m"
          memory: "64Mi"
        limits:
          cpu: "500m"
          memory: "256Mi"
---
# Pod 3: BestEffort (no resources specified)
apiVersion: v1
kind: Pod
metadata:
  name: qos-besteffort
spec:
  containers:
    - name: app
      image: nginx:1.25
```

```bash
kubectl apply -f /tmp/qos-exercise.yaml

# Wait for all pods to be ready
kubectl wait --for=condition=Ready pod/qos-guaranteed pod/qos-burstable pod/qos-besteffort --timeout=60s

# Check QoS classes
echo "=== QoS Classes ==="
kubectl get pod qos-guaranteed -o jsonpath='qos-guaranteed:  {.status.qosClass}{"\n"}'
kubectl get pod qos-burstable -o jsonpath='qos-burstable:   {.status.qosClass}{"\n"}'
kubectl get pod qos-besteffort -o jsonpath='qos-besteffort:  {.status.qosClass}{"\n"}'

# Expected output:
# qos-guaranteed:  Guaranteed
# qos-burstable:   Burstable
# qos-besteffort:  BestEffort

# View all pods with QoS
kubectl get pods -o custom-columns='NAME:.metadata.name,QOS:.status.qosClass,CPU_REQ:.spec.containers[0].resources.requests.cpu,CPU_LIM:.spec.containers[0].resources.limits.cpu,MEM_REQ:.spec.containers[0].resources.requests.memory,MEM_LIM:.spec.containers[0].resources.limits.memory'

# Clean up
kubectl delete pod qos-guaranteed qos-burstable qos-besteffort
```

</details>

---

**Previous**: [Architecture Deep Dive](./01_Architecture_Deep_Dive.md) | **Next**: [Networking Fundamentals](./03_Networking_Fundamentals.md)

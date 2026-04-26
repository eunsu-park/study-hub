# 08. Kubernetes Advanced

**Previous**: [Kubernetes Security](./07_Kubernetes_Security.md) | **Next**: [Helm Package Management](./09_Helm_Package_Management.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Configure Ingress resources to route external HTTP/HTTPS traffic to cluster services
2. Describe the Kubernetes Gateway API resource model (GatewayClass, Gateway, HTTPRoute) and contrast it with Ingress
3. Deploy stateful applications using StatefulSet with stable network identities
4. Provision persistent storage with PersistentVolume and PersistentVolumeClaim
5. Apply advanced ConfigMap and Secret patterns including file mounts and dynamic updates
6. Use DaemonSet for node-level agents and Job/CronJob for batch workloads
7. Implement advanced scheduling with node affinity, taints, tolerations, and topology spread

---

The basic Kubernetes primitives -- Pods, Deployments, and Services -- cover many use cases, but production workloads demand more. Databases need stable storage and network identities, web applications need external HTTP routing with TLS, and cluster-wide agents must run on every node. This lesson introduces the advanced Kubernetes resources and scheduling techniques that bridge the gap between simple container orchestration and production-grade infrastructure.

## Table of Contents

Before the resource reference, read [**Theory & Principles**](#theory--principles) — the scheduler's filter+score algorithm, the controller pattern (informer + work queue) that powers every controller including custom operators, and the CRD/operator extension model.

1. [Ingress](#1-ingress)
2. [Gateway API](#2-gateway-api)
3. [StatefulSet](#3-statefulset)
4. [Persistent Storage](#4-persistent-storage)
5. [ConfigMap Advanced](#5-configmap-advanced)
6. [DaemonSet and Job](#6-daemonset-and-job)
7. [Advanced Scheduling](#7-advanced-scheduling)
8. [Practice Exercises](#8-practice-exercises)

---

## Theory & Principles

The advanced lesson is mostly about *extending* Kubernetes — new resources (CRDs), new controllers (operators), and new ways of telling the existing scheduler what you actually want. Three deep ideas underlie almost everything in the rest of the lesson: the **scheduler's two-phase decision**, the **controller pattern** (informer + work queue + reconcile), and the **CRD/operator extension model**.

### A. The Scheduler in Detail: Filter + Score

The scheduler runs once per unscheduled Pod (`spec.nodeName == ""`). Its decision is two phases:

**Filter phase (predicates).** From all nodes, drop any that cannot run this Pod. The classic predicates:

- `PodFitsResources` — does the node have enough allocatable CPU/memory left to satisfy `requests`?
- `PodFitsHostPorts` — are the requested host ports free?
- `PodToleratesNodeTaints` — does the Pod tolerate the node's `NoSchedule` taints?
- `MatchNodeSelector` — does the node's labels match `spec.nodeSelector` and `nodeAffinity`?
- `NoVolumeZoneConflict` — for stateful Pods, is the node in the same cloud zone as the PVC's volume?
- `MatchInterPodAffinity` — does this node satisfy `podAffinity` and `podAntiAffinity`?

A node that fails any predicate is removed from consideration. If zero nodes survive, the Pod stays Pending forever (events explain why).

**Score phase (priorities).** For each surviving node, compute a 0–100 score from a weighted sum of priority functions:

- `BalancedResourceAllocation` — prefer nodes whose CPU and memory utilization are balanced (don't fill CPU to 90% while memory sits at 10%).
- `LeastRequestedPriority` — prefer nodes with more free resources (spreads load).
- `NodeAffinityPriority` — prefer nodes that *prefer* (soft `nodeAffinity`) match.
- `InterPodAffinityPriority` — same for soft inter-pod affinity.
- `ImageLocalityPriority` — prefer nodes that already have the image cached (faster start).
- `TaintTolerationPriority` — prefer nodes whose `PreferNoSchedule` taints are all tolerated.

Highest score wins; ties broken randomly. The selected node is written to `spec.nodeName`, the kubelet on that node picks the Pod up, and runtime starts.

The scheduler is **pluggable**: you can write your own predicates and priorities and run a *second scheduler* alongside the default. Pods choose the scheduler with `spec.schedulerName: my-scheduler`. This is how teams with weird hardware (GPUs of varying generations, FPGAs, special NICs) extend scheduling without forking the default.

**Taints and tolerations** are the most useful manual scheduling tool: tainting a node with `gpu=true:NoSchedule` keeps anything without the matching toleration off it. The Pod's toleration is *permission* to land on the tainted node, not a *requirement* (use `nodeSelector`/`nodeAffinity` for "must run here"). The combination — taints to dedicate nodes, affinity to require/prefer them — covers most real-world topology.

### B. The Controller Pattern: Informer + Work Queue + Reconcile

Every K8s controller (built-in or custom) follows the same internal architecture, called the **controller pattern**. It is not a framework you call; it is a structure you adopt because the API server's design pushes you toward it.

**Informer.** A long-lived watch on the API server, with a local in-memory cache. The informer:

1. On startup, lists every object of its watched type and stores them in cache.
2. Establishes a watch (`?watch=true&resourceVersion=...`) and receives streaming Add/Update/Delete events.
3. Updates the local cache and fires registered event handlers.

The cache means controllers do not hammer the API server with reads; the watch means they hear about changes within milliseconds.

**Work queue.** Event handlers do *not* do work directly. They enqueue the *key* of the affected object (`namespace/name`) into a deduplicating, rate-limited work queue. Multiple events for the same key collapse into one work item. If reconciliation fails, the item is requeued with exponential backoff.

**Reconcile loop.** A worker goroutine pulls a key off the queue, looks up the current state of that object via the informer cache, computes the desired state from the spec, and takes whatever action narrows the diff (create children, update status, delete orphans). It is *not* told what changed — it computes the diff fresh every time. This makes the reconcile function **idempotent** and **state-driven** rather than event-driven.

The whole pattern is captured in `controller-runtime` (the Go library that builds operators). The same shape underlies the Deployment controller, ReplicaSet controller, kube-controller-manager bundles, and every CRD operator you write.

The discipline: **never trust the event; always re-read state.** If the queue is delayed, the world has moved on; reconciling against the cache instead of the event you just received protects against stale-event bugs.

### C. CRDs and Operators: Extending the API

Kubernetes was designed from the start to be extensible. The mechanism is the **Custom Resource Definition (CRD)** — a YAML object that registers a new resource type with the API server.

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: postgresqls.acid.zalan.do
spec:
  group: acid.zalan.do
  names: {kind: PostgreSQL, plural: postgresqls, shortNames: [pg]}
  scope: Namespaced
  versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema: {...}
```

Apply this and the API server learns to serve `kubectl get postgresqls`. Users can now create `PostgreSQL` objects and they get persisted to etcd like any built-in resource — but *nothing happens* until a controller watches them.

That controller is the **operator**. It is just a controller-pattern process (§B) that watches the CRD and reconciles its spec into actual resources: a StatefulSet for the Postgres pods, a Service for clients, Secrets for credentials, regular backups, failover logic. The operator encodes the operational knowledge — "to make a PostgreSQL highly available, you need to do X, Y, Z" — as code that runs continuously.

The pattern works because CRDs let domain experts define a high-level resource (`PostgreSQL`) and ship a controller that knows how to operate it. Users see `kubectl apply -f postgres.yaml`; the operator does the StatefulSet/PVC/replication wrangling underneath. Examples: cert-manager (`Certificate`, `Issuer`), Prometheus operator (`Prometheus`, `ServiceMonitor`), Argo CD (`Application`), Strimzi (`Kafka`).

The Operator Framework / OperatorSDK / kubebuilder are scaffolding tools to write operators in Go (or any language with a controller-runtime port). Helm charts can also use the operator model: helm-operator wraps a chart so it can be deployed via a CRD instance.

### D. StatefulSet and the "Stable Identity" Problem

A Deployment treats Pods as interchangeable replicas — same image, same config, no identity. For databases and consensus systems (Postgres replicas, ZooKeeper ensembles, Cassandra rings) that's wrong. They need:

- **Stable network identity.** Pod 0 should always be `db-0.db-service.namespace.svc.cluster.local`, even after restart.
- **Stable storage.** Pod 0's PVC follows Pod 0 across reschedules — not a fresh empty volume.
- **Ordered rollout.** During an update, replace Pod N-1 only after Pod N is healthy. Reverse for scale-down.

`StatefulSet` provides all three: Pods are named `<name>-0`, `<name>-1`, ..., a `headless Service` (clusterIP=None) gives each Pod its own DNS name, and `volumeClaimTemplates` creates one PVC per replica that survives pod deletion.

The trade-off is operations. Updating a StatefulSet is slower (one Pod at a time, in order). Scale-down keeps PVCs around (so re-scaling-up reattaches them — but you must clean them manually if you really want them gone). And the application itself has to know it has identity ("I am replica 0, I am the primary").

### E. Persistent Storage: PV, PVC, StorageClass

Storage in K8s is split into three resources to decouple developers from cluster admins:

- **PersistentVolume (PV)** — a cluster-scoped resource representing a chunk of storage that *exists* (a real EBS volume, a real NFS export, a real Ceph image). Created by an admin or by a dynamic provisioner.
- **PersistentVolumeClaim (PVC)** — a namespaced request for storage: "I need 10Gi of `fast-ssd` storage, ReadWriteOnce." Created by the application developer.
- **StorageClass** — a template for dynamic provisioning: "when a PVC references this class, run *this* CSI driver to allocate a PV with these parameters." Created by the cluster admin once.

The matchmaking logic: a new PVC binds to a fitting existing PV (size + class + access mode), or — if none exists — triggers the StorageClass's provisioner to create one. Once bound, the Pod's `volumes` field references the PVC by name and the kubelet mounts the underlying volume into the container.

`accessModes` matter: `ReadWriteOnce` (single node, fits block devices like EBS), `ReadWriteMany` (multi-node, requires NFS-like backends), `ReadOnlyMany`. Most cloud block storage is RWO and forces single-node placement of stateful workloads.

CSI (Container Storage Interface) is the standard plugin API — every cloud and storage vendor ships a CSI driver, and K8s talks to all of them through the same interface. Snapshots, clones, and resizing are CSI features your storage backend may or may not support.

### From Theory to the Resources Below

- **Ingress / Gateway API** — controllers that watch `Ingress` / `HTTPRoute` resources (CRD pattern §C) and reconcile them into nginx/Envoy/Istio configuration.
- **StatefulSet** — §D in YAML form. Wires headless Service + volumeClaimTemplates for stable identity + storage.
- **PV / PVC / StorageClass** — §E. The storage decoupling that makes "I need persistent storage" meaningful without naming the underlying disk.
- **DaemonSet** — a controller (§B) whose reconcile rule is "one replica per node matching this selector." Used for log shippers, node monitoring, CNI agents.
- **Job / CronJob** — controllers reconciling "run this Pod to completion N times" (Job) or "create a Job on this schedule" (CronJob).
- **Affinity / antiAffinity / taints / tolerations** — §A. The vocabulary you write to bias the scheduler's decision.
- **Custom controllers / operators** — §B + §C combined. Most production K8s ends up running several of these.

The remaining sections walk these resources. Whenever a resource feels mysterious, identify which controller owns it and what it watches; the reconciliation pattern explains the rest.

---

## 1. Ingress

### 1.1 Ingress Concepts

```
┌─────────────────────────────────────────────────────────────┐
│                    Ingress Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Internet                                                  │
│      │                                                      │
│      ▼                                                      │
│  ┌───────────────────────────────────────────┐             │
│  │         Ingress Controller                 │             │
│  │    (nginx, traefik, haproxy, etc.)        │             │
│  └───────────────────┬───────────────────────┘             │
│                      │                                      │
│        ┌─────────────┼─────────────┐                       │
│        │             │             │                        │
│        ▼             ▼             ▼                        │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐                   │
│   │Ingress  │  │Ingress  │  │Ingress  │                   │
│   │Resource │  │Resource │  │Resource │                   │
│   └────┬────┘  └────┬────┘  └────┬────┘                   │
│        │             │             │                        │
│        ▼             ▼             ▼                        │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐                   │
│   │Service A│  │Service B│  │Service C│                   │
│   └─────────┘  └─────────┘  └─────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Installing Ingress Controller

```bash
# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.11.3/deploy/static/provider/cloud/deploy.yaml

# Verify installation
kubectl get pods -n ingress-nginx
kubectl get svc -n ingress-nginx

# Check IngressClass
kubectl get ingressclass
```

### 1.3 Basic Ingress

```yaml
# simple-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: simple-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  rules:
  - host: myapp.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 80

---
# Host-based routing
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: host-based-ingress
spec:
  ingressClassName: nginx
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 8080
  - host: admin.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: admin-service
            port:
              number: 3000
```

### 1.4 Path-Based Routing

```yaml
# path-based-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: path-based-ingress
  annotations:
    nginx.ingress.kubernetes.io/use-regex: "true"
spec:
  ingressClassName: nginx
  rules:
  - host: app.example.com
    http:
      paths:
      # /api/* → api-service
      - path: /api(/|$)(.*)
        pathType: ImplementationSpecific
        backend:
          service:
            name: api-service
            port:
              number: 8080
      # /static/* → static-service
      - path: /static
        pathType: Prefix
        backend:
          service:
            name: static-service
            port:
              number: 80
      # Default → frontend
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 80
```

### 1.5 TLS Configuration

```yaml
# tls-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tls-ingress
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"  # Force HTTPS — prevents sensitive data from traveling over plaintext HTTP
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - secure.example.com
    secretName: tls-secret  # TLS Secret
  rules:
  - host: secure.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: secure-service
            port:
              number: 443

---
# Create TLS Secret
apiVersion: v1
kind: Secret
metadata:
  name: tls-secret
type: kubernetes.io/tls
data:
  tls.crt: <base64-encoded-cert>
  tls.key: <base64-encoded-key>
```

### 1.6 Advanced Ingress Configuration

```yaml
# advanced-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: advanced-ingress
  annotations:
    # Basic settings
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "60"

    # Rate Limiting — protects backend from DoS and abusive clients
    nginx.ingress.kubernetes.io/limit-rps: "100"
    nginx.ingress.kubernetes.io/limit-connections: "50"

    # CORS
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "https://frontend.example.com"
    nginx.ingress.kubernetes.io/cors-allow-methods: "GET, POST, PUT, DELETE, OPTIONS"

    # Authentication
    nginx.ingress.kubernetes.io/auth-type: basic
    nginx.ingress.kubernetes.io/auth-secret: basic-auth
    nginx.ingress.kubernetes.io/auth-realm: "Authentication Required"

    # Custom headers — security headers that defend against clickjacking and MIME-sniffing attacks
    nginx.ingress.kubernetes.io/configuration-snippet: |
      add_header X-Frame-Options "SAMEORIGIN";
      add_header X-Content-Type-Options "nosniff";

spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.example.com
    secretName: api-tls
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 8080
```

---

## 2. Gateway API

The Gateway API is the official successor to Ingress, GA since Kubernetes 1.31. It addresses Ingress's limitations through a role-oriented resource model that separates infrastructure concerns from application routing.

### 2.1 Ingress vs Gateway API

```
┌─────────────────────────────────────────────────────────────┐
│              Ingress vs Gateway API                          │
├──────────────────────┬──────────────────────────────────────┤
│      Ingress         │        Gateway API                   │
├──────────────────────┼──────────────────────────────────────┤
│  Single resource     │  3 layered resources:                │
│                      │  GatewayClass → Gateway → *Route     │
├──────────────────────┼──────────────────────────────────────┤
│  L7 HTTP only        │  L4+L7: HTTP, gRPC, TCP, TLS, UDP   │
├──────────────────────┼──────────────────────────────────────┤
│  No traffic split    │  Native weight-based traffic split   │
├──────────────────────┼──────────────────────────────────────┤
│  Annotations-heavy   │  Typed, structured configuration     │
├──────────────────────┼──────────────────────────────────────┤
│  Single-tenant       │  Multi-tenant (role separation)      │
├──────────────────────┼──────────────────────────────────────┤
│  Stable since 1.19   │  GA since 1.31                       │
└──────────────────────┴──────────────────────────────────────┘

Role-oriented design:
  Infra team   → deploys GatewayClass (cluster-wide)
  Platform team → creates Gateway    (namespace-scoped)
  App team      → attaches HTTPRoute (namespace-scoped)
```

### 2.2 Core Resources

```yaml
# gateway-api-example.yaml

# 1. GatewayClass — cluster-scoped, defines implementation
apiVersion: gateway.networking.k8s.io/v1
kind: GatewayClass
metadata:
  name: nginx
spec:
  controllerName: gateway.nginx.org/nginx-gateway-controller

---
# 2. Gateway — namespaced, creates the load balancer
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: main-gateway
  namespace: infra
spec:
  gatewayClassName: nginx
  listeners:
  - name: http
    port: 80
    protocol: HTTP
    allowedRoutes:
      namespaces:
        from: All  # Allow routes from any namespace — adjust to "Same" or "Selector" for stricter multi-tenancy
  - name: https
    port: 443
    protocol: HTTPS
    tls:
      mode: Terminate
      certificateRefs:
      - name: tls-cert
    allowedRoutes:
      namespaces:
        from: All

---
# 3. HTTPRoute — namespaced, defines routing rules
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: app-routes
  namespace: production
spec:
  parentRefs:
  - name: main-gateway
    namespace: infra
  hostnames:
  - "app.example.com"
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /api
    backendRefs:
    - name: api-service
      port: 8080
  - matches:
    - path:
        type: PathPrefix
        value: /
    backendRefs:
    - name: frontend-service
      port: 80
```

### 2.3 Traffic Splitting (Canary Deployment)

Gateway API makes canary deployments a first-class feature with weight-based routing — no custom annotations required.

```yaml
# canary-httproute.yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: canary-route
spec:
  parentRefs:
  - name: main-gateway
    namespace: infra
  hostnames:
  - "app.example.com"
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /
    backendRefs:
    - name: app-stable
      port: 80
      weight: 90   # 90% of traffic to stable version
    - name: app-canary
      port: 80
      weight: 10   # 10% to canary — gradually increase as confidence grows
```

### 2.4 Header-Based Routing

```yaml
# header-routing.yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: header-route
spec:
  parentRefs:
  - name: main-gateway
    namespace: infra
  rules:
  # Internal users get routed to beta version
  - matches:
    - headers:
      - name: X-User-Group
        value: internal
    backendRefs:
    - name: app-beta
      port: 80
  # Everyone else gets stable
  - backendRefs:
    - name: app-stable
      port: 80
```

---

## 3. StatefulSet

### 2.1 StatefulSet Concepts

```
┌─────────────────────────────────────────────────────────────┐
│            StatefulSet vs Deployment                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Deployment (Stateless)                                     │
│  ┌───────┐ ┌───────┐ ┌───────┐                             │
│  │pod-xyz│ │pod-abc│ │pod-123│  Random names, replaceable  │
│  └───────┘ └───────┘ └───────┘                             │
│                                                             │
│  StatefulSet (Stateful)                                     │
│  ┌───────┐ ┌───────┐ ┌───────┐                             │
│  │web-0  │ │web-1  │ │web-2  │  Ordered, unique IDs        │
│  │  ↓    │ │  ↓    │ │  ↓    │                             │
│  │pvc-0  │ │pvc-1  │ │pvc-2  │  Dedicated storage each     │
│  └───────┘ └───────┘ └───────┘                             │
│                                                             │
│  Features:                                                  │
│  • Ordered creation/deletion (0 → 1 → 2)                   │
│  • Fixed network IDs (pod-name.service-name)               │
│  • Persistent storage attached                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 StatefulSet Definition

```yaml
# statefulset-example.yaml
apiVersion: v1
kind: Service
metadata:
  name: web-headless
  labels:
    app: web
spec:
  ports:
  - port: 80
    name: web
  clusterIP: None  # Headless Service — gives each pod a stable DNS name (pod-name.svc) instead of a single cluster IP
  selector:
    app: web

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: web
spec:
  serviceName: "web-headless"  # Connect to Headless Service — required for stable per-pod DNS names
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
          name: web
        volumeMounts:
        - name: www
          mountPath: /usr/share/nginx/html

  # Volume claim templates
  volumeClaimTemplates:
  - metadata:
      name: www
    spec:
      accessModes: ["ReadWriteOnce"]  # Each pod gets its own PVC — prevents data corruption from concurrent writes
      storageClassName: standard
      resources:
        requests:
          storage: 1Gi

  # Update strategy
  updateStrategy:
    type: RollingUpdate  # Zero-downtime deploy: new pods start before old ones terminate
    rollingUpdate:
      partition: 0  # Only Pods >= this number are updated — useful for canary-testing a subset of replicas

  # Pod management policy
  podManagementPolicy: OrderedReady  # Or Parallel — OrderedReady ensures each pod is Running before starting the next (safe for leader election)
```

### 2.3 Database StatefulSet

```yaml
# mysql-statefulset.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mysql-config
data:
  my.cnf: |
    [mysqld]
    bind-address = 0.0.0.0
    default_authentication_plugin = mysql_native_password

---
apiVersion: v1
kind: Secret
metadata:
  name: mysql-secret
type: Opaque
stringData:
  root-password: "rootpass123"
  user-password: "userpass123"

---
apiVersion: v1
kind: Service
metadata:
  name: mysql-headless
spec:
  clusterIP: None
  selector:
    app: mysql
  ports:
  - port: 3306

---
apiVersion: v1
kind: Service
metadata:
  name: mysql
spec:
  selector:
    app: mysql
    statefulset.kubernetes.io/pod-name: mysql-0  # Primary only — routes all write traffic to the leader, preventing split-brain
  ports:
  - port: 3306

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mysql
spec:
  serviceName: mysql-headless
  replicas: 3
  selector:
    matchLabels:
      app: mysql
  template:
    metadata:
      labels:
        app: mysql
    spec:
      initContainers:
      - name: init-mysql
        image: mysql:8.0
        command:
        - bash
        - "-c"
        - |
          set -ex
          # Generate server-id from Pod index
          [[ `hostname` =~ -([0-9]+)$ ]] || exit 1
          ordinal=${BASH_REMATCH[1]}
          echo [mysqld] > /mnt/conf.d/server-id.cnf
          echo server-id=$((100 + $ordinal)) >> /mnt/conf.d/server-id.cnf
        volumeMounts:
        - name: conf
          mountPath: /mnt/conf.d

      containers:
      - name: mysql
        image: mysql:8.0
        env:
        - name: MYSQL_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mysql-secret
              key: root-password
        ports:
        - containerPort: 3306
        volumeMounts:
        - name: data
          mountPath: /var/lib/mysql
        - name: conf
          mountPath: /etc/mysql/conf.d
        - name: config
          mountPath: /etc/mysql/my.cnf
          subPath: my.cnf
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
        livenessProbe:  # liveness restarts the pod if MySQL hangs; separate from readiness to avoid cascading restarts
          exec:
            command: ["mysqladmin", "ping"]
          initialDelaySeconds: 30  # Give MySQL time to initialize before checking — avoids restart loops on slow starts
          periodSeconds: 10
        readinessProbe:  # readiness gates traffic; a failing probe removes the pod from the Service
          exec:
            command: ["mysql", "-h", "127.0.0.1", "-e", "SELECT 1"]  # Verifies the query engine is ready, not just the process
          initialDelaySeconds: 5
          periodSeconds: 2

      volumes:
      - name: conf
        emptyDir: {}
      - name: config
        configMap:
          name: mysql-config

  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast
      resources:
        requests:
          storage: 10Gi
```

### 2.4 StatefulSet Management

```bash
# View StatefulSets
kubectl get statefulset
kubectl describe statefulset web

# Check Pods (ordered names)
kubectl get pods -l app=web
# NAME    READY   STATUS
# web-0   1/1     Running
# web-1   1/1     Running
# web-2   1/1     Running

# Check DNS
# Each Pod: web-0.web-headless.default.svc.cluster.local
kubectl run -it --rm debug --image=busybox -- nslookup web-0.web-headless

# Scaling
kubectl scale statefulset web --replicas=5

# Rolling update
kubectl set image statefulset/web nginx=nginx:1.26

# Update specific Pods only (using partition)
kubectl patch statefulset web -p '{"spec":{"updateStrategy":{"rollingUpdate":{"partition":2}}}}'

# Delete (PVCs are retained)
kubectl delete statefulset web
kubectl delete pvc -l app=web  # Delete PVCs
```

---

## 4. Persistent Storage

### 3.1 Storage Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    Storage Hierarchy                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────┐               │
│  │              Pod                         │               │
│  │  ┌─────────────────────────────────┐   │               │
│  │  │     Volume Mount                 │   │               │
│  │  │     /data                        │   │               │
│  │  └─────────────┬───────────────────┘   │               │
│  └─────────────────┼───────────────────────┘               │
│                    │                                        │
│                    ▼                                        │
│  ┌─────────────────────────────────────────┐               │
│  │     PersistentVolumeClaim (PVC)         │               │
│  │     • Storage request                   │               │
│  │     • Namespace-scoped                  │               │
│  └─────────────────┬───────────────────────┘               │
│                    │ Binding                                │
│                    ▼                                        │
│  ┌─────────────────────────────────────────┐               │
│  │     PersistentVolume (PV)               │               │
│  │     • Actual storage                    │               │
│  │     • Cluster-scoped                    │               │
│  └─────────────────┬───────────────────────┘               │
│                    │                                        │
│                    ▼                                        │
│  ┌─────────────────────────────────────────┐               │
│  │     StorageClass                        │               │
│  │     • Dynamic provisioning              │               │
│  │     • Storage type definition           │               │
│  └─────────────────────────────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 StorageClass

```yaml
# storageclass.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: kubernetes.io/gce-pd  # Varies by cloud provider
parameters:
  type: pd-ssd
reclaimPolicy: Delete  # Delete or Retain — Delete auto-cleans cloud disks; use Retain for databases that need manual backup before deletion
allowVolumeExpansion: true  # Lets you grow PVCs in place — avoids data migration when storage needs increase
volumeBindingMode: WaitForFirstConsumer  # Delays provisioning until a pod is scheduled — ensures the disk is in the same zone as the pod

---
# AWS EBS StorageClass
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: aws-fast
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
reclaimPolicy: Delete
allowVolumeExpansion: true

---
# Local StorageClass
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: local-storage
provisioner: kubernetes.io/no-provisioner
volumeBindingMode: WaitForFirstConsumer
```

### 3.3 PersistentVolume (Static Provisioning)

```yaml
# pv-static.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-manual
  labels:
    type: local
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: manual
  hostPath:
    path: /mnt/data

---
# NFS PV
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-nfs
spec:
  capacity:
    storage: 100Gi
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: nfs
  nfs:
    server: nfs-server.example.com
    path: /exports/data

---
# AWS EBS PV
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-aws-ebs
spec:
  capacity:
    storage: 50Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: aws-fast
  awsElasticBlockStore:
    volumeID: vol-0123456789abcdef
    fsType: ext4
```

### 3.4 PersistentVolumeClaim

```yaml
# pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: app-data-pvc
  namespace: production
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: fast  # Dynamic provisioning
  # selector:             # Use for static binding
  #   matchLabels:
  #     type: local

---
# Using PVC in Pod
apiVersion: v1
kind: Pod
metadata:
  name: app-with-storage
spec:
  containers:
  - name: app
    image: myapp:latest
    volumeMounts:
    - name: data
      mountPath: /app/data
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: app-data-pvc
```

### 3.5 Volume Expansion and Snapshots

```yaml
# Expand PVC (requires allowVolumeExpansion: true)
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: expandable-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi  # Expanded from 5Gi
  storageClassName: fast

---
# VolumeSnapshot (requires CSI driver)
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: data-snapshot
spec:
  volumeSnapshotClassName: csi-snapclass
  source:
    persistentVolumeClaimName: app-data-pvc

---
# Restore PVC from snapshot
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: restored-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: fast
  dataSource:
    name: data-snapshot
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
```

---

## 5. ConfigMap Advanced

### 4.1 ConfigMap Creation Methods

```bash
# Create from literals
kubectl create configmap app-config \
  --from-literal=LOG_LEVEL=info \
  --from-literal=MAX_CONNECTIONS=100

# Create from file
kubectl create configmap nginx-config \
  --from-file=nginx.conf

# Create from directory
kubectl create configmap app-configs \
  --from-file=config/
```

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  # Simple key-value
  LOG_LEVEL: "info"
  DATABASE_HOST: "db.example.com"

  # File format
  app.properties: |
    server.port=8080
    server.host=0.0.0.0
    logging.level=INFO

  nginx.conf: |
    server {
        listen 80;
        server_name localhost;

        location / {
            root /usr/share/nginx/html;
            index index.html;
        }

        location /api {
            proxy_pass http://backend:8080;
        }
    }
```

### 4.2 ConfigMap Usage

```yaml
# configmap-usage.yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
  - name: app
    image: myapp:latest

    # Inject as environment variable
    env:
    - name: LOG_LEVEL
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: LOG_LEVEL

    # Entire ConfigMap as environment variables
    envFrom:
    - configMapRef:
        name: app-config
      prefix: APP_  # Optional prefix

    volumeMounts:
    # Mount as file
    - name: config-volume
      mountPath: /etc/app
    # Mount specific key only
    - name: nginx-volume
      mountPath: /etc/nginx/conf.d

  volumes:
  - name: config-volume
    configMap:
      name: app-config
      # All items
  - name: nginx-volume
    configMap:
      name: app-config
      items:
      - key: nginx.conf
        path: default.conf
        mode: 0644
```

### 4.3 ConfigMap Change Detection

```yaml
# Auto-reload (using Reloader)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-deployment
  annotations:
    # stakater/Reloader annotation
    reloader.stakater.com/auto: "true"
    # Or specific ConfigMap only
    configmap.reloader.stakater.com/reload: "app-config"
spec:
  template:
    spec:
      containers:
      - name: app
        image: myapp:latest
        volumeMounts:
        - name: config
          mountPath: /etc/app
      volumes:
      - name: config
        configMap:
          name: app-config

---
# Sidecar for change detection
apiVersion: v1
kind: Pod
metadata:
  name: app-with-reloader
spec:
  containers:
  - name: app
    image: myapp:latest
    volumeMounts:
    - name: config
      mountPath: /etc/app

  - name: config-reloader
    image: jimmidyson/configmap-reload:v0.5.0
    args:
    - --volume-dir=/etc/app
    - --webhook-url=http://localhost:8080/-/reload
    volumeMounts:
    - name: config
      mountPath: /etc/app
      readOnly: true

  volumes:
  - name: config
    configMap:
      name: app-config
```

---

## 6. DaemonSet and Job

### 5.1 DaemonSet

```yaml
# daemonset.yaml
# Deploy Pod on every node
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: node-exporter
  labels:
    app: node-exporter
spec:
  selector:
    matchLabels:
      app: node-exporter
  template:
    metadata:
      labels:
        app: node-exporter
    spec:
      # Deploy on specific nodes only
      nodeSelector:
        monitoring: "true"

      tolerations:
      # Deploy on master nodes too — monitoring must cover control-plane nodes for full cluster visibility
      - key: node-role.kubernetes.io/control-plane
        operator: Exists
        effect: NoSchedule

      containers:
      - name: node-exporter
        image: prom/node-exporter:v1.6.1
        ports:
        - containerPort: 9100
          hostPort: 9100
        volumeMounts:
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: sys
          mountPath: /host/sys
          readOnly: true
        resources:
          limits:
            cpu: 200m
            memory: 100Mi
          requests:
            cpu: 100m
            memory: 50Mi

      hostNetwork: true  # Needed so the exporter can see host-level network metrics and bind to the node's IP
      hostPID: true  # Required to enumerate host processes for per-process metrics

      volumes:
      - name: proc
        hostPath:
          path: /proc
      - name: sys
        hostPath:
          path: /sys

  updateStrategy:
    type: RollingUpdate  # Zero-downtime: updates one node at a time so monitoring coverage is never fully lost
    rollingUpdate:
      maxUnavailable: 1  # Only one node loses its exporter during rollout — balances speed vs observability gap
```

### 5.2 Job

```yaml
# job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: data-migration
spec:
  # Completion conditions
  completions: 1        # Number of successful Pods required
  parallelism: 1        # Number of concurrent Pods
  backoffLimit: 3       # Retry count on failure — prevents infinite retry loops on permanent errors
  activeDeadlineSeconds: 600  # Maximum execution time — hard kill prevents runaway jobs from consuming resources forever

  # Delete after completion — auto-cleanup keeps the cluster tidy and avoids accumulating finished pods
  ttlSecondsAfterFinished: 3600

  template:
    spec:
      restartPolicy: Never  # OnFailure or Never
      containers:
      - name: migrator
        image: myapp/migrator:latest
        command: ["python", "migrate.py"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url

---
# Parallel Job
apiVersion: batch/v1
kind: Job
metadata:
  name: parallel-processing
spec:
  completions: 10     # Total 10 completions
  parallelism: 3      # 3 at a time
  template:
    spec:
      restartPolicy: OnFailure
      containers:
      - name: worker
        image: myapp/worker:latest
```

### 5.3 CronJob

```yaml
# cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: db-backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  timeZone: "Asia/Seoul"

  # Concurrency policy — Forbid prevents overlapping backup runs that could corrupt data
  concurrencyPolicy: Forbid  # Allow, Forbid, or Replace

  # Starting deadline — if the scheduler misses the window by >300s, skip this run rather than starting a stale backup
  startingDeadlineSeconds: 300

  # Success/failure history — keep enough history for debugging but avoid cluttering etcd with old Job objects
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 1

  # Suspend
  suspend: false

  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: backup
            image: postgres:15
            command:
            - /bin/sh
            - -c
            - |
              pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME > /backup/db_$(date +%Y%m%d).sql
              aws s3 cp /backup/db_$(date +%Y%m%d).sql s3://backups/
            env:
            - name: DB_HOST
              value: "postgres"
            - name: DB_USER
              valueFrom:
                secretKeyRef:
                  name: db-secret
                  key: username
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: db-secret
                  key: password
            volumeMounts:
            - name: backup
              mountPath: /backup
          volumes:
          - name: backup
            emptyDir: {}
```

---

## 7. Advanced Scheduling

### 6.1 Node Affinity

```yaml
# node-affinity.yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  affinity:
    nodeAffinity:
      # Required conditions
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: gpu
            operator: In
            values:
            - "true"
          - key: kubernetes.io/arch
            operator: In
            values:
            - amd64

      # Preferred conditions
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        preference:
          matchExpressions:
          - key: gpu-type
            operator: In
            values:
            - nvidia-a100
      - weight: 50
        preference:
          matchExpressions:
          - key: gpu-type
            operator: In
            values:
            - nvidia-v100

  containers:
  - name: gpu-app
    image: nvidia/cuda:12.0-base
    resources:
      limits:
        nvidia.com/gpu: 1
```

### 6.2 Pod Affinity/Anti-Affinity

```yaml
# pod-affinity.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-server
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
      affinity:
        # Prefer same node as cache Pod
        podAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - cache
              topologyKey: kubernetes.io/hostname

        # Deploy on different nodes than other Pods of same app — ensures a single node failure doesn't take down all replicas
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - web
            topologyKey: kubernetes.io/hostname

      containers:
      - name: web
        image: nginx:latest
```

### 6.3 Taints and Tolerations

```bash
# Add Taint to node
kubectl taint nodes node1 dedicated=gpu:NoSchedule
kubectl taint nodes node2 special=true:PreferNoSchedule
kubectl taint nodes node3 critical=true:NoExecute

# Remove Taint
kubectl taint nodes node1 dedicated=gpu:NoSchedule-
```

```yaml
# tolerations.yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  tolerations:
  # Exact match
  - key: "dedicated"
    operator: "Equal"
    value: "gpu"
    effect: "NoSchedule"

  # Key exists
  - key: "special"
    operator: "Exists"
    effect: "PreferNoSchedule"

  # NoExecute + tolerationSeconds
  - key: "critical"
    operator: "Equal"
    value: "true"
    effect: "NoExecute"
    tolerationSeconds: 3600  # Evict after 1 hour

  containers:
  - name: app
    image: myapp:latest
```

### 6.4 Topology Spread Constraints

```yaml
# topology-spread.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: distributed-app
spec:
  replicas: 6
  selector:
    matchLabels:
      app: distributed
  template:
    metadata:
      labels:
        app: distributed
    spec:
      topologySpreadConstraints:
      # Even distribution across zones — survives an entire AZ outage with minimal capacity loss
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule  # Hard constraint: refuse to schedule rather than create an imbalanced deployment
        labelSelector:
          matchLabels:
            app: distributed

      # Even distribution across nodes
      - maxSkew: 1
        topologyKey: kubernetes.io/hostname
        whenUnsatisfiable: ScheduleAnyway
        labelSelector:
          matchLabels:
            app: distributed

      containers:
      - name: app
        image: myapp:latest
```

---

## 8. Practice Exercises

### Exercise 1: Microservices Ingress
```yaml
# Requirements:
# - api.example.com/v1/* → api-v1-service
# - api.example.com/v2/* → api-v2-service
# - TLS enabled, HTTP→HTTPS redirect
# - Rate limiting applied

# Write Ingress
```

### Exercise 2: Redis Cluster StatefulSet
```yaml
# Requirements:
# - 3-node Redis Cluster
# - 1Gi PVC for each node
# - Headless Service for inter-node communication
# - Appropriate resource limits

# Write StatefulSet
```

### Exercise 3: Log Collection DaemonSet
```yaml
# Requirements:
# - Collect /var/log from all nodes
# - Send to Elasticsearch
# - Manage config with ConfigMap
# - Deploy on master nodes too

# Write DaemonSet
```

### Exercise 4: Batch Data Processing Job
```yaml
# Requirements:
# - Process 100 data items (completions: 100)
# - Process 10 at a time (parallelism: 10)
# - Retry 3 times on failure
# - Must complete within 2 hours
# - Delete 24 hours after completion

# Write Job
```

---

## References

- [Kubernetes Ingress](https://kubernetes.io/docs/concepts/services-networking/ingress/)
- [Kubernetes Gateway API](https://gateway-api.sigs.k8s.io/)
- [StatefulSets](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/)
- [Persistent Volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)
- [DaemonSet](https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/)

---

## Exercises

### Exercise 1: Deploy a Stateful Application with StatefulSet

Experience the difference between a Deployment and a StatefulSet by deploying a replicated database.

1. Write a StatefulSet manifest for a 3-replica Redis cluster using `redis:7-alpine`
2. Include a headless Service (`clusterIP: None`) with the same selector
3. Apply both manifests and observe that each Pod gets a stable, ordered name (`redis-0`, `redis-1`, `redis-2`)
4. Exec into `redis-0` and set a key: `redis-cli set mykey "hello"`
5. Delete `redis-0` and observe Kubernetes recreate it with the same name and ordinal
6. Verify the key is gone after recreation (no persistent storage yet) — note the difference from a PVC-backed StatefulSet

### Exercise 2: Provision Persistent Storage with PVC

Attach a PersistentVolumeClaim to a StatefulSet to give each replica its own storage.

1. Update the StatefulSet from Exercise 1 to include a `volumeClaimTemplate` requesting 1Gi of storage
2. Apply and confirm each Pod has its own PVC: `kubectl get pvc`
3. Exec into `redis-0` and set a key, then delete the Pod
4. After `redis-0` restarts, exec back in and confirm the key is still present
5. Run `kubectl get pv` to see the dynamically provisioned PersistentVolume
6. Delete the StatefulSet and observe whether the PVCs are retained or deleted

### Exercise 3: Expose a Service with Ingress

Configure an Ingress resource to route external HTTP traffic to multiple Services.

1. Enable the Nginx Ingress controller on minikube: `minikube addons enable ingress`
2. Create two Deployments and their ClusterIP Services: `app-v1` on port 8080 and `app-v2` on port 8081
3. Write an Ingress manifest that routes:
   - `/v1` path → `app-v1` Service
   - `/v2` path → `app-v2` Service
4. Apply the Ingress and get its IP: `kubectl get ingress`
5. Add the IP to `/etc/hosts` (e.g., `192.168.x.x myapp.local`)
6. Test routing: `curl http://myapp.local/v1` and `curl http://myapp.local/v2`

### Exercise 4: Run a Batch Workload with Job and CronJob

Use Job and CronJob resources to run one-time and scheduled batch tasks.

1. Write a Job manifest that runs a container, prints "Batch job complete", and exits with code 0
2. Apply it and observe the Pod run to completion: `kubectl get jobs` and `kubectl get pods`
3. Read the job logs: `kubectl logs job/my-job`
4. Write a CronJob that runs the same task every minute using schedule `*/1 * * * *`
5. Wait 2 minutes and confirm multiple Jobs were created: `kubectl get jobs`
6. Set `successfulJobsHistoryLimit: 3` and `failedJobsHistoryLimit: 1` in the CronJob spec and apply it

### Exercise 5: Control Pod Placement with Node Affinity

Use scheduling rules to influence which nodes workloads run on.

1. Label one of your minikube nodes (or the control-plane): `kubectl label node minikube tier=frontend`
2. Write a Deployment that uses `nodeAffinity` to require placement on nodes with `tier=frontend`
3. Apply it and verify the Pods are scheduled on the labeled node: `kubectl get pods -o wide`
4. Add a second label `tier=backend` to a different node (or remove and re-add on the same node for minikube)
5. Create a second Deployment with `preferredDuringSchedulingIgnoredDuringExecution` affinity for `tier=backend` nodes
6. Observe the scheduling behavior — preferred affinity does not block scheduling if no matching node exists

---

**Previous**: [Kubernetes Security](./07_Kubernetes_Security.md) | **Next**: [Helm Package Management](./09_Helm_Package_Management.md)

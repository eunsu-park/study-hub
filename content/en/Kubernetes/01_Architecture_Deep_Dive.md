# 01. Architecture Deep Dive

**Previous**: [Overview](./00_Overview.md) | **Next**: [Workload Resources](./02_Workload_Resources.md)

## Learning Objectives
- Understand the Kubernetes control plane and node components in depth
- Trace the full lifecycle of an API request from kubectl to etcd
- Explain the etcd data model and how Kubernetes objects are stored
- Describe the scheduler algorithm including filtering and scoring phases
- Articulate how control loops and reconciliation drive the declarative model

---

Kubernetes is a distributed system designed around a declarative API and a set of
independent controllers that continuously reconcile actual state with desired state.
Understanding its architecture is essential for effective operation, debugging, and
extension. This lesson dissects every major component, the request lifecycle, and
the control loop pattern that makes Kubernetes self-healing.

## Table of Contents
1. [High-Level Architecture](#1-high-level-architecture)
2. [Control Plane Components](#2-control-plane-components)
3. [Node Components](#3-node-components)
4. [API Request Lifecycle](#4-api-request-lifecycle)
5. [etcd Data Model](#5-etcd-data-model)
6. [Kubernetes Object Model (GVR / GVK)](#6-kubernetes-object-model-gvr--gvk)
7. [Authentication and Authorization Flow](#7-authentication-and-authorization-flow)
8. [Scheduler Algorithm](#8-scheduler-algorithm)
9. [Control Loops and Reconciliation](#9-control-loops-and-reconciliation)
10. [Exercises](#exercises)

---

## 1. High-Level Architecture

Kubernetes follows a hub-and-spoke topology. The **control plane** is the hub,
exposing the API server as the single entry point. **Nodes** (worker machines) are
the spokes, each running an agent (kubelet) that communicates with the API server.

```
┌──────────────────────────────────────────────────────────┐
│                     Control Plane                        │
│  ┌────────────┐  ┌───────┐  ┌───────────┐  ┌─────────┐ │
│  │ API Server │──│ etcd  │  │ Scheduler │  │ CM / CCM│ │
│  └─────┬──────┘  └───────┘  └───────────┘  └─────────┘ │
│        │                                                 │
└────────┼─────────────────────────────────────────────────┘
         │  Watch / List / Update
    ┌────┴────┬────────────┬────────────┐
    │ Node 1  │  Node 2    │  Node N    │
    │ kubelet │  kubelet   │  kubelet   │
    │ kproxy  │  kproxy    │  kproxy    │
    │ runtime │  runtime   │  runtime   │
    └─────────┴────────────┴────────────┘
```

### 1.1 Design Principles

- **Declarative over imperative**: Users declare desired state; controllers converge.
- **Level-triggered, not edge-triggered**: Controllers react to the current state of
  the world, not to events. If a controller restarts, it re-reads the full state and
  continues from there.
- **No single point of failure in the data path**: The control plane can go down
  temporarily without affecting running workloads. Pods keep running; only changes
  are delayed.

### 1.2 Cluster Topology in Production

A production-grade cluster typically runs:

| Component | Recommended Count | Notes |
|-----------|------------------|-------|
| API server | 3+ (behind LB) | Stateless; horizontal scaling |
| etcd | 3 or 5 (odd number) | Raft quorum requires majority |
| Scheduler | 1 active + standby | Leader election via Lease |
| Controller Manager | 1 active + standby | Leader election via Lease |
| Worker nodes | Varies | Up to 5,000 per cluster |

### 1.3 Exploring with Minikube

```bash
# Start a local cluster
minikube start --nodes 2 --driver=docker

# List nodes
kubectl get nodes -o wide

# See control plane pods
kubectl get pods -n kube-system

# Inspect the API server
kubectl describe pod -n kube-system kube-apiserver-minikube
```

---

## 2. Control Plane Components

### 2.1 kube-apiserver

The API server is the front door to the cluster. Every operation—whether from
`kubectl`, the dashboard, or an in-cluster controller—goes through the API server.

Key responsibilities:
- **RESTful API**: CRUD operations on all Kubernetes objects
- **Admission control**: Mutating and validating webhooks
- **Authentication & authorization**: Plugin-based (see Section 7)
- **Watch mechanism**: Long-lived HTTP streams for change notification
- **OpenAPI schema serving**: Enables client-side validation

```bash
# Check the API server's command-line arguments
kubectl -n kube-system get pod kube-apiserver-minikube -o jsonpath='{.spec.containers[0].command}' | python3 -m json.tool
```

The API server is **stateless**. All persistent state is stored in etcd. This means
you can run multiple API server replicas behind a load balancer for high availability.

### 2.2 etcd

etcd is a distributed, strongly consistent key-value store that serves as the
cluster's single source of truth.

Properties:
- **Raft consensus**: Tolerates `(n-1)/2` node failures for `n` members
- **Serializable reads by default**: Configurable for linearizable reads
- **Watch support**: Clients receive ordered change notifications
- **MVCC**: Multi-version concurrency control enables efficient watch and compaction

```bash
# Inside minikube, query etcd directly (for learning only)
minikube ssh

# etcd stores data under /registry/<resource>/<namespace>/<name>
# Example: /registry/pods/default/my-pod

# Check etcd health (if etcdctl is available)
ETCDCTL_API=3 etcdctl \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/var/lib/minikube/certs/etcd/ca.crt \
  --cert=/var/lib/minikube/certs/etcd/server.crt \
  --key=/var/lib/minikube/certs/etcd/server.key \
  endpoint health
```

### 2.3 kube-scheduler

The scheduler watches for newly created Pods that have no assigned node and selects
the best node for each one. Its algorithm is covered in detail in Section 8.

```bash
# View scheduler configuration
kubectl -n kube-system describe pod kube-scheduler-minikube

# Check scheduler logs
kubectl -n kube-system logs kube-scheduler-minikube --tail=20
```

### 2.4 kube-controller-manager

The controller manager runs a suite of controllers, each implementing a specific
control loop. Examples:

| Controller | Watches | Manages |
|------------|---------|---------|
| Deployment | Deployments | ReplicaSets |
| ReplicaSet | ReplicaSets | Pods |
| Node | Nodes | Taints, evictions |
| Job | Jobs | Pods |
| Endpoint | Services, Pods | Endpoints |
| ServiceAccount | Namespaces | Default SA + token |
| Garbage Collector | Owner references | Cascading deletion |

All controllers share one process but are logically independent. Each controller
operates on a **work queue** fed by informers (cached watch streams).

```bash
# List all controllers running inside the controller manager
kubectl -n kube-system get pod kube-controller-manager-minikube \
  -o jsonpath='{.spec.containers[0].command}' | tr ',' '\n' | grep controllers
```

### 2.5 cloud-controller-manager

In cloud environments, a separate binary handles cloud-specific logic:
- Node lifecycle (detecting when a VM is deleted)
- Route configuration
- Load balancer provisioning
- Volume attachment (legacy; now CSI)

This separation allows cloud providers to release at their own cadence.

---

## 3. Node Components

### 3.1 kubelet

The kubelet is the primary agent on every node. It registers the node with the
API server and ensures containers described by PodSpecs are running and healthy.

Responsibilities:
- Pod lifecycle management (start, stop, restart)
- Liveness, readiness, and startup probe execution
- Resource reporting (CPU, memory, storage, PIDs)
- Container log management
- CSI and device plugin interfaces
- Static pod management (reads from a local directory)

```bash
# Check kubelet status on minikube
minikube ssh -- systemctl status kubelet

# View kubelet logs
minikube ssh -- journalctl -u kubelet --no-pager --tail=30
```

### 3.2 kube-proxy

kube-proxy implements the Service abstraction by programming network rules on each
node. It watches Service and EndpointSlice objects and configures:

- **iptables mode** (default): Creates NAT rules for service IP to pod IP translation
- **IPVS mode**: Uses Linux IPVS kernel module for L4 load balancing
- **nftables mode** (v1.29+): Uses nftables instead of iptables

```bash
# Check kube-proxy mode
kubectl -n kube-system get configmap kube-proxy -o yaml | grep mode

# List iptables rules created by kube-proxy (on a node)
minikube ssh -- sudo iptables -t nat -L KUBE-SERVICES -n | head -20
```

### 3.3 Container Runtime

Kubernetes communicates with container runtimes through the **Container Runtime
Interface (CRI)**. Popular runtimes:

| Runtime | Description |
|---------|------------|
| containerd | Industry standard, extracted from Docker |
| CRI-O | OCI-native runtime designed for Kubernetes |
| Docker (via cri-dockerd) | Legacy; requires shim |

```bash
# Check which runtime minikube uses
minikube ssh -- crictl info | head -5

# List running containers via CRI
minikube ssh -- crictl ps
```

---

## 4. API Request Lifecycle

When you run `kubectl apply -f pod.yaml`, the request goes through a precise
pipeline:

```
kubectl → HTTP request → API Server
                           │
                    ┌──────┴───────┐
                    │ Authentication│  (Who are you?)
                    ├──────────────┤
                    │ Authorization │  (Can you do this?)
                    ├──────────────┤
                    │ Admission     │  (Mutating → Validating)
                    ├──────────────┤
                    │ Validation    │  (Schema + custom)
                    ├──────────────┤
                    │ etcd Write    │  (Persist object)
                    ├──────────────┤
                    │ Post-hooks    │  (Informer notification)
                    └──────────────┘
```

### 4.1 Step-by-Step Trace

1. **kubectl** reads kubeconfig, discovers the API server URL and credentials
2. **HTTP request** is constructed: `POST /api/v1/namespaces/default/pods`
3. **TLS termination**: The API server verifies the client certificate (or token)
4. **Authentication**: One of the configured authenticators validates the identity
5. **Authorization**: RBAC (or another authorizer) checks if the user can create a
   pod in the `default` namespace
6. **Mutating admission**: Webhooks may inject sidecars, set defaults, add labels
7. **Object defaults**: The API server fills in missing fields (e.g., `restartPolicy`)
8. **Validating admission**: Webhooks may reject the request (e.g., policy checks)
9. **Schema validation**: The object is validated against its OpenAPI schema
10. **etcd write**: The object is serialized (protobuf) and written to etcd
11. **Watch notifications**: All watchers (scheduler, controllers) are notified

### 4.2 Observing the Lifecycle

```bash
# Enable verbose output to see the full request
kubectl apply -f pod.yaml -v=8

# Trace API calls with audit logging (requires API server flag)
# --audit-log-path=/var/log/kube-audit.log
# --audit-policy-file=/etc/kubernetes/audit-policy.yaml
```

Example audit policy:

```yaml
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
  - level: RequestResponse
    resources:
      - group: ""
        resources: ["pods"]
    verbs: ["create", "update", "delete"]
  - level: Metadata
    resources:
      - group: ""
        resources: ["services", "configmaps"]
  - level: None
    resources:
      - group: ""
        resources: ["events"]
```

---

## 5. etcd Data Model

### Theory: etcd and the Raft Consensus Algorithm

Every Kubernetes object lives in **etcd**, a distributed key-value store that uses **Raft** for consensus. Raft is what makes etcd correct under partial failure: even if a control plane node crashes mid-write, you never get two API servers reporting different states for the same pod.

Raft elects a single **leader** among (typically 3 or 5) etcd members. All writes go through the leader, which appends them to a replicated log. A write is only acknowledged after a **quorum** (majority) of members have persisted the log entry. This gives two key properties:

- **Linearizability**: once a write is acknowledged, all subsequent reads see it. There are no stale reads from etcd's perspective.
- **Split-brain immunity**: a minority partition cannot make progress (cannot reach quorum), so two halves of a partitioned cluster cannot both elect leaders and accept writes.

Why an **odd** number of etcd members? A 3-member cluster tolerates 1 failure (2/3 still quorum). A 4-member cluster also tolerates only 1 failure (3/4 needed for majority) but doubles the write latency for the same fault tolerance. So odd numbers (3, 5, 7) are strictly better. Production clusters virtually always run 3 or 5.

Inside etcd, Kubernetes objects are stored as protobuf-encoded values under hierarchical keys like `/registry/pods/default/my-pod`. The API server is the only client that talks to etcd; everyone else talks to the API server. This is the **single source of truth** principle made literal.

### 5.1 Key Structure

etcd stores Kubernetes objects under a hierarchical key scheme:

```
/registry/<api-group>/<resource>/<namespace>/<name>
```

Examples:
```
/registry/pods/default/nginx
/registry/deployments/kube-system/coredns
/registry/services/specs/default/kubernetes
/registry/clusterroles/cluster-admin
```

Cluster-scoped resources omit the namespace component:
```
/registry/nodes/worker-1
/registry/namespaces/production
```

### 5.2 Serialization Format

By default, objects are stored in **Protocol Buffers** format for efficiency.
The API server handles conversion between JSON (client-facing) and protobuf
(storage).

### 5.3 Resource Versions

Every object in etcd has a `resourceVersion` field, which maps to the etcd
**modified revision**. This is used for:

- **Optimistic concurrency**: Updates must include the current `resourceVersion`;
  conflicts return `409 Conflict`
- **Watch bookmarks**: Clients resume watching from a specific revision
- **List pagination**: `continue` tokens encode revision information

```bash
# Observe resourceVersion
kubectl get pod nginx -o jsonpath='{.metadata.resourceVersion}'

# Attempt a conflicting update (will fail if someone else modified it)
kubectl get pod nginx -o yaml > pod.yaml
# Edit pod.yaml, then:
kubectl replace -f pod.yaml
```

### 5.4 Compaction and Defragmentation

etcd keeps a history of all revisions. Over time, this consumes disk space.

```bash
# Check etcd database size
ETCDCTL_API=3 etcdctl \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/var/lib/minikube/certs/etcd/ca.crt \
  --cert=/var/lib/minikube/certs/etcd/server.crt \
  --key=/var/lib/minikube/certs/etcd/server.key \
  endpoint status --write-out=table
```

The API server automatically compacts old revisions (default: 5 minutes).

---

## 6. Kubernetes Object Model (GVR / GVK)

### 6.1 Group, Version, Resource (GVR)

Every REST endpoint in Kubernetes is identified by a GVR:

| Component | Example | Description |
|-----------|---------|-------------|
| Group | `apps` | API group (empty string for core) |
| Version | `v1` | API version |
| Resource | `deployments` | Plural resource name |

REST path: `/apis/{group}/{version}/namespaces/{ns}/{resource}/{name}`

Core group uses `/api/v1/...` (no group in path).

### 6.2 Group, Version, Kind (GVK)

The GVK identifies the **Go type** of an object:

| Component | Example |
|-----------|---------|
| Group | `apps` |
| Version | `v1` |
| Kind | `Deployment` |

The mapping between GVK and GVR is maintained by the **REST mapper**.

```bash
# Discover all API resources (GVR + Kind)
kubectl api-resources

# Get details for a specific resource
kubectl api-resources | grep -i deployment

# Explore the OpenAPI schema
kubectl get --raw /openapi/v2 | python3 -m json.tool | head -50
```

### 6.3 Writing a Go Client with GVR

```go
package main

import (
	"context"
	"fmt"
	"os"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/clientcmd"
)

func main() {
	// Build config from kubeconfig
	config, err := clientcmd.BuildConfigFromFlags("", os.Getenv("KUBECONFIG"))
	if err != nil {
		panic(err)
	}

	// Create dynamic client
	client, err := dynamic.NewForConfig(config)
	if err != nil {
		panic(err)
	}

	// Define the GVR for Deployments
	gvr := schema.GroupVersionResource{
		Group:    "apps",
		Version:  "v1",
		Resource: "deployments",
	}

	// List deployments in the default namespace
	deployments, err := client.Resource(gvr).Namespace("default").List(
		context.TODO(),
		metav1.ListOptions{},
	)
	if err != nil {
		panic(err)
	}

	for _, d := range deployments.Items {
		fmt.Printf("Deployment: %s (replicas: %v)\n",
			d.GetName(),
			d.Object["spec"].(map[string]interface{})["replicas"],
		)
	}
}
```

### 6.4 Custom Resources

Custom Resource Definitions (CRDs) extend the API with new GVR/GVK pairs:

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: certificates.cert-manager.io
spec:
  group: cert-manager.io
  versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                secretName:
                  type: string
                issuerRef:
                  type: object
                  properties:
                    name:
                      type: string
                    kind:
                      type: string
  scope: Namespaced
  names:
    plural: certificates
    singular: certificate
    kind: Certificate
    shortNames:
      - cert
```

---

## 7. Authentication and Authorization Flow

### 7.1 Authentication (AuthN)

The API server supports multiple authentication strategies, evaluated in order:

| Method | Mechanism | Common Use |
|--------|-----------|-----------|
| X.509 client certs | TLS client certificate | kubeadm clusters |
| Bearer tokens | `Authorization: Bearer <token>` | Service accounts |
| OIDC tokens | JWT from identity provider | Human users (SSO) |
| Webhook token review | External auth service | Custom integrations |
| Bootstrap tokens | Short-lived tokens | Node bootstrapping |

```bash
# Check your current identity
kubectl auth whoami

# View the kubeconfig authentication info
kubectl config view --minify -o jsonpath='{.users[0]}'
```

### 7.2 Service Account Tokens

Every pod gets a projected service account token mounted at
`/var/run/secrets/kubernetes.io/serviceaccount/token`.

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: my-app
  namespace: default
automountServiceAccountToken: true
---
apiVersion: v1
kind: Pod
metadata:
  name: my-app-pod
spec:
  serviceAccountName: my-app
  containers:
    - name: app
      image: curlimages/curl:8.5.0
      command:
        - sh
        - -c
        - |
          TOKEN=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)
          curl -s -k -H "Authorization: Bearer $TOKEN" \
            https://kubernetes.default.svc/api/v1/namespaces/default/pods
```

### 7.3 Authorization (AuthZ)

After authentication, the API server checks authorization. The default mode is
**RBAC** (Role-Based Access Control).

RBAC objects:
- **Role** / **ClusterRole**: Define permissions (verbs on resources)
- **RoleBinding** / **ClusterRoleBinding**: Bind roles to subjects

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: production
  name: pod-reader
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "watch", "list"]
  - apiGroups: [""]
    resources: ["pods/log"]
    verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-pods
  namespace: production
subjects:
  - kind: ServiceAccount
    name: monitoring-agent
    namespace: production
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
```

```bash
# Check if a user can perform an action
kubectl auth can-i create deployments --namespace=production

# Check as a specific service account
kubectl auth can-i get pods \
  --as=system:serviceaccount:production:monitoring-agent \
  --namespace=production

# List all roles in a namespace
kubectl get roles -n production

# Describe a ClusterRole
kubectl describe clusterrole cluster-admin
```

### 7.4 Admission Control

After authorization, the request passes through admission controllers:

**Mutating admission** (modifies the object):
- `MutatingAdmissionWebhook`
- `DefaultStorageClass`
- `PodPreset` (deprecated)

**Validating admission** (rejects the object):
- `ValidatingAdmissionWebhook`
- `ValidatingAdmissionPolicy` (CEL-based, v1.30 GA)
- `ResourceQuota`
- `LimitRanger`

```yaml
# Example: ValidatingAdmissionPolicy (no webhook needed)
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
metadata:
  name: require-labels
spec:
  failurePolicy: Fail
  matchConstraints:
    resourceRules:
      - apiGroups: ["apps"]
        apiVersions: ["v1"]
        operations: ["CREATE", "UPDATE"]
        resources: ["deployments"]
  validations:
    - expression: "has(object.metadata.labels) && 'app' in object.metadata.labels"
      message: "All deployments must have an 'app' label"
```

---

## 8. Scheduler Algorithm

### Theory: The Scheduler: A Two-Phase Optimizer

When you create a Pod with no `nodeName` set, it lands in etcd in a "Pending" state. The scheduler watches for these unscheduled pods and, for each one, runs a two-phase algorithm:

**Phase 1 — Filtering (Predicates).** The scheduler evaluates each node against a series of hard constraints:
- Does the node have enough CPU and memory for the pod's requests?
- Does the pod tolerate the node's taints?
- Does the pod's node selector / affinity match this node's labels?
- Does the volume the pod claims actually attach to this node's zone?

Nodes that fail any predicate are eliminated. If zero nodes survive filtering, the pod stays Pending and the scheduler logs `FailedScheduling`.

**Phase 2 — Scoring (Priorities).** Surviving nodes are ranked by soft preferences:
- `LeastAllocated`: prefer nodes with more free resources (spread).
- `BalancedResourceAllocation`: prefer nodes where CPU and memory utilization are balanced.
- `ImageLocality`: prefer nodes that already have the container image cached.
- `InterPodAffinity`: prefer nodes that satisfy pod-affinity rules (e.g., place a cache pod near its app).

Each scorer returns 0–100; the scheduler weights and sums them, then picks the highest-scoring node. The choice is committed by **binding** — a single API call (`POST /pods/{name}/binding`) that writes `spec.nodeName` to the pod object. The kubelet on that node sees the watch event and starts the containers.

If a high-priority pod cannot be scheduled, **preemption** kicks in: the scheduler finds lower-priority pods to evict so the high-priority one fits. This is why pod priority classes matter for production — they govern who gets evicted under contention.

### 8.1 Overview

The scheduler assigns pods to nodes in two phases:

1. **Filtering** (Predicates): Eliminate nodes that cannot run the pod
2. **Scoring** (Priorities): Rank remaining nodes and pick the best

```
Unscheduled Pod
      │
      ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Filtering  │───▶│   Scoring    │───▶│  Binding    │
│ (eliminate) │    │ (rank 0-100) │    │ (assign)    │
└─────────────┘    └──────────────┘    └─────────────┘
  N nodes            M ≤ N nodes          1 node
```

### 8.2 Filtering Plugins

| Plugin | What it checks |
|--------|---------------|
| NodeResourcesFit | CPU/memory requests fit available capacity |
| NodeAffinity | Node selector and affinity rules match |
| TaintToleration | Pod tolerates node taints |
| PodTopologySpread | Topology constraints are satisfiable |
| VolumeBinding | Required PVs can be bound on this node |
| InterPodAffinity | Pod affinity/anti-affinity constraints |
| NodePorts | Requested host ports are available |
| NodeUnschedulable | Node is not cordoned |

### 8.3 Scoring Plugins

| Plugin | Strategy |
|--------|----------|
| NodeResourcesBalancedAllocation | Prefer balanced CPU/memory usage |
| NodeResourcesFit (LeastAllocated) | Prefer nodes with more free resources |
| InterPodAffinity | Score based on pod affinity preferences |
| TaintToleration | Prefer nodes with fewer untolerated taints |
| ImageLocality | Prefer nodes that already have the container image |
| PodTopologySpread | Prefer even distribution across topology domains |

Each plugin returns a score from 0 to 100. The scheduler sums weighted scores
and picks the node with the highest total.

### 8.4 Scheduler Profiles

```yaml
apiVersion: kubescheduler.config.k8s.io/v1
kind: KubeSchedulerConfiguration
profiles:
  - schedulerName: default-scheduler
    plugins:
      score:
        enabled:
          - name: NodeResourcesFit
            weight: 2
          - name: PodTopologySpread
            weight: 3
        disabled:
          - name: ImageLocality
      filter:
        enabled:
          - name: NodeResourcesFit
          - name: TaintToleration
```

### 8.5 Influencing Scheduling

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-workload
spec:
  # Node selector (simple)
  nodeSelector:
    accelerator: nvidia-a100

  # Node affinity (expressive)
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: topology.kubernetes.io/zone
                operator: In
                values: ["us-east-1a", "us-east-1b"]
      preferredDuringSchedulingIgnoredDuringExecution:
        - weight: 80
          preference:
            matchExpressions:
              - key: node.kubernetes.io/instance-type
                operator: In
                values: ["p4d.24xlarge"]

    # Pod anti-affinity
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        - labelSelector:
            matchLabels:
              app: gpu-workload
          topologyKey: kubernetes.io/hostname

  # Tolerations
  tolerations:
    - key: "nvidia.com/gpu"
      operator: "Exists"
      effect: "NoSchedule"

  containers:
    - name: trainer
      image: pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
      resources:
        requests:
          cpu: "4"
          memory: "16Gi"
          nvidia.com/gpu: "1"
        limits:
          nvidia.com/gpu: "1"
```

---

## 9. Control Loops and Reconciliation

### Theory: Watch, Don't Poll: How Components Talk Without Talking

Naively, you might expect controllers and kubelets to constantly poll the API server: "any new pods for me?" That would not scale to thousands of nodes and millions of objects. Instead, every long-running Kubernetes component opens a single HTTP connection and issues a **watch** request:

```
GET /api/v1/pods?watch=1&resourceVersion=12345
```

The API server holds the connection open and streams **deltas** (`ADDED`, `MODIFIED`, `DELETED` events) every time something changes, starting just after `resourceVersion=12345`. The client maintains a local cache (the **informer**), updating it from the stream. This means:

- The kubelet on every node has a real-time mirror of "the pods assigned to me" and reacts within milliseconds of an assignment.
- The Deployment controller has a mirror of "all Deployments and their child ReplicaSets" and triggers reconciliation the moment a replica count changes.
- Polling the API server is rare; it is mostly a one-time **list** (full snapshot) followed by an indefinite watch.

If a watch breaks (network blip), the client lists again from the last known `resourceVersion`. This is how the cluster tolerates control plane disruptions — components rebuild their view from etcd's truth and continue without restart.

### Theory: Controllers as Level-Triggered Reconciliation Loops

Every controller in `kube-controller-manager` (Deployment, ReplicaSet, Node, Job, Endpoints, ...) implements the same loop:

```
loop forever:
    desired = read from informer cache (mirrors API server / etcd)
    actual  = observe the world (or read status fields)
    if desired != actual:
        take action to converge (create/update/delete objects)
    else:
        do nothing
```

Two properties make this robust:

- **Level-triggered, not edge-triggered.** A controller does not say "I saw an event, I will react once." It says "the current state is X, the desired state is Y, I will act to make X = Y." If the controller crashes and restarts, it re-reads the state and resumes — no missed events to recover.
- **Idempotent actions.** Asking "create ReplicaSet R if it does not exist" has no effect on the second call. So even if a controller's action is duplicated (because of a retry, restart, or split work queue), the system converges to the same state.

The `controller-manager` typically runs all built-in controllers in a single process, but only **one replica is active at a time** thanks to a leader-election lease in etcd. Standby replicas wait for the lease to expire, then take over. Custom controllers (Operators, lesson 11) use the same pattern via the controller-runtime library.

This loop is *the* Kubernetes paradigm. Deployments, autoscalers, ingress controllers, cert-manager, ArgoCD — they are all the same algorithm applied to different desired-state schemas.

### 9.1 The Reconciliation Pattern

Every Kubernetes controller follows the same pattern:

```
          ┌──────────┐
          │  Observe  │  (Read current state from API server)
          └────┬─────┘
               │
          ┌────▼─────┐
          │   Diff    │  (Compare desired vs. actual)
          └────┬─────┘
               │
          ┌────▼─────┐
          │   Act     │  (Take action to converge)
          └────┬─────┘
               │
               └───────── (loop)
```

This pattern is **level-triggered**: the controller does not need to see every
event. It only needs the current state to decide what to do.

### 9.2 Informers and Work Queues

Controllers do not poll the API server. Instead, they use **informers**:

1. An informer does a `List` followed by a `Watch` on a resource type
2. Received objects are stored in a **local cache** (thread-safe store)
3. Events trigger **event handlers** that enqueue keys (`namespace/name`)
4. A **worker goroutine** dequeues keys and calls the `Reconcile` function
5. `Reconcile` reads the desired state from the cache and takes action

```go
package main

import (
	"context"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/workqueue"
)

func main() {
	config, _ := clientcmd.BuildConfigFromFlags("", "/path/to/kubeconfig")
	clientset, _ := kubernetes.NewForConfig(config)

	// Create a shared informer factory (resync every 30s)
	factory := informers.NewSharedInformerFactory(clientset, 30*time.Second)
	podInformer := factory.Core().V1().Pods().Informer()

	// Work queue
	queue := workqueue.NewRateLimitingQueue(
		workqueue.DefaultControllerRateLimiter(),
	)

	// Event handlers enqueue keys
	podInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			key, _ := cache.MetaNamespaceKeyFunc(obj)
			queue.Add(key)
			fmt.Printf("Pod added: %s\n", key)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			key, _ := cache.MetaNamespaceKeyFunc(newObj)
			queue.Add(key)
		},
		DeleteFunc: func(obj interface{}) {
			key, _ := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
			queue.Add(key)
			fmt.Printf("Pod deleted: %s\n", key)
		},
	})

	// Start the informer
	ctx := context.Background()
	factory.Start(ctx.Done())
	factory.WaitForCacheSync(ctx.Done())

	// Worker loop
	go wait.Until(func() {
		for {
			key, shutdown := queue.Get()
			if shutdown {
				return
			}
			// Reconcile logic goes here
			fmt.Printf("Reconciling: %s\n", key)
			queue.Done(key)
		}
	}, time.Second, ctx.Done())

	// Run until interrupted
	select {}
}
```

### 9.3 Example: Deployment Controller Reconciliation

When a Deployment is updated (e.g., image change):

1. Deployment controller sees the updated Deployment
2. It creates a new ReplicaSet with the new pod template
3. It scales up the new ReplicaSet and scales down the old one
4. ReplicaSet controller creates/deletes Pods as needed
5. Scheduler assigns new Pods to nodes
6. Kubelet pulls images and starts containers

Each step is a separate controller acting on its own resources, yet they
compose into a coordinated rollout.

### 9.4 Owner References and Garbage Collection

Kubernetes tracks object ownership through `ownerReferences`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-deploy-7d4b8c6f5-abc12
  ownerReferences:
    - apiVersion: apps/v1
      kind: ReplicaSet
      name: my-deploy-7d4b8c6f5
      uid: a1b2c3d4-e5f6-7890-abcd-ef1234567890
      controller: true
      blockOwnerDeletion: true
```

When a parent object is deleted, the **garbage collector** controller deletes
all dependent objects (cascading deletion). You can control this behavior:

```bash
# Foreground deletion (wait for dependents to be deleted first)
kubectl delete deployment my-deploy --cascade=foreground

# Orphan dependents (ReplicaSets and Pods keep running)
kubectl delete deployment my-deploy --cascade=orphan

# Background deletion (default: delete parent immediately, GC cleans up)
kubectl delete deployment my-deploy
```

### 9.5 Leader Election

In HA setups, only one instance of a controller should be active. Kubernetes
uses a Lease object for leader election:

```bash
# Check who holds the scheduler lease
kubectl get lease -n kube-system kube-scheduler -o yaml

# Check controller-manager lease
kubectl get lease -n kube-system kube-controller-manager -o yaml
```

---

## Exercises

### Exercise 1: Trace an API Request

Create a simple pod and use `kubectl -v=8` to trace the full API request.
Identify the HTTP method, URL path, and response code for the create operation.

```bash
# Create this pod with verbose output
cat <<EOF > /tmp/trace-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: trace-test
spec:
  containers:
    - name: nginx
      image: nginx:1.25
EOF
```

<details>
<summary>Show Answer</summary>

```bash
kubectl apply -f /tmp/trace-pod.yaml -v=8

# Look for the line similar to:
# I0115 POST https://192.168.49.2:8443/api/v1/namespaces/default/pods 201 Created

# The key observations:
# - Method: POST (creating a new resource)
# - Path: /api/v1/namespaces/default/pods (core group, v1, pods resource)
# - Response: 201 Created (object was persisted in etcd)
# - Content-Type: application/json (kubectl sends JSON by default)
# - You will also see the response body with the full pod spec including
#   server-assigned fields like uid, resourceVersion, and creationTimestamp

# Clean up
kubectl delete pod trace-test
```

</details>

### Exercise 2: Explore etcd Keys

Using `kubectl get --raw`, explore the API to understand how different resources
are organized. List all API groups and find the GVR for Deployments.

<details>
<summary>Show Answer</summary>

```bash
# List all API groups
kubectl get --raw /apis | python3 -c "
import json, sys
data = json.load(sys.stdin)
for g in data['groups']:
    print(f\"{g['name']:40s} preferred: {g['preferredVersion']['groupVersion']}\")
"

# List resources in the apps/v1 group
kubectl get --raw /apis/apps/v1 | python3 -c "
import json, sys
data = json.load(sys.stdin)
for r in data['resources']:
    if '/' not in r['name']:  # skip subresources
        print(f\"  {r['name']:30s} kind={r['kind']:20s} namespaced={r['namespaced']}\")
"

# The GVR for Deployments is:
# Group:    apps
# Version:  v1
# Resource: deployments
# REST path: /apis/apps/v1/namespaces/{ns}/deployments/{name}

# Verify by getting a specific deployment
kubectl get --raw /apis/apps/v1/namespaces/kube-system/deployments/coredns | python3 -m json.tool | head -20
```

</details>

### Exercise 3: RBAC Configuration

Create a ServiceAccount that can only read pods and their logs in the `default`
namespace. Verify the permissions using `kubectl auth can-i`.

<details>
<summary>Show Answer</summary>

```yaml
# Save as /tmp/rbac-exercise.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: pod-log-reader
  namespace: default
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-log-reader-role
  namespace: default
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["pods/log"]
    verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: pod-log-reader-binding
  namespace: default
subjects:
  - kind: ServiceAccount
    name: pod-log-reader
    namespace: default
roleRef:
  kind: Role
  name: pod-log-reader-role
  apiGroup: rbac.authorization.k8s.io
```

```bash
kubectl apply -f /tmp/rbac-exercise.yaml

# Test permissions
kubectl auth can-i get pods \
  --as=system:serviceaccount:default:pod-log-reader
# yes

kubectl auth can-i get pods/log \
  --as=system:serviceaccount:default:pod-log-reader
# yes

kubectl auth can-i create pods \
  --as=system:serviceaccount:default:pod-log-reader
# no

kubectl auth can-i get deployments \
  --as=system:serviceaccount:default:pod-log-reader
# no

kubectl auth can-i get pods \
  --as=system:serviceaccount:default:pod-log-reader \
  --namespace=kube-system
# no (role is namespaced to default)
```

</details>

### Exercise 4: Scheduler Observation

Create a pod with specific node affinity and resource requests. Observe the
scheduler's decision by examining Events.

<details>
<summary>Show Answer</summary>

```yaml
# Save as /tmp/scheduler-exercise.yaml
apiVersion: v1
kind: Pod
metadata:
  name: scheduler-test
spec:
  affinity:
    nodeAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
        - weight: 100
          preference:
            matchExpressions:
              - key: kubernetes.io/os
                operator: In
                values: ["linux"]
  containers:
    - name: nginx
      image: nginx:1.25
      resources:
        requests:
          cpu: "100m"
          memory: "128Mi"
        limits:
          cpu: "200m"
          memory: "256Mi"
```

```bash
kubectl apply -f /tmp/scheduler-exercise.yaml

# Watch the scheduler event
kubectl describe pod scheduler-test | grep -A 5 "Events:"

# You should see an event like:
# Type    Reason     Age   From               Message
# ----    ------     ---   ----               -------
# Normal  Scheduled  10s   default-scheduler  Successfully assigned default/scheduler-test to minikube
# Normal  Pulling    9s    kubelet            Pulling image "nginx:1.25"

# Check which node was selected
kubectl get pod scheduler-test -o jsonpath='{.spec.nodeName}'

# View scheduler logs for this specific pod
kubectl -n kube-system logs kube-scheduler-minikube --tail=20 | grep scheduler-test

# Clean up
kubectl delete pod scheduler-test
```

</details>

### Exercise 5: Controller Reconciliation

Create a ReplicaSet with 3 replicas, then manually delete one pod. Observe the
ReplicaSet controller recreating it to maintain the desired count.

<details>
<summary>Show Answer</summary>

```yaml
# Save as /tmp/reconcile-exercise.yaml
apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: reconcile-test
spec:
  replicas: 3
  selector:
    matchLabels:
      app: reconcile-demo
  template:
    metadata:
      labels:
        app: reconcile-demo
    spec:
      containers:
        - name: nginx
          image: nginx:1.25
```

```bash
kubectl apply -f /tmp/reconcile-exercise.yaml

# Wait for all pods to be running
kubectl get pods -l app=reconcile-demo -w

# Note the pod names
kubectl get pods -l app=reconcile-demo -o name

# Delete one pod
POD=$(kubectl get pods -l app=reconcile-demo -o jsonpath='{.items[0].metadata.name}')
kubectl delete pod $POD

# Immediately check: the ReplicaSet controller creates a replacement
kubectl get pods -l app=reconcile-demo

# Check ReplicaSet events
kubectl describe rs reconcile-test | grep -A 10 "Events:"

# You should see:
# Normal  SuccessfulCreate  Created pod: reconcile-test-xxxxx (original)
# Normal  SuccessfulCreate  Created pod: reconcile-test-yyyyy (replacement)

# The ownerReferences show the RS owns the pods
kubectl get pod -l app=reconcile-demo -o jsonpath='{.items[0].metadata.ownerReferences[0].kind}'
# ReplicaSet

# Clean up
kubectl delete rs reconcile-test
```

</details>

---

**Previous**: [Overview](./00_Overview.md) | **Next**: [Workload Resources](./02_Workload_Resources.md)

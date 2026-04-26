# 11. Operators

**Previous**: [Custom Resource Definitions](./10_Custom_Resource_Definitions.md) | **Next**: [Admission Controllers](./12_Admission_Controllers.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the Operator pattern and articulate why it exists
2. Scaffold, build, and deploy an operator using Kubebuilder and operator-sdk
3. Implement a reconciliation loop with proper error handling and requeue logic
4. Use finalizers, owner references, and leader election correctly
5. Deploy operators via the Operator Lifecycle Manager (OLM)

---

Running stateless workloads on Kubernetes is straightforward -- Deployments, Services, and Ingresses handle the heavy lifting. But stateful, domain-specific applications (databases, message queues, ML pipelines) require human expertise to install, configure, scale, upgrade, and recover. The Operator pattern encodes that human knowledge into software that runs inside the cluster and continuously drives the system toward the desired state. This lesson covers the full lifecycle of building, deploying, and maintaining Kubernetes operators.

Before the scaffolding, read [**Theory & Principles**](#theory--principles) — why the Operator pattern is just CRD + controller-runtime applied to domain knowledge, the informer + work queue + reconcile loop that is the controller's heartbeat, why finalizers and owner references make ownership explicit, and how leader election prevents two operators from racing each other.

## Table of Contents

- [Theory & Principles](#theory--principles)
- [1. The Operator Pattern](#1-the-operator-pattern)
- [2. Operator Framework and operator-sdk](#2-operator-framework-and-operator-sdk)
- [3. Kubebuilder](#3-kubebuilder)
- [4. Controller-Runtime Library](#4-controller-runtime-library)
- [5. Implementing a Reconciliation Loop](#5-implementing-a-reconciliation-loop)
- [6. Leader Election](#6-leader-election)
- [7. Finalizers](#7-finalizers)
- [8. Owner References](#8-owner-references)
- [9. Operator Lifecycle Manager (OLM)](#9-operator-lifecycle-manager-olm)
- [10. Best Practices and Anti-Patterns](#10-best-practices-and-anti-patterns)
- [Exercises](#exercises)

---

## Theory & Principles

The Operator pattern is the answer to a recurring observation: stateful systems (databases, message brokers, ML pipelines) are operated mostly by humans following runbooks. "Initialize the cluster, then add a replica, then wait for it to catch up, then promote it..." Each step is mechanical but requires domain knowledge about *this particular system*. An **Operator** encodes that runbook as a controller running inside the cluster. When you define a custom resource (`PostgresCluster`, lesson 10) and write a controller that knows how to install, upgrade, and recover Postgres, you have made `kubectl apply -f cluster.yaml` produce a managed database. This section explains the controller-runtime architecture that almost every operator builds on, the work-queue pattern that makes reconciliation efficient, finalizers and owner references for cleanup and ownership, and leader election for HA operator deployments.

### A. Operator = Custom Resource + Domain-Aware Controller

An Operator is the composition of two things you already know:

1. **A CRD** (lesson 10) defines the *shape* of the user's intent — a `PostgresCluster` has `spec.replicas`, `spec.version`, `spec.storage`, etc.
2. **A controller** watches that CRD and reconciles the world to match — actually creating StatefulSets, Services, Secrets, PVCs, configuring streaming replication, monitoring health, performing rolling upgrades.

The CRD by itself is just a typed shape stored in etcd. Without the controller, applying a `PostgresCluster` object does nothing. With the controller, the same apply produces a fully working cluster — because the controller has all the operational knowledge baked in.

The pattern's elegance is that it follows the same *model* as built-in Kubernetes (Deployment, ReplicaSet, ...) but for domain objects Kubernetes itself doesn't know about. Once you adopt this pattern, your platform vocabulary expands: instead of "create the StatefulSet, then configure replication, then ..." you say `kubectl apply -f my-database.yaml`.

The Operator pattern is *not* magic — it is just lesson 10's CRD plus a controller written in Go using **controller-runtime** (or in any language using the Kubernetes API directly, but Go has the best ecosystem). What makes operators powerful is the *encoding of expertise*, not any new framework feature.

### B. The Controller's Heartbeat: Informer + Work Queue + Reconcile

Every operator (and every built-in controller) runs the same architecture, provided by **controller-runtime**:

```
Watch → Informer (cache) → Event Handler → Work Queue → Reconciler
```

**Informer** is a long-lived watch on a resource type that maintains a local cache. Why a cache? Because the alternative — every reconcile reads from the API server — is unaffordable. An informer does one initial list and then streams deltas (lesson 01 §A); reads are local and fast.

**Event Handler** sees `ADDED`/`MODIFIED`/`DELETED` events from the informer and decides what to do. Typically: extract the object's namespace/name and enqueue a *reconcile request* for it. Note: handlers do not do the work; they enqueue it.

**Work Queue** is the buffer between event production and reconciliation. It deduplicates (if 100 events for the same object arrive, only one reconcile happens), supports rate limiting (exponential backoff on errors), and enforces sequential processing per key (one reconcile for `default/my-cluster` at a time, no races). controller-runtime gives you a sensible default: a rate-limited queue with exponential-backoff-on-error.

**Reconciler** is the function you write. Its signature is:

```go
func (r *PostgresReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error)
```

Inside, you:
1. Get the desired state: read the CR from the cache.
2. Get the actual state: read child resources (StatefulSet, Service) and their status.
3. Compute the diff and act: create/update/delete as needed.
4. Return: `Result{Requeue: true}` to come back immediately, `Result{RequeueAfter: 30s}` for periodic checking, or `error` to trigger backoff retry.

Two properties that come for free with this pattern:

- **Idempotence by default.** The reconciler is called multiple times for the same object — on initial create, on every change, on resync intervals, on retry. Your code must produce the same outcome each time. The pattern is "create-or-update," not "create" — typically using `controllerutil.CreateOrUpdate`.
- **Level-triggered, not edge-triggered.** You react to *current state*, not to events. If your controller crashes during a reconcile, the next start sees the same state and continues — no missed events to recover.

This loop is *the* programming model for Kubernetes extension. Everything else (finalizers, owner references, status updates) is a refinement on top.

### C. Owner References and Finalizers: Making Ownership Explicit

A `PostgresCluster` resource owns child objects: a StatefulSet, several Services, Secrets for credentials, PVCs for storage. Two mechanisms tie them together:

**Owner References** are metadata pointing from a child to its parent:

```yaml
metadata:
  name: my-cluster-sts
  ownerReferences:
    - apiVersion: example.com/v1
      kind: PostgresCluster
      name: my-cluster
      uid: 12345...
      controller: true
      blockOwnerDeletion: true
```

When you delete the `PostgresCluster`, the **garbage collector** controller (built into kube-controller-manager) sees the dangling owner reference and cascade-deletes the children. You don't write deletion logic for your StatefulSet — the GC handles it because you set `ownerReferences` on creation. This is how Deployments delete their ReplicaSets and ReplicaSets delete their Pods, all "for free."

**Finalizers** are the inverse: a list of strings under `metadata.finalizers` that block deletion until removed. When a user runs `kubectl delete postgrescluster my-cluster`:

1. Kubernetes sets `metadata.deletionTimestamp` (a soft-delete marker).
2. Garbage collection notices but waits because the finalizers list is non-empty.
3. Your controller's reconciler sees `deletionTimestamp != nil` and runs cleanup (e.g., take a final backup, deregister from monitoring, release the cloud-managed disk).
4. After cleanup, your controller removes its finalizer from the list.
5. With finalizers empty, GC actually deletes the object.

Finalizers are the only correct way to do "synchronous cleanup before delete" — without them, the object is gone before you have a chance to react.

A common pattern: register your finalizer on first reconcile (via update), and check `deletionTimestamp` at the top of every reconcile to branch into delete-handling.

### D. Leader Election: One Active, Many Standby

Your operator should run with multiple replicas for HA, but only **one** should reconcile at a time — otherwise two replicas race to create the same StatefulSet. controller-runtime's solution is **leader election**: replicas compete for a `Lease` object in the cluster; whoever holds the Lease is the leader and runs the reconcile loop. Standbys watch the Lease and take over if it expires (default 15s TTL, 10s renew, 2s retry).

```go
mgr, _ := manager.New(cfg, manager.Options{
    LeaderElection:   true,
    LeaderElectionID: "my-operator-lock",
    LeaderElectionNamespace: "my-operator-system",
})
```

The Lease object lives in etcd, so the same consensus that makes etcd safe (lesson 01 §B) makes leader election safe — under network partition, only the side that can reach a quorum of etcd members can hold the lease.

This is the same pattern kube-controller-manager uses for itself. The Deployment controller is highly available: three controller-manager replicas, one elected leader running the reconcile loop, two warm standbys.

### From Theory to the Code Below

The lesson now applies these abstractions:

- **Section 1 (The Operator Pattern)** is §A — why the pattern exists and the canonical examples.
- **Section 2 (Operator Framework, operator-sdk)** is the higher-level scaffolding around controller-runtime.
- **Section 3 (Kubebuilder)** is the standard project layout and code generation for Go-based operators.
- **Section 4 (Controller-Runtime Library)** is §B in Go — Manager, Reconciler, Builder, Client.
- **Section 5 (Implementing a Reconciliation Loop)** is the §B reconciler pattern with proper Result return values, error handling, and requeue strategies.
- **Section 6 (Leader Election)** is §D in code.
- **Section 7 (Finalizers)** is §C's finalizer flow with the `deletionTimestamp` branch.
- **Section 8 (Owner References)** is §C's parent-child wiring with `SetControllerReference`.
- **Section 9 (OLM)** is the lifecycle layer for distributing operators through catalogs.
- **Section 10 (Best Practices and Anti-Patterns)** are operational lessons learned from production operators.

Once you see the operator as "CRD + controller running the watch/queue/reconcile loop," the framework choice (Kubebuilder vs operator-sdk vs Java Operator SDK) becomes a syntax detail. The hard part is the domain knowledge, not the boilerplate.

---

## 1. The Operator Pattern

### 1.1 What is an Operator?

An Operator is a Kubernetes controller that watches one or more custom resources (CRs) and performs application-specific automation. It extends the Kubernetes control loop -- the same pattern that built-in controllers (Deployment, ReplicaSet) use -- to manage complex, stateful workloads.

```
                  ┌──────────────────────────────────────────────┐
                  │            Kubernetes API Server              │
                  └────────────┬────────────────┬────────────────┘
                               │                │
                       Watch CRs/Events   Update Status
                               │                │
                  ┌────────────▼────────────────▼────────────────┐
                  │              Operator Controller              │
                  │  ┌─────────────────────────────────────────┐  │
                  │  │         Reconciliation Loop              │  │
                  │  │  1. Observe current state                │  │
                  │  │  2. Compare with desired state (spec)    │  │
                  │  │  3. Act to converge                      │  │
                  │  │  4. Report status                        │  │
                  │  └─────────────────────────────────────────┘  │
                  └──────────────────────────────────────────────┘
                               │
                  ┌────────────▼────────────────┐
                  │   Managed Resources          │
                  │   (Pods, Services, PVCs...)   │
                  └─────────────────────────────┘
```

### 1.2 Motivation: Why Not Just Use Helm?

Helm deploys resources and walks away. An Operator stays running and continuously manages the application.

| Capability | Helm Chart | Operator |
|---|---|---|
| Initial deployment | Yes | Yes |
| Day-2 operations (backup, failover) | No | Yes |
| Self-healing beyond pod restart | No | Yes |
| Schema migration on upgrade | Manual | Automated |
| Application-aware scaling | No | Yes |
| Lifecycle management | Limited | Full |

### 1.3 Maturity Model

The Operator SDK defines five capability levels:

| Level | Capability | Example |
|---|---|---|
| 1 | Basic Install | Automated provisioning |
| 2 | Seamless Upgrades | Patch and minor version upgrades |
| 3 | Full Lifecycle | Backup, restore, failure recovery |
| 4 | Deep Insights | Metrics, alerts, log processing |
| 5 | Auto Pilot | Auto-scaling, auto-tuning, anomaly detection |

---

## 2. Operator Framework and operator-sdk

### 2.1 Architecture Overview

The Operator Framework consists of three components:

1. **Operator SDK** -- scaffolding and build tooling
2. **Operator Lifecycle Manager (OLM)** -- installation and upgrade management
3. **OperatorHub** -- discovery and sharing

### 2.2 Installation

```bash
# Install operator-sdk CLI
# macOS
brew install operator-sdk

# Linux (amd64)
export ARCH=$(case $(uname -m) in x86_64) echo -n amd64 ;; aarch64) echo -n arm64 ;; esac)
export OS=$(uname | awk '{print tolower($0)}')
export OPERATOR_SDK_DL_URL=https://github.com/operator-framework/operator-sdk/releases/download/v1.34.1
curl -LO ${OPERATOR_SDK_DL_URL}/operator-sdk_${OS}_${ARCH}
chmod +x operator-sdk_${OS}_${ARCH}
sudo mv operator-sdk_${OS}_${ARCH} /usr/local/bin/operator-sdk

# Verify
operator-sdk version
```

### 2.3 SDK Project Types

The operator-sdk supports three project types:

| Type | Language | Use Case |
|---|---|---|
| Go | Go | Full-featured operators with maximum control |
| Ansible | YAML/Ansible | Operators for teams familiar with Ansible |
| Helm | Go template | Wrapping existing Helm charts as operators |

### 2.4 Scaffolding a Go Operator

```bash
# Create project directory
mkdir memcached-operator && cd memcached-operator

# Initialize the project
operator-sdk init \
  --domain example.com \
  --repo github.com/example/memcached-operator

# Create an API (CRD + controller)
operator-sdk create api \
  --group cache \
  --version v1alpha1 \
  --kind Memcached \
  --resource --controller

# Project structure after scaffolding
# .
# ├── Dockerfile
# ├── Makefile
# ├── PROJECT
# ├── api/
# │   └── v1alpha1/
# │       ├── memcached_types.go    # CRD spec/status types
# │       └── zz_generated.deepcopy.go
# ├── cmd/
# │   └── main.go                   # Manager entrypoint
# ├── config/
# │   ├── crd/                      # Generated CRD manifests
# │   ├── manager/                  # Controller manager deployment
# │   ├── rbac/                     # RBAC for the operator
# │   └── samples/                  # Example CR
# └── internal/
#     └── controller/
#         └── memcached_controller.go  # Reconciliation logic
```

---

## 3. Kubebuilder

### 3.1 Kubebuilder vs operator-sdk

Kubebuilder is the upstream project that operator-sdk builds upon. As of v1.28+, operator-sdk uses Kubebuilder's project layout directly.

| Feature | Kubebuilder | operator-sdk |
|---|---|---|
| Go scaffolding | Yes | Yes (wraps Kubebuilder) |
| Ansible/Helm support | No | Yes |
| OLM integration | No | Yes |
| Scorecard testing | No | Yes |
| Bundle/catalog tooling | No | Yes |

### 3.2 Installing Kubebuilder

```bash
# macOS
brew install kubebuilder

# Linux
curl -L -o kubebuilder "https://go.kubebuilder.io/dl/latest/$(go env GOOS)/$(go env GOARCH)"
chmod +x kubebuilder
sudo mv kubebuilder /usr/local/bin/

# Verify
kubebuilder version
```

### 3.3 Defining the API Types

```go
// api/v1alpha1/memcached_types.go
package v1alpha1

import (
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// MemcachedSpec defines the desired state of Memcached
type MemcachedSpec struct {
    // Size is the number of Memcached pods
    // +kubebuilder:validation:Minimum=1
    // +kubebuilder:validation:Maximum=10
    Size int32 `json:"size"`

    // ContainerPort is the port for the Memcached container
    // +kubebuilder:validation:Minimum=1024
    // +kubebuilder:validation:Maximum=65535
    // +kubebuilder:default:=11211
    ContainerPort int32 `json:"containerPort,omitempty"`

    // Image is the Memcached container image
    // +kubebuilder:default:="memcached:1.6-alpine"
    Image string `json:"image,omitempty"`
}

// MemcachedStatus defines the observed state of Memcached
type MemcachedStatus struct {
    // Conditions store the status conditions of the Memcached instances
    // +operator-sdk:csv:customresourcedefinitions:type=status
    Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`

    // ReadyReplicas is the number of ready pods
    ReadyReplicas int32 `json:"readyReplicas,omitempty"`

    // Nodes are the names of the Memcached pods
    Nodes []string `json:"nodes,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Size",type="integer",JSONPath=".spec.size"
// +kubebuilder:printcolumn:name="Ready",type="integer",JSONPath=".status.readyReplicas"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// Memcached is the Schema for the memcacheds API
type Memcached struct {
    metav1.TypeMeta   `json:",inline"`
    metav1.ObjectMeta `json:"metadata,omitempty"`

    Spec   MemcachedSpec   `json:"spec,omitempty"`
    Status MemcachedStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// MemcachedList contains a list of Memcached
type MemcachedList struct {
    metav1.TypeMeta `json:",inline"`
    metav1.ListMeta `json:"metadata,omitempty"`
    Items           []Memcached `json:"items"`
}

func init() {
    SchemeBuilder.Register(&Memcached{}, &MemcachedList{})
}
```

### 3.4 Kubebuilder Markers Reference

Common markers used in type definitions:

```go
// Validation markers
// +kubebuilder:validation:Minimum=0
// +kubebuilder:validation:Maximum=100
// +kubebuilder:validation:Enum=Active;Standby;Failed
// +kubebuilder:validation:Pattern=`^[a-z]+$`
// +kubebuilder:validation:Required

// Resource markers
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:subresource:scale:specpath=.spec.size,statuspath=.status.readyReplicas

// Print column markers
// +kubebuilder:printcolumn:name="Status",type="string",JSONPath=".status.phase"

// RBAC markers (on controller methods)
// +kubebuilder:rbac:groups=cache.example.com,resources=memcacheds,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=cache.example.com,resources=memcacheds/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
```

---

## 4. Controller-Runtime Library

### 4.1 Core Components

Controller-runtime is the library that powers both Kubebuilder and operator-sdk controllers. Its key components are:

```
┌─────────────────────────────────────────────────────┐
│                    Manager                           │
│  ┌────────────┐  ┌────────────┐  ┌───────────────┐  │
│  │   Cache     │  │   Client   │  │ Leader Election│  │
│  │ (Informers) │  │ (API calls)│  │               │  │
│  └──────┬─────┘  └──────┬─────┘  └───────────────┘  │
│         │               │                            │
│  ┌──────▼───────────────▼──────┐                     │
│  │       Controller            │                     │
│  │  ┌──────────────────────┐   │                     │
│  │  │    Work Queue        │   │                     │
│  │  └──────────┬───────────┘   │                     │
│  │             │               │                     │
│  │  ┌──────────▼───────────┐   │                     │
│  │  │    Reconciler        │   │                     │
│  │  │  (Your logic here)   │   │                     │
│  │  └──────────────────────┘   │                     │
│  └─────────────────────────────┘                     │
└─────────────────────────────────────────────────────┘
```

### 4.2 Manager Setup

```go
// cmd/main.go
package main

import (
    "flag"
    "os"

    "k8s.io/apimachinery/pkg/runtime"
    utilruntime "k8s.io/apimachinery/pkg/util/runtime"
    clientgoscheme "k8s.io/client-go/kubernetes/scheme"
    ctrl "sigs.k8s.io/controller-runtime"
    "sigs.k8s.io/controller-runtime/pkg/healthz"
    "sigs.k8s.io/controller-runtime/pkg/log/zap"
    metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

    cachev1alpha1 "github.com/example/memcached-operator/api/v1alpha1"
    "github.com/example/memcached-operator/internal/controller"
)

var (
    scheme   = runtime.NewScheme()
    setupLog = ctrl.Log.WithName("setup")
)

func init() {
    utilruntime.Must(clientgoscheme.AddToScheme(scheme))
    utilruntime.Must(cachev1alpha1.AddToScheme(scheme))
}

func main() {
    var metricsAddr string
    var probeAddr string
    var enableLeaderElection bool

    flag.StringVar(&metricsAddr, "metrics-bind-address", ":8080", "The address the metric endpoint binds to.")
    flag.StringVar(&probeAddr, "health-probe-bind-address", ":8081", "The address the probe endpoint binds to.")
    flag.BoolVar(&enableLeaderElection, "leader-elect", false, "Enable leader election.")
    flag.Parse()

    ctrl.SetLogger(zap.New(zap.UseDevMode(true)))

    mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
        Scheme: scheme,
        Metrics: metricsserver.Options{
            BindAddress: metricsAddr,
        },
        HealthProbeBindAddress: probeAddr,
        LeaderElection:         enableLeaderElection,
        LeaderElectionID:       "memcached-operator-lock",
    })
    if err != nil {
        setupLog.Error(err, "unable to start manager")
        os.Exit(1)
    }

    if err = (&controller.MemcachedReconciler{
        Client: mgr.GetClient(),
        Scheme: mgr.GetScheme(),
    }).SetupWithManager(mgr); err != nil {
        setupLog.Error(err, "unable to create controller", "controller", "Memcached")
        os.Exit(1)
    }

    if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
        setupLog.Error(err, "unable to set up health check")
        os.Exit(1)
    }
    if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
        setupLog.Error(err, "unable to set up ready check")
        os.Exit(1)
    }

    setupLog.Info("starting manager")
    if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
        setupLog.Error(err, "problem running manager")
        os.Exit(1)
    }
}
```

### 4.3 The Client Interface

Controller-runtime provides a unified client that reads from cache and writes to the API server:

```go
// Reading (from cache by default)
instance := &cachev1alpha1.Memcached{}
err := r.Get(ctx, req.NamespacedName, instance)

// Listing with label selectors
podList := &corev1.PodList{}
listOpts := []client.ListOption{
    client.InNamespace(req.Namespace),
    client.MatchingLabels{"app": "memcached", "memcached_cr": req.Name},
}
err := r.List(ctx, podList, listOpts...)

// Writing (always goes to API server)
err := r.Create(ctx, deployment)
err := r.Update(ctx, instance)
err := r.Status().Update(ctx, instance)  // status subresource
err := r.Delete(ctx, pod)

// Patch for conflict-free updates
patch := client.MergeFrom(instance.DeepCopy())
instance.Status.ReadyReplicas = readyCount
err := r.Status().Patch(ctx, instance, patch)
```

---

## 5. Implementing a Reconciliation Loop

### 5.1 The Reconciler Interface

Every controller must implement the `Reconciler` interface:

```go
type Reconciler interface {
    Reconcile(ctx context.Context, req Request) (Result, error)
}
```

The return values control requeue behavior:

| Return | Behavior |
|---|---|
| `Result{}, nil` | Success, stop reconciling |
| `Result{Requeue: true}, nil` | Requeue immediately |
| `Result{RequeueAfter: 30 * time.Second}, nil` | Requeue after duration |
| `Result{}, err` | Requeue with exponential backoff |

### 5.2 Full Reconciler Implementation

```go
// internal/controller/memcached_controller.go
package controller

import (
    "context"
    "fmt"
    "time"

    appsv1 "k8s.io/api/apps/v1"
    corev1 "k8s.io/api/core/v1"
    apierrors "k8s.io/apimachinery/pkg/api/errors"
    "k8s.io/apimachinery/pkg/api/meta"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/runtime"
    "k8s.io/apimachinery/pkg/types"
    "k8s.io/client-go/tools/record"
    ctrl "sigs.k8s.io/controller-runtime"
    "sigs.k8s.io/controller-runtime/pkg/client"
    "sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
    "sigs.k8s.io/controller-runtime/pkg/log"

    cachev1alpha1 "github.com/example/memcached-operator/api/v1alpha1"
)

const memcachedFinalizer = "cache.example.com/finalizer"

// MemcachedReconciler reconciles a Memcached object
type MemcachedReconciler struct {
    client.Client
    Scheme   *runtime.Scheme
    Recorder record.EventRecorder
}

// +kubebuilder:rbac:groups=cache.example.com,resources=memcacheds,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=cache.example.com,resources=memcacheds/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=cache.example.com,resources=memcacheds/finalizers,verbs=update
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=events,verbs=create;patch

func (r *MemcachedReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    log := log.FromContext(ctx)

    // Step 1: Fetch the Memcached instance
    memcached := &cachev1alpha1.Memcached{}
    err := r.Get(ctx, req.NamespacedName, memcached)
    if err != nil {
        if apierrors.IsNotFound(err) {
            log.Info("Memcached resource not found. Ignoring since object must be deleted.")
            return ctrl.Result{}, nil
        }
        log.Error(err, "Failed to get Memcached")
        return ctrl.Result{}, err
    }

    // Step 2: Handle finalizer for cleanup
    if memcached.ObjectMeta.DeletionTimestamp.IsZero() {
        // Object is not being deleted -- add finalizer if not present
        if !controllerutil.ContainsFinalizer(memcached, memcachedFinalizer) {
            controllerutil.AddFinalizer(memcached, memcachedFinalizer)
            if err := r.Update(ctx, memcached); err != nil {
                return ctrl.Result{}, err
            }
        }
    } else {
        // Object is being deleted -- run cleanup logic
        if controllerutil.ContainsFinalizer(memcached, memcachedFinalizer) {
            if err := r.cleanupResources(ctx, memcached); err != nil {
                return ctrl.Result{}, err
            }
            controllerutil.RemoveFinalizer(memcached, memcachedFinalizer)
            if err := r.Update(ctx, memcached); err != nil {
                return ctrl.Result{}, err
            }
        }
        return ctrl.Result{}, nil
    }

    // Step 3: Check if the Deployment already exists, create if not
    found := &appsv1.Deployment{}
    err = r.Get(ctx, types.NamespacedName{Name: memcached.Name, Namespace: memcached.Namespace}, found)
    if err != nil && apierrors.IsNotFound(err) {
        dep, err := r.deploymentForMemcached(memcached)
        if err != nil {
            log.Error(err, "Failed to define new Deployment for Memcached")
            meta.SetStatusCondition(&memcached.Status.Conditions, metav1.Condition{
                Type:    "Available",
                Status:  metav1.ConditionFalse,
                Reason:  "Reconciling",
                Message: fmt.Sprintf("Failed to create Deployment: %s", err),
            })
            if statusErr := r.Status().Update(ctx, memcached); statusErr != nil {
                log.Error(statusErr, "Failed to update Memcached status")
                return ctrl.Result{}, statusErr
            }
            return ctrl.Result{}, err
        }

        log.Info("Creating a new Deployment", "Deployment.Namespace", dep.Namespace, "Deployment.Name", dep.Name)
        if err = r.Create(ctx, dep); err != nil {
            log.Error(err, "Failed to create new Deployment")
            return ctrl.Result{}, err
        }
        r.Recorder.Event(memcached, corev1.EventTypeNormal, "Created", "Deployment created successfully")
        return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
    } else if err != nil {
        log.Error(err, "Failed to get Deployment")
        return ctrl.Result{}, err
    }

    // Step 4: Ensure the Deployment size matches the spec
    size := memcached.Spec.Size
    if *found.Spec.Replicas != size {
        found.Spec.Replicas = &size
        if err = r.Update(ctx, found); err != nil {
            log.Error(err, "Failed to update Deployment", "Deployment.Namespace", found.Namespace, "Deployment.Name", found.Name)
            return ctrl.Result{}, err
        }
        r.Recorder.Eventf(memcached, corev1.EventTypeNormal, "Scaled", "Scaled deployment to %d replicas", size)
        return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
    }

    // Step 5: Update status with ready replicas
    memcached.Status.ReadyReplicas = found.Status.ReadyReplicas
    podList := &corev1.PodList{}
    listOpts := []client.ListOption{
        client.InNamespace(memcached.Namespace),
        client.MatchingLabels(labelsForMemcached(memcached.Name)),
    }
    if err = r.List(ctx, podList, listOpts...); err != nil {
        log.Error(err, "Failed to list pods")
        return ctrl.Result{}, err
    }
    podNames := getPodNames(podList.Items)
    memcached.Status.Nodes = podNames

    meta.SetStatusCondition(&memcached.Status.Conditions, metav1.Condition{
        Type:    "Available",
        Status:  metav1.ConditionTrue,
        Reason:  "Reconciling",
        Message: fmt.Sprintf("Deployment has %d ready replicas", found.Status.ReadyReplicas),
    })

    if err := r.Status().Update(ctx, memcached); err != nil {
        log.Error(err, "Failed to update Memcached status")
        return ctrl.Result{}, err
    }

    return ctrl.Result{RequeueAfter: 1 * time.Minute}, nil
}

// deploymentForMemcached returns a Deployment object for the Memcached CR
func (r *MemcachedReconciler) deploymentForMemcached(m *cachev1alpha1.Memcached) (*appsv1.Deployment, error) {
    labels := labelsForMemcached(m.Name)
    replicas := m.Spec.Size

    dep := &appsv1.Deployment{
        ObjectMeta: metav1.ObjectMeta{
            Name:      m.Name,
            Namespace: m.Namespace,
            Labels:    labels,
        },
        Spec: appsv1.DeploymentSpec{
            Replicas: &replicas,
            Selector: &metav1.LabelSelector{
                MatchLabels: labels,
            },
            Template: corev1.PodTemplateSpec{
                ObjectMeta: metav1.ObjectMeta{
                    Labels: labels,
                },
                Spec: corev1.PodSpec{
                    Containers: []corev1.Container{{
                        Name:    "memcached",
                        Image:   m.Spec.Image,
                        Command: []string{"memcached", "-m=64", "-o", "modern", "-v"},
                        Ports: []corev1.ContainerPort{{
                            ContainerPort: m.Spec.ContainerPort,
                            Name:          "memcached",
                        }},
                    }},
                },
            },
        },
    }

    // Set the owning CR as the owner of the Deployment
    if err := ctrl.SetControllerReference(m, dep, r.Scheme); err != nil {
        return nil, err
    }
    return dep, nil
}

func labelsForMemcached(name string) map[string]string {
    return map[string]string{
        "app.kubernetes.io/name":       "memcached",
        "app.kubernetes.io/instance":   name,
        "app.kubernetes.io/managed-by": "memcached-operator",
    }
}

func getPodNames(pods []corev1.Pod) []string {
    podNames := make([]string, len(pods))
    for i, pod := range pods {
        podNames[i] = pod.Name
    }
    return podNames
}

func (r *MemcachedReconciler) cleanupResources(ctx context.Context, m *cachev1alpha1.Memcached) error {
    log := log.FromContext(ctx)
    log.Info("Running cleanup for Memcached", "name", m.Name)
    // External cleanup logic goes here (e.g., deregister from service mesh)
    return nil
}

// SetupWithManager sets up the controller with the Manager
func (r *MemcachedReconciler) SetupWithManager(mgr ctrl.Manager) error {
    return ctrl.NewControllerManagedBy(mgr).
        For(&cachev1alpha1.Memcached{}).
        Owns(&appsv1.Deployment{}).
        Complete(r)
}
```

### 5.3 Reconciliation Patterns

```
                Idempotent Reconciliation Flow
                ================================

    ┌──────────┐    ┌──────────────┐    ┌──────────────┐
    │  Observe  │───▶│   Compare    │───▶│     Act      │
    │  current  │    │  current vs  │    │  create/     │
    │  state    │    │  desired     │    │  update/     │
    └──────────┘    └──────────────┘    │  delete      │
         ▲                               └──────┬───────┘
         │                                      │
         └──────────────────────────────────────┘
                    (requeue)
```

Key principles:

1. **Idempotency** -- running Reconcile multiple times with the same input produces the same output
2. **Level-triggered** -- react to current state, not events
3. **Edge-agnostic** -- do not assume reconciliation is triggered by a specific event

---

## 6. Leader Election

### 6.1 Why Leader Election?

When running multiple operator replicas for high availability, only one instance should actively reconcile at a time. Leader election uses a Kubernetes Lease object to ensure single-writer semantics.

### 6.2 How It Works

```
   Pod-A (Leader)           Pod-B (Standby)          Pod-C (Standby)
   ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
   │ Reconciling  │         │   Watching   │         │   Watching   │
   │ actively     │         │   lease      │         │   lease      │
   └──────┬───────┘         └──────┬───────┘         └──────┬───────┘
          │                        │                        │
          ▼                        ▼                        ▼
   ┌─────────────────────────────────────────────────────────┐
   │              Lease: memcached-operator-lock              │
   │              holder: pod-a                               │
   │              renewTime: 2024-01-15T10:00:00Z             │
   └─────────────────────────────────────────────────────────┘
```

### 6.3 Configuration

```go
// Leader election is configured in the Manager options
mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
    LeaderElection:          true,
    LeaderElectionID:        "memcached-operator-lock",
    LeaderElectionNamespace: "memcached-operator-system",
    // Tune timings for faster failover (defaults shown)
    LeaseDuration: durationPtr(15 * time.Second),
    RenewDeadline: durationPtr(10 * time.Second),
    RetryPeriod:   durationPtr(2 * time.Second),
})

func durationPtr(d time.Duration) *time.Duration {
    return &d
}
```

The operator deployment should specify `replicas: 2` for high availability:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: memcached-operator-controller-manager
  namespace: memcached-operator-system
spec:
  replicas: 2
  selector:
    matchLabels:
      control-plane: controller-manager
  template:
    spec:
      containers:
      - name: manager
        image: controller:latest
        args:
        - --leader-elect
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8081
          initialDelaySeconds: 15
          periodSeconds: 20
        readinessProbe:
          httpGet:
            path: /readyz
            port: 8081
          initialDelaySeconds: 5
          periodSeconds: 10
```

---

## 7. Finalizers

### 7.1 What Are Finalizers?

Finalizers are keys on a resource that signal pre-deletion hooks. When a resource with a finalizer is deleted, Kubernetes sets `deletionTimestamp` but does not remove the object until all finalizers are removed.

### 7.2 Finalizer Lifecycle

```
User: kubectl delete memcached my-cache
    │
    ▼
API Server sets deletionTimestamp
    │
    ▼
Operator sees deletionTimestamp is non-zero
    │
    ▼
Operator runs cleanup logic
    (e.g., delete external resources, revoke credentials)
    │
    ▼
Operator removes finalizer from object
    │
    ▼
API Server garbage-collects the object
```

### 7.3 Implementation Pattern

```go
const myFinalizer = "cache.example.com/finalizer"

func (r *MemcachedReconciler) handleFinalizer(ctx context.Context, m *cachev1alpha1.Memcached) (ctrl.Result, error) {
    if m.ObjectMeta.DeletionTimestamp.IsZero() {
        // Not being deleted -- ensure finalizer is present
        if !controllerutil.ContainsFinalizer(m, myFinalizer) {
            controllerutil.AddFinalizer(m, myFinalizer)
            if err := r.Update(ctx, m); err != nil {
                return ctrl.Result{}, err
            }
        }
        return ctrl.Result{}, nil
    }

    // Being deleted -- run cleanup
    if controllerutil.ContainsFinalizer(m, myFinalizer) {
        // Cleanup: delete PVCs that are not garbage-collected
        if err := r.deleteOrphanedPVCs(ctx, m); err != nil {
            return ctrl.Result{}, err
        }

        // Cleanup: remove external DNS record
        if err := r.removeExternalDNS(ctx, m); err != nil {
            return ctrl.Result{}, err
        }

        // Remove the finalizer to allow garbage collection
        controllerutil.RemoveFinalizer(m, myFinalizer)
        if err := r.Update(ctx, m); err != nil {
            return ctrl.Result{}, err
        }
    }

    return ctrl.Result{}, nil
}
```

### 7.4 Common Finalizer Pitfalls

| Pitfall | Consequence | Solution |
|---|---|---|
| Finalizer cleanup hangs forever | Object stuck in Terminating state | Add timeouts and fallback logic |
| Finalizer added but cleanup logic removed | Object permanently undeletable | Always test deletion path |
| Not checking `IsNotFound` during cleanup | Cleanup fails on already-deleted resources | Ignore NotFound errors |
| Mutating spec during deletion | Validation webhook rejects the update | Only modify metadata/status during deletion |

---

## 8. Owner References

### 8.1 How Owner References Work

Owner references create a parent-child relationship between Kubernetes objects. When the parent is deleted, the garbage collector automatically deletes all children.

```go
// Setting owner reference with controller-runtime
func (r *MemcachedReconciler) deploymentForMemcached(m *cachev1alpha1.Memcached) (*appsv1.Deployment, error) {
    dep := &appsv1.Deployment{
        // ... deployment spec ...
    }
    // This sets the Memcached CR as the owner of the Deployment
    if err := ctrl.SetControllerReference(m, dep, r.Scheme); err != nil {
        return nil, err
    }
    return dep, nil
}
```

### 8.2 Owner Reference Fields

```yaml
# The child resource (Deployment) will have:
metadata:
  ownerReferences:
  - apiVersion: cache.example.com/v1alpha1
    kind: Memcached
    name: my-cache
    uid: d9607e19-f88f-11e6-a518-42010a800195
    controller: true        # This owner is the managing controller
    blockOwnerDeletion: true # Block deletion until child is cleaned up
```

### 8.3 Cross-Namespace Ownership

Owner references cannot cross namespace boundaries. For cross-namespace relationships, use labels and finalizers instead:

```go
// Label the resource with the owner's identity
labels := map[string]string{
    "managed-by":      "memcached-operator",
    "owner-name":      m.Name,
    "owner-namespace": m.Namespace,
}
```

### 8.4 The Owns() Watch

When you use `Owns()` in the controller setup, controller-runtime automatically watches child resources and maps events back to the parent:

```go
func (r *MemcachedReconciler) SetupWithManager(mgr ctrl.Manager) error {
    return ctrl.NewControllerManagedBy(mgr).
        For(&cachev1alpha1.Memcached{}).          // Watch the primary resource
        Owns(&appsv1.Deployment{}).                // Watch owned Deployments
        Owns(&corev1.Service{}).                   // Watch owned Services
        Owns(&corev1.ConfigMap{}).                 // Watch owned ConfigMaps
        WithOptions(controller.Options{
            MaxConcurrentReconciles: 2,
        }).
        Complete(r)
}
```

---

## 9. Operator Lifecycle Manager (OLM)

### 9.1 What is OLM?

OLM manages the lifecycle of operators themselves -- installation, upgrades, RBAC, and dependency resolution. It treats operators as first-class citizens with versioned packages.

### 9.2 OLM Architecture

```
┌──────────────────────────────────────────────────┐
│                  OLM Components                   │
│                                                   │
│  ┌──────────────┐    ┌──────────────────────┐    │
│  │   Catalog    │    │   OLM Operator       │    │
│  │   Operator   │    │   (installs CSVs)    │    │
│  │ (indexes)    │    │                      │    │
│  └──────┬───────┘    └──────────┬───────────┘    │
│         │                       │                │
│         ▼                       ▼                │
│  ┌──────────────┐    ┌──────────────────────┐    │
│  │ CatalogSource│    │ ClusterServiceVersion│    │
│  │ (package     │───▶│ (CSV - describes     │    │
│  │  index)      │    │  an operator version)│    │
│  └──────────────┘    └──────────────────────┘    │
│                              │                   │
│                              ▼                   │
│                      ┌───────────────┐           │
│                      │ Subscription  │           │
│                      │ (auto-update  │           │
│                      │  channel)     │           │
│                      └───────────────┘           │
└──────────────────────────────────────────────────┘
```

### 9.3 Building an OLM Bundle

```bash
# Generate the ClusterServiceVersion (CSV)
operator-sdk generate kustomize manifests

# Build the bundle (CRDs + CSV + metadata)
make bundle IMG=example.com/memcached-operator:v0.1.0

# Bundle directory structure
# bundle/
# ├── manifests/
# │   ├── cache.example.com_memcacheds.yaml    # CRD
# │   └── memcached-operator.clusterserviceversion.yaml  # CSV
# ├── metadata/
# │   └── annotations.yaml
# └── tests/
#     └── scorecard/
#         └── config.yaml

# Build and push the bundle image
make bundle-build bundle-push BUNDLE_IMG=example.com/memcached-operator-bundle:v0.1.0

# Build a catalog containing the bundle
make catalog-build catalog-push CATALOG_IMG=example.com/memcached-operator-catalog:v0.1.0
```

### 9.4 Installing an Operator via OLM

```bash
# Install OLM itself (if not pre-installed)
operator-sdk olm install

# Create a CatalogSource
kubectl apply -f - <<EOF
apiVersion: operators.coreos.com/v1alpha1
kind: CatalogSource
metadata:
  name: memcached-operator-catalog
  namespace: olm
spec:
  sourceType: grpc
  image: example.com/memcached-operator-catalog:v0.1.0
  displayName: Memcached Operator Catalog
  updateStrategy:
    registryPoll:
      interval: 10m
EOF

# Create a Subscription to install the operator
kubectl apply -f - <<EOF
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: memcached-operator
  namespace: operators
spec:
  channel: alpha
  name: memcached-operator
  source: memcached-operator-catalog
  sourceNamespace: olm
  installPlanApproval: Automatic
EOF
```

### 9.5 Upgrade Channels

```yaml
# CSV defines the upgrade path
apiVersion: operators.coreos.com/v1alpha1
kind: ClusterServiceVersion
metadata:
  name: memcached-operator.v0.2.0
spec:
  replaces: memcached-operator.v0.1.0  # Upgrade from v0.1.0
  version: 0.2.0
  # skips can be used to skip intermediate versions
  # skips:
  # - memcached-operator.v0.1.1
```

---

## 10. Best Practices and Anti-Patterns

### 10.1 Best Practices

**Idempotent reconciliation**: Every call to Reconcile should produce the same result given the same cluster state.

```go
// GOOD: Create-or-update pattern
func (r *Reconciler) reconcileDeployment(ctx context.Context, m *cachev1alpha1.Memcached) error {
    dep := &appsv1.Deployment{
        ObjectMeta: metav1.ObjectMeta{
            Name:      m.Name,
            Namespace: m.Namespace,
        },
    }
    op, err := controllerutil.CreateOrUpdate(ctx, r.Client, dep, func() error {
        // Mutate the deployment to match desired state
        dep.Spec.Replicas = &m.Spec.Size
        dep.Spec.Template.Spec.Containers[0].Image = m.Spec.Image
        return ctrl.SetControllerReference(m, dep, r.Scheme)
    })
    if err != nil {
        return err
    }
    log.FromContext(ctx).Info("Deployment reconciled", "operation", op)
    return nil
}
```

**Use status conditions**: Report meaningful status using standard condition types.

```go
meta.SetStatusCondition(&m.Status.Conditions, metav1.Condition{
    Type:               "Degraded",
    Status:             metav1.ConditionTrue,
    Reason:             "InsufficientReplicas",
    Message:            "Only 1 of 3 desired replicas are ready",
    ObservedGeneration: m.Generation,
})
```

**Emit events**: Use the event recorder for user-visible operations.

```go
r.Recorder.Event(m, corev1.EventTypeNormal, "Upgraded", "Memcached version upgraded to 1.6.18")
r.Recorder.Event(m, corev1.EventTypeWarning, "BackupFailed", "Scheduled backup failed: connection timeout")
```

### 10.2 Anti-Patterns

| Anti-Pattern | Problem | Solution |
|---|---|---|
| Reconcile does I/O in a goroutine | Untracked work, resource leaks | Keep all work inside Reconcile |
| Using `watch.Interface` directly | Bypasses the cache, creates API load | Use informers via controller setup |
| Storing state in controller struct | Lost on restart, breaks leader election | Store state in CR status or ConfigMap |
| No rate limiting on external calls | Thundering herd on operator restart | Use workqueue rate limiter |
| Ignoring `Generation` vs `ResourceVersion` | Reconcile on every status update | Filter with `GenerationChangedPredicate` |
| Not setting resource limits on operator pod | Operator OOMKilled under load | Set requests and limits |

### 10.3 Filtering Events with Predicates

```go
import "sigs.k8s.io/controller-runtime/pkg/predicate"

func (r *MemcachedReconciler) SetupWithManager(mgr ctrl.Manager) error {
    return ctrl.NewControllerManagedBy(mgr).
        For(&cachev1alpha1.Memcached{}, builder.WithPredicates(
            predicate.GenerationChangedPredicate{}, // Only spec changes
        )).
        Owns(&appsv1.Deployment{}).
        Complete(r)
}
```

### 10.4 Testing Operators

```go
// Using envtest for integration testing
package controller

import (
    "context"
    "time"

    . "github.com/onsi/ginkgo/v2"
    . "github.com/onsi/gomega"
    appsv1 "k8s.io/api/apps/v1"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/types"

    cachev1alpha1 "github.com/example/memcached-operator/api/v1alpha1"
)

var _ = Describe("Memcached controller", func() {
    const (
        timeout  = time.Second * 30
        interval = time.Second * 1
    )

    Context("When creating a Memcached CR", func() {
        It("Should create a Deployment with correct replica count", func() {
            ctx := context.Background()
            memcached := &cachev1alpha1.Memcached{
                ObjectMeta: metav1.ObjectMeta{
                    Name:      "test-memcached",
                    Namespace: "default",
                },
                Spec: cachev1alpha1.MemcachedSpec{
                    Size:  3,
                    Image: "memcached:1.6-alpine",
                },
            }
            Expect(k8sClient.Create(ctx, memcached)).Should(Succeed())

            deploymentKey := types.NamespacedName{Name: "test-memcached", Namespace: "default"}
            createdDeployment := &appsv1.Deployment{}

            Eventually(func() bool {
                err := k8sClient.Get(ctx, deploymentKey, createdDeployment)
                return err == nil
            }, timeout, interval).Should(BeTrue())

            Expect(*createdDeployment.Spec.Replicas).Should(Equal(int32(3)))
        })
    })
})
```

---

## Exercises

### Exercise 1: Scaffold an Operator

Using operator-sdk or kubebuilder, scaffold a new operator project for managing a custom `Redis` resource. The CRD should have the following spec fields: `replicas` (int32), `version` (string), and `persistenceEnabled` (bool). Write out the complete `redis_types.go` file with proper validation markers.

<details>
<summary>Show Answer</summary>

```bash
# Scaffold the project
mkdir redis-operator && cd redis-operator
operator-sdk init --domain example.com --repo github.com/example/redis-operator
operator-sdk create api --group database --version v1alpha1 --kind Redis --resource --controller
```

```go
// api/v1alpha1/redis_types.go
package v1alpha1

import (
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type RedisSpec struct {
    // +kubebuilder:validation:Minimum=1
    // +kubebuilder:validation:Maximum=6
    // +kubebuilder:default:=1
    Replicas int32 `json:"replicas"`

    // +kubebuilder:validation:Pattern=`^[0-9]+\.[0-9]+\.[0-9]+$`
    // +kubebuilder:default:="7.2.4"
    Version string `json:"version,omitempty"`

    // +kubebuilder:default:=false
    PersistenceEnabled bool `json:"persistenceEnabled,omitempty"`
}

type RedisStatus struct {
    Conditions    []metav1.Condition `json:"conditions,omitempty"`
    ReadyReplicas int32             `json:"readyReplicas,omitempty"`
    Phase         string            `json:"phase,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Replicas",type="integer",JSONPath=".spec.replicas"
// +kubebuilder:printcolumn:name="Version",type="string",JSONPath=".spec.version"
// +kubebuilder:printcolumn:name="Ready",type="integer",JSONPath=".status.readyReplicas"
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
type Redis struct {
    metav1.TypeMeta   `json:",inline"`
    metav1.ObjectMeta `json:"metadata,omitempty"`
    Spec              RedisSpec   `json:"spec,omitempty"`
    Status            RedisStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true
type RedisList struct {
    metav1.TypeMeta `json:",inline"`
    metav1.ListMeta `json:"metadata,omitempty"`
    Items           []Redis `json:"items"`
}

func init() {
    SchemeBuilder.Register(&Redis{}, &RedisList{})
}
```

</details>

### Exercise 2: Implement a Reconciliation Loop

Write a complete Reconcile function for the Redis operator from Exercise 1. The controller should: (a) create a StatefulSet with the specified number of replicas, (b) create a headless Service for the StatefulSet, (c) update the status with ready replica count and a Phase field (Pending/Running/Failed).

<details>
<summary>Show Answer</summary>

```go
func (r *RedisReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    log := log.FromContext(ctx)

    redis := &databasev1alpha1.Redis{}
    if err := r.Get(ctx, req.NamespacedName, redis); err != nil {
        if apierrors.IsNotFound(err) {
            return ctrl.Result{}, nil
        }
        return ctrl.Result{}, err
    }

    // Reconcile the headless Service
    svc := &corev1.Service{
        ObjectMeta: metav1.ObjectMeta{
            Name:      redis.Name + "-headless",
            Namespace: redis.Namespace,
        },
    }
    _, err := controllerutil.CreateOrUpdate(ctx, r.Client, svc, func() error {
        svc.Spec.ClusterIP = "None"
        svc.Spec.Selector = map[string]string{"app": redis.Name}
        svc.Spec.Ports = []corev1.ServicePort{{
            Port:     6379,
            Name:     "redis",
            Protocol: corev1.ProtocolTCP,
        }}
        return ctrl.SetControllerReference(redis, svc, r.Scheme)
    })
    if err != nil {
        return ctrl.Result{}, err
    }

    // Reconcile the StatefulSet
    sts := &appsv1.StatefulSet{
        ObjectMeta: metav1.ObjectMeta{
            Name:      redis.Name,
            Namespace: redis.Namespace,
        },
    }
    _, err = controllerutil.CreateOrUpdate(ctx, r.Client, sts, func() error {
        replicas := redis.Spec.Replicas
        labels := map[string]string{"app": redis.Name}
        sts.Spec.Replicas = &replicas
        sts.Spec.ServiceName = redis.Name + "-headless"
        sts.Spec.Selector = &metav1.LabelSelector{MatchLabels: labels}
        sts.Spec.Template = corev1.PodTemplateSpec{
            ObjectMeta: metav1.ObjectMeta{Labels: labels},
            Spec: corev1.PodSpec{
                Containers: []corev1.Container{{
                    Name:  "redis",
                    Image: "redis:" + redis.Spec.Version,
                    Ports: []corev1.ContainerPort{{ContainerPort: 6379}},
                }},
            },
        }
        return ctrl.SetControllerReference(redis, sts, r.Scheme)
    })
    if err != nil {
        log.Error(err, "Failed to reconcile StatefulSet")
        return ctrl.Result{}, err
    }

    // Update status
    existingSts := &appsv1.StatefulSet{}
    if err := r.Get(ctx, types.NamespacedName{Name: redis.Name, Namespace: redis.Namespace}, existingSts); err != nil {
        return ctrl.Result{}, err
    }
    redis.Status.ReadyReplicas = existingSts.Status.ReadyReplicas
    if existingSts.Status.ReadyReplicas == redis.Spec.Replicas {
        redis.Status.Phase = "Running"
    } else if existingSts.Status.ReadyReplicas > 0 {
        redis.Status.Phase = "Pending"
    } else {
        redis.Status.Phase = "Pending"
    }
    if err := r.Status().Update(ctx, redis); err != nil {
        return ctrl.Result{}, err
    }

    return ctrl.Result{RequeueAfter: 30 * time.Second}, nil
}
```

</details>

### Exercise 3: Implement a Finalizer

Add a finalizer to the Redis operator that performs the following cleanup when the CR is deleted: (a) takes a final backup by creating a Job that runs `redis-cli BGSAVE`, (b) waits for the Job to complete, (c) removes the finalizer. Handle the case where the backup Job fails.

<details>
<summary>Show Answer</summary>

```go
const redisFinalizer = "database.example.com/backup-finalizer"

func (r *RedisReconciler) handleDeletion(ctx context.Context, redis *databasev1alpha1.Redis) (ctrl.Result, error) {
    log := log.FromContext(ctx)

    if !redis.ObjectMeta.DeletionTimestamp.IsZero() {
        if controllerutil.ContainsFinalizer(redis, redisFinalizer) {
            // Check if backup Job already exists
            backupJob := &batchv1.Job{}
            jobName := redis.Name + "-final-backup"
            err := r.Get(ctx, types.NamespacedName{Name: jobName, Namespace: redis.Namespace}, backupJob)

            if apierrors.IsNotFound(err) {
                // Create the backup Job
                job := &batchv1.Job{
                    ObjectMeta: metav1.ObjectMeta{
                        Name:      jobName,
                        Namespace: redis.Namespace,
                    },
                    Spec: batchv1.JobSpec{
                        Template: corev1.PodTemplateSpec{
                            Spec: corev1.PodSpec{
                                RestartPolicy: corev1.RestartPolicyNever,
                                Containers: []corev1.Container{{
                                    Name:    "backup",
                                    Image:   "redis:7.2",
                                    Command: []string{"redis-cli", "-h", redis.Name + "-headless", "BGSAVE"},
                                }},
                            },
                        },
                        BackoffLimit: int32Ptr(3),
                    },
                }
                if err := r.Create(ctx, job); err != nil {
                    log.Error(err, "Failed to create backup job")
                    return ctrl.Result{}, err
                }
                log.Info("Created final backup job", "job", jobName)
                return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
            } else if err != nil {
                return ctrl.Result{}, err
            }

            // Check Job status
            if backupJob.Status.Succeeded > 0 {
                log.Info("Backup completed successfully")
            } else if backupJob.Status.Failed >= 3 {
                log.Info("Backup failed after 3 attempts, proceeding with deletion")
                r.Recorder.Event(redis, corev1.EventTypeWarning, "BackupFailed",
                    "Final backup failed, proceeding with deletion")
            } else {
                // Job still running
                return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
            }

            // Remove the finalizer
            controllerutil.RemoveFinalizer(redis, redisFinalizer)
            if err := r.Update(ctx, redis); err != nil {
                return ctrl.Result{}, err
            }
        }
        return ctrl.Result{}, nil
    }

    // Not being deleted -- ensure finalizer is present
    if !controllerutil.ContainsFinalizer(redis, redisFinalizer) {
        controllerutil.AddFinalizer(redis, redisFinalizer)
        if err := r.Update(ctx, redis); err != nil {
            return ctrl.Result{}, err
        }
    }
    return ctrl.Result{}, nil
}

func int32Ptr(i int32) *int32 { return &i }
```

</details>

### Exercise 4: Leader Election Configuration

You need to deploy an operator with 3 replicas for high availability. Write the Deployment manifest and the manager configuration code that: (a) enables leader election, (b) configures a 10-second lease duration with a 7-second renew deadline, (c) adds health and readiness probes, (d) sets appropriate resource requests and limits.

<details>
<summary>Show Answer</summary>

```go
// Manager setup with leader election tuning
mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
    Scheme:                 scheme,
    LeaderElection:         true,
    LeaderElectionID:       "redis-operator-lock",
    LeaseDuration:          durationPtr(10 * time.Second),
    RenewDeadline:          durationPtr(7 * time.Second),
    RetryPeriod:            durationPtr(2 * time.Second),
    HealthProbeBindAddress: ":8081",
    Metrics: metricsserver.Options{
        BindAddress: ":8080",
    },
})
if err != nil {
    setupLog.Error(err, "unable to start manager")
    os.Exit(1)
}

mgr.AddHealthzCheck("healthz", healthz.Ping)
mgr.AddReadyzCheck("readyz", healthz.Ping)
```

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis-operator-controller-manager
  namespace: redis-operator-system
spec:
  replicas: 3
  selector:
    matchLabels:
      control-plane: controller-manager
  template:
    metadata:
      labels:
        control-plane: controller-manager
    spec:
      serviceAccountName: redis-operator-controller-manager
      terminationGracePeriodSeconds: 10
      containers:
      - name: manager
        image: example.com/redis-operator:v0.1.0
        command:
        - /manager
        args:
        - --leader-elect
        ports:
        - containerPort: 8080
          name: metrics
        - containerPort: 8081
          name: health
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8081
          initialDelaySeconds: 15
          periodSeconds: 20
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /readyz
            port: 8081
          initialDelaySeconds: 5
          periodSeconds: 10
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 256Mi
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
          readOnlyRootFilesystem: true
          runAsNonRoot: true
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchLabels:
                  control-plane: controller-manager
              topologyKey: kubernetes.io/hostname
```

</details>

### Exercise 5: OLM Bundle

Create the ClusterServiceVersion (CSV) manifest for the Memcached operator v0.2.0 that: (a) replaces v0.1.0, (b) declares a required API dependency on `certificates.cert-manager.io/v1`, (c) specifies the operator's required RBAC permissions, (d) includes install strategy with the controller deployment.

<details>
<summary>Show Answer</summary>

```yaml
apiVersion: operators.coreos.com/v1alpha1
kind: ClusterServiceVersion
metadata:
  name: memcached-operator.v0.2.0
  namespace: placeholder
  annotations:
    alm-examples: |
      [
        {
          "apiVersion": "cache.example.com/v1alpha1",
          "kind": "Memcached",
          "metadata": {"name": "memcached-sample"},
          "spec": {"size": 3, "containerPort": 11211}
        }
      ]
    capabilities: "Full Lifecycle"
    categories: "Database"
spec:
  displayName: Memcached Operator
  description: Manages Memcached clusters on Kubernetes
  version: 0.2.0
  replaces: memcached-operator.v0.1.0
  maturity: beta
  minKubeVersion: "1.25.0"

  maintainers:
  - name: Example Inc.
    email: ops@example.com

  customresourcedefinitions:
    owned:
    - name: memcacheds.cache.example.com
      version: v1alpha1
      kind: Memcached
      displayName: Memcached
      description: A Memcached cluster
      statusDescriptors:
      - path: readyReplicas
        displayName: Ready Replicas
        description: Number of ready replicas

    required:
    - name: certificates.cert-manager.io
      version: v1
      kind: Certificate
      displayName: Certificate
      description: TLS certificate management

  install:
    strategy: deployment
    spec:
      clusterPermissions:
      - serviceAccountName: memcached-operator-controller-manager
        rules:
        - apiGroups: ["cache.example.com"]
          resources: ["memcacheds"]
          verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
        - apiGroups: ["cache.example.com"]
          resources: ["memcacheds/status"]
          verbs: ["get", "update", "patch"]
        - apiGroups: ["cache.example.com"]
          resources: ["memcacheds/finalizers"]
          verbs: ["update"]
        - apiGroups: ["apps"]
          resources: ["deployments"]
          verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
        - apiGroups: [""]
          resources: ["pods", "services", "configmaps"]
          verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
        - apiGroups: [""]
          resources: ["events"]
          verbs: ["create", "patch"]
        - apiGroups: ["coordination.k8s.io"]
          resources: ["leases"]
          verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
      deployments:
      - name: memcached-operator-controller-manager
        spec:
          replicas: 2
          selector:
            matchLabels:
              control-plane: controller-manager
          template:
            metadata:
              labels:
                control-plane: controller-manager
            spec:
              serviceAccountName: memcached-operator-controller-manager
              containers:
              - name: manager
                image: example.com/memcached-operator:v0.2.0
                args:
                - --leader-elect
                ports:
                - containerPort: 8080
                  name: metrics
                resources:
                  requests:
                    cpu: 100m
                    memory: 128Mi
                  limits:
                    cpu: 500m
                    memory: 256Mi

  installModes:
  - type: OwnNamespace
    supported: true
  - type: SingleNamespace
    supported: true
  - type: MultiNamespace
    supported: false
  - type: AllNamespaces
    supported: true
```

</details>

---

**Previous**: [Custom Resource Definitions](./10_Custom_Resource_Definitions.md) | **Next**: [Admission Controllers](./12_Admission_Controllers.md)

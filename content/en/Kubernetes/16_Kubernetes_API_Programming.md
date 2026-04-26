# 16. Kubernetes API Programming

**Previous**: [15. Multi-Cluster](./15_Multi_Cluster.md) | **Next**: [17. Production Operations](./17_Production_Operations.md)

## Learning Objectives

- Understand the Kubernetes API structure including groups, versions, and resources
- Use client-go to interact with the Kubernetes API programmatically in Go
- Build custom controllers using informers, work queues, and the reconciliation loop
- Leverage the controller-runtime library for production-grade controller development
- Test controllers using envtest and integration testing patterns

---

The Kubernetes API server is the central hub of every cluster. Every `kubectl` command, every controller, and every operator communicates through this single RESTful interface. Understanding how to program against the Kubernetes API unlocks the ability to build custom automation, extend the platform with new behaviors, and integrate Kubernetes into larger systems. This lesson teaches you to write Go programs that interact with the API server — from simple CRUD operations to full-blown controllers that watch resources and reconcile state continuously.

Before the Go code, read [**Theory & Principles**](#theory--principles) — the GVR/GVK system that catalogs every Kubernetes resource, why client-go's typed clientsets and the dynamic client trade compile-time safety against generality, the informer + work queue pattern that powers every controller, and the testing strategies (envtest, fake client) that let you verify reconciliation logic without a real cluster.

## Table of Contents

- [Theory & Principles](#theory--principles)
- [1. Kubernetes API Structure](#1-kubernetes-api-structure)
- [2. The client-go Library](#2-the-client-go-library)
- [3. REST Client and Clientsets](#3-rest-client-and-clientsets)
- [4. Dynamic Client and Unstructured Objects](#4-dynamic-client-and-unstructured-objects)
- [5. Informers and Caching](#5-informers-and-caching)
- [6. Work Queues](#6-work-queues)
- [7. Building a Custom Controller](#7-building-a-custom-controller)
- [8. Controller-Runtime Library](#8-controller-runtime-library)
- [9. Watching Resources and Handling Events](#9-watching-resources-and-handling-events)
- [10. Testing Controllers](#10-testing-controllers)
- [Exercises](#exercises)

---

## Theory & Principles

Programming the Kubernetes API in Go is what every controller, operator, and platform tool you've used (Argo CD, cert-manager, Prometheus Operator, ...) does under the hood. The API server itself is a RESTful HTTP+JSON service — you could call it with `curl` — but production code uses the **client-go** library because it provides typed access, caching (informers), efficient change notification (watches), and a work-queue pattern that makes reconciliation correct under restart and concurrency. This section explains the resource taxonomy (GVR/GVK), the client choices, the informer architecture (which is also the foundation of operator-runtime in lesson 11), and the testing approaches that distinguish hobby code from production controllers.

### A. The Resource Taxonomy: GVR and GVK

Every Kubernetes resource has two parallel identities:

**GVK (Group, Version, Kind)** identifies the *Go type*: e.g., `apps/v1.Deployment`. This is what your code constructs and inspects (`appsv1.Deployment{...}`). Kinds are PascalCase and singular.

**GVR (Group, Version, Resource)** identifies the *REST URL path*: e.g., `apps/v1/deployments`. This is what shows up in URLs (`/apis/apps/v1/namespaces/default/deployments`) and what RBAC rules reference (`apiGroups: ["apps"], resources: ["deployments"]`). Resources are lowercase and plural.

The mapping between them is via the API server's **discovery** endpoint, which lists every registered (group, version) and the kinds and resources within. The library `RESTMapper` does this lookup for you so you can write `meta.RESTMapper.RESTMapping(GroupKind, version)` and get back the right URL fragment.

Why two? Because the wire format and the in-memory representation evolve independently. A `Deployment` Kind always means "the same conceptual object," but its REST resource path could (in principle) change between API versions. Most code uses Kind in Go (`*appsv1.Deployment`) and only touches Resource at the RBAC and dynamic-client layer.

### B. Three Client Styles: Typed, Discovery, Dynamic

client-go offers three ways to talk to the API server:

**1. Typed clientset (`kubernetes.Clientset`)**: a Go-typed interface for built-in resources. You write `clientset.AppsV1().Deployments("default").Get(ctx, "my-app", metav1.GetOptions{})` and get back `*appsv1.Deployment`. Compile-time safety, IDE autocomplete, easy to refactor. **Limitation**: only works for resources whose types were known at clientset compile time — built-ins and CRDs you've generated typed clients for.

**2. Dynamic client (`dynamic.Interface`)**: works on `unstructured.Unstructured` (a `map[string]interface{}`). You construct a GVR, get a `ResourceInterface`, and operate on `*unstructured.Unstructured` objects. Trade compile-time safety for the ability to handle arbitrary CRDs without code generation. **Use when**: writing a generic operator (like Argo CD) that handles user-supplied CRDs unknown at build time.

**3. controller-runtime client (`client.Client`)**: introduced in lesson 11; built on top of clientset but unified for built-in and custom types via runtime registration. The standard for new controllers because it integrates cleanly with the Reconciler pattern.

Behind each is a `rest.RESTClient` that handles the HTTP, auth, content negotiation (JSON vs protobuf), and rate limiting. You rarely interact with this layer directly; the higher-level clients wrap it.

The **discovery client** (`discovery.DiscoveryInterface`) is a fourth, special-purpose client — it returns the API server's list of available groups/versions/resources. Useful for tools that need to enumerate "what can I work with on this cluster?"

### C. Informer Architecture: List-Watch + Cache + Indexed Reads

Every controller needs to watch resources for changes. Doing this naively (a watch HTTP connection per resource per controller) doesn't scale. The **informer** pattern solves this with a shared cache:

```
API Server ←─watch─ Informer ─→ Indexer (cache) ─→ Lister
                       │
                       └─→ Event Handler ─→ Work Queue ─→ Reconciler
```

**SharedInformerFactory** creates one informer per (resource, namespace) and shares it across all consumers. So if your operator watches Deployments and your CRD controller also watches Deployments, only one watch HTTP connection is opened. The factory tracks reference counts and cleans up when no consumer remains.

**The Indexer** is the local cache. It holds the result of the initial list plus all subsequent watch deltas. Reads (Get, List) hit the indexer, never the API server — which means a controller can list 10,000 pods locally in microseconds rather than make a multi-megabyte API server round trip. Custom indexes can be built on labels, fields, or arbitrary functions for fast lookup ("give me all pods owned by ReplicaSet X" without scanning).

**Event handlers** are user-provided callbacks invoked on `ADDED`/`MODIFIED`/`DELETED`. The standard pattern: handlers do *not* do work; they extract a key (`namespace/name`) and `Add()` it to a work queue. This decouples event speed from work speed — burst events are absorbed by the queue, and work proceeds at the reconciler's pace.

**Work queue** (`workqueue.RateLimitingInterface`) provides three properties critical to correct controllers:
- **Deduplication**: 100 events for the same key result in one reconcile.
- **Per-key serialization**: only one worker reconciles a given key at a time.
- **Rate limiting**: failed reconciles back off exponentially.

The **Reconciler** is your code. It pulls a key from the queue, gets the current object from the indexer, computes the desired state, and acts. On error, it returns the key to the queue for retry; on success, it forgets it. This is the same pattern from lesson 11; here we see it from the lower-level client-go perspective.

### D. Testing Controllers: envtest, Fake Client, and Why Both Exist

Controllers are notoriously hard to test because they depend on the API server's behavior — admission, defaulting, status updates, watch semantics. Two complementary approaches:

**Fake client** (`fake.NewSimpleClientset`): an in-memory implementation of the clientset interface that records actions and returns canned responses. Pros: blazingly fast (microseconds per operation), no external dependencies, easy to assert "controller called Update with X." Cons: doesn't run admission, doesn't enforce schema, doesn't generate watch events properly across goroutines. Best for unit tests of pure reconciler logic.

**envtest** (controller-runtime): boots a real `etcd` and `kube-apiserver` binary in your test process. Pros: exercises real API behavior including admission, validation, defaulting, watches. Your reconciler runs against a real API server. Cons: slower (~5s startup, ~100ms per operation); requires the kubebuilder envtest binaries to be installed. Best for integration tests of controller behavior end-to-end.

A common test layout: fast unit tests on reconciler logic with the fake client (fail-fast in CI), plus a smaller suite of envtest-based integration tests that exercise the full reconciler-API interaction (slow but high confidence).

A subtle point: a controller that passes fake-client tests but fails envtest tests is usually relying on something the fake client doesn't simulate (admission webhooks, server-side apply semantics, watch event ordering). When debugging "works in test, fails in cluster," envtest is closer to truth.

### From Theory to the Code Below

The lesson now applies these abstractions:

- **Section 1 (Kubernetes API Structure)** is §A — GVR/GVK, discovery, the API server's resource graph.
- **Section 2 (The client-go Library)** is §B's overview — the package layout and high-level design.
- **Section 3 (REST Client and Clientsets)** is §B's typed client in concrete code.
- **Section 4 (Dynamic Client and Unstructured Objects)** is §B's dynamic client for generic tools.
- **Section 5 (Informers and Caching)** is §C's informer architecture in code with `SharedInformerFactory`.
- **Section 6 (Work Queues)** is §C's queue with rate limiting and per-key serialization.
- **Section 7 (Building a Custom Controller)** stitches §C together: informer + queue + reconciler in a runnable program.
- **Section 8 (Controller-Runtime Library)** is the higher-level abstraction (lesson 11) on top of the same primitives.
- **Section 9 (Watching Resources and Handling Events)** is the event-handler patterns (filter, requeue, owner-reference watches).
- **Section 10 (Testing Controllers)** is §D — fake client and envtest in practice.

Once you see GVR/GVK as the resource taxonomy, the three client styles as a generality-vs-safety trade-off, and informer + queue + reconciler as the universal controller pattern, every Kubernetes-aware Go program decomposes into the same building blocks.

---

## 1. Kubernetes API Structure

### 1.1 API Groups and Versions

The Kubernetes API is organized into **API groups** that allow independent versioning and evolution. Each resource belongs to a group, has a version, and exposes standard REST operations.

```
API Group Structure:
┌──────────────────────────────────────────────────────────────┐
│                      API Server                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Core Group ("")           /api/v1                           │
│  ├── pods                                                    │
│  ├── services                                                │
│  ├── configmaps                                              │
│  ├── secrets                                                 │
│  ├── namespaces                                              │
│  └── nodes                                                   │
│                                                              │
│  Named Groups              /apis/<group>/<version>           │
│  ├── apps/v1               Deployments, StatefulSets, ...    │
│  ├── batch/v1              Jobs, CronJobs                    │
│  ├── networking.k8s.io/v1  NetworkPolicies, Ingresses        │
│  ├── rbac.authorization.k8s.io/v1  Roles, RoleBindings       │
│  ├── autoscaling/v2        HPA                               │
│  └── apiextensions.k8s.io/v1  CustomResourceDefinitions      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

Every resource in Kubernetes is identified by a **GroupVersionResource (GVR)** triple:

```go
// GroupVersionResource uniquely identifies a resource type
import "k8s.io/apimachinery/pkg/runtime/schema"

// Deployments in the apps/v1 group
deploymentsGVR := schema.GroupVersionResource{
    Group:    "apps",
    Version:  "v1",
    Resource: "deployments",
}

// Pods in the core group (empty string)
podsGVR := schema.GroupVersionResource{
    Group:    "",
    Version:  "v1",
    Resource: "pods",
}
```

### 1.2 Resource and Kind

A **Resource** is the URL path component (always lowercase plural: `pods`, `deployments`). A **Kind** is the Go type name used in serialization (always CamelCase singular: `Pod`, `Deployment`). The mapping is tracked by a **scheme**.

```go
// GroupVersionKind identifies the Go type
deploymentsGVK := schema.GroupVersionKind{
    Group:   "apps",
    Version: "v1",
    Kind:    "Deployment",
}
```

### 1.3 API Discovery

The API server exposes discovery endpoints that enumerate available resources:

```bash
# List all API groups
kubectl api-versions

# List all resources with their group, version, kind, and verbs
kubectl api-resources -o wide

# Explore a specific resource
kubectl explain deployment.spec.strategy --api-version=apps/v1

# Raw API call to the discovery endpoint
kubectl get --raw /apis | jq '.groups[].name'

# Get resources for a specific group/version
kubectl get --raw /apis/apps/v1 | jq '.resources[].name'
```

### 1.4 API Request Anatomy

Every API request follows a predictable URL pattern:

```
Namespaced resource:
  GET /apis/<group>/<version>/namespaces/<namespace>/<resource>/<name>

Cluster-scoped resource:
  GET /apis/<group>/<version>/<resource>/<name>

Core group (no group prefix):
  GET /api/v1/namespaces/<namespace>/<resource>/<name>

Examples:
  GET /apis/apps/v1/namespaces/default/deployments/nginx
  GET /api/v1/namespaces/kube-system/pods
  GET /apis/rbac.authorization.k8s.io/v1/clusterroles
```

Responses include metadata that enables optimistic concurrency:

```yaml
# Every Kubernetes object includes this metadata
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
  namespace: default
  uid: "a1b2c3d4-..."
  resourceVersion: "12345"       # Optimistic concurrency token
  generation: 3                   # Spec change counter
  creationTimestamp: "2025-01-15T10:00:00Z"
```

The `resourceVersion` field is critical for the watch protocol — it tells the API server where to start streaming changes.

---

## 2. The client-go Library

### 2.1 Overview

`client-go` is the official Go client library for Kubernetes. It provides typed clients, informers, caching, and utilities that most Go-based Kubernetes tools depend on.

```
client-go Architecture:
┌───────────────────────────────────────────────────┐
│                  Your Application                 │
├───────────────────────────────────────────────────┤
│  Clientset  │  Dynamic Client  │  Discovery       │
├─────────────┴──────────────────┴──────────────────┤
│           REST Client (rest.Interface)            │
├───────────────────────────────────────────────────┤
│           Transport (TLS, auth, retry)            │
├───────────────────────────────────────────────────┤
│               HTTP/2 to API Server                │
└───────────────────────────────────────────────────┘
```

### 2.2 Project Setup

```bash
# Initialize a Go module
mkdir k8s-controller && cd k8s-controller
go mod init github.com/example/k8s-controller

# Add client-go (version should match your cluster)
go get k8s.io/client-go@v0.29.0
go get k8s.io/apimachinery@v0.29.0
go get k8s.io/api@v0.29.0
```

### 2.3 Connecting to a Cluster

```go
package main

import (
    "context"
    "fmt"
    "os"
    "path/filepath"

    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/client-go/kubernetes"
    "k8s.io/client-go/rest"
    "k8s.io/client-go/tools/clientcmd"
)

func main() {
    config, err := getConfig()
    if err != nil {
        panic(err)
    }

    clientset, err := kubernetes.NewForConfig(config)
    if err != nil {
        panic(err)
    }

    // List pods in default namespace
    pods, err := clientset.CoreV1().Pods("default").List(
        context.TODO(),
        metav1.ListOptions{},
    )
    if err != nil {
        panic(err)
    }

    for _, pod := range pods.Items {
        fmt.Printf("Pod: %s (Phase: %s)\n", pod.Name, pod.Status.Phase)
    }
}

// getConfig returns in-cluster config or falls back to kubeconfig
func getConfig() (*rest.Config, error) {
    // Try in-cluster config first (when running inside a pod)
    config, err := rest.InClusterConfig()
    if err == nil {
        return config, nil
    }

    // Fall back to kubeconfig
    kubeconfig := filepath.Join(os.Getenv("HOME"), ".kube", "config")
    if envKC := os.Getenv("KUBECONFIG"); envKC != "" {
        kubeconfig = envKC
    }
    return clientcmd.BuildConfigFromFlags("", kubeconfig)
}
```

### 2.4 Authentication Methods

```go
// Method 1: Bearer token (ServiceAccount)
config := &rest.Config{
    Host:        "https://api-server:6443",
    BearerToken: "eyJhbGciOiJSU...",
    TLSClientConfig: rest.TLSClientConfig{
        CAFile: "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
    },
}

// Method 2: Client certificate
config := &rest.Config{
    Host: "https://api-server:6443",
    TLSClientConfig: rest.TLSClientConfig{
        CertFile: "/path/to/client.crt",
        KeyFile:  "/path/to/client.key",
        CAFile:   "/path/to/ca.crt",
    },
}

// Method 3: Configure rate limiting
config.QPS = 50       // Queries per second (default: 5)
config.Burst = 100     // Burst capacity (default: 10)
```

---

## 3. REST Client and Clientsets

### 3.1 Typed Clientsets

A **clientset** provides strongly-typed methods for every built-in Kubernetes resource:

```go
import (
    "context"
    appsv1 "k8s.io/api/apps/v1"
    corev1 "k8s.io/api/core/v1"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/client-go/kubernetes"
    "k8s.io/utils/ptr"
)

func crudOperations(clientset kubernetes.Interface) error {
    ctx := context.TODO()

    // CREATE a Deployment
    deploy := &appsv1.Deployment{
        ObjectMeta: metav1.ObjectMeta{
            Name:      "web-server",
            Namespace: "default",
            Labels: map[string]string{
                "app": "web-server",
            },
        },
        Spec: appsv1.DeploymentSpec{
            Replicas: ptr.To(int32(3)),
            Selector: &metav1.LabelSelector{
                MatchLabels: map[string]string{"app": "web-server"},
            },
            Template: corev1.PodTemplateSpec{
                ObjectMeta: metav1.ObjectMeta{
                    Labels: map[string]string{"app": "web-server"},
                },
                Spec: corev1.PodSpec{
                    Containers: []corev1.Container{
                        {
                            Name:  "nginx",
                            Image: "nginx:1.25",
                            Ports: []corev1.ContainerPort{
                                {ContainerPort: 80},
                            },
                        },
                    },
                },
            },
        },
    }

    created, err := clientset.AppsV1().Deployments("default").Create(
        ctx, deploy, metav1.CreateOptions{},
    )
    if err != nil {
        return fmt.Errorf("create deployment: %w", err)
    }
    fmt.Printf("Created deployment: %s (rv=%s)\n",
        created.Name, created.ResourceVersion)

    // GET a Deployment
    fetched, err := clientset.AppsV1().Deployments("default").Get(
        ctx, "web-server", metav1.GetOptions{},
    )
    if err != nil {
        return fmt.Errorf("get deployment: %w", err)
    }

    // UPDATE (scale to 5 replicas)
    fetched.Spec.Replicas = ptr.To(int32(5))
    updated, err := clientset.AppsV1().Deployments("default").Update(
        ctx, fetched, metav1.UpdateOptions{},
    )
    if err != nil {
        return fmt.Errorf("update deployment: %w", err)
    }
    fmt.Printf("Updated replicas to %d (rv=%s)\n",
        *updated.Spec.Replicas, updated.ResourceVersion)

    // LIST with label selector
    deploys, err := clientset.AppsV1().Deployments("default").List(
        ctx, metav1.ListOptions{
            LabelSelector: "app=web-server",
        },
    )
    if err != nil {
        return fmt.Errorf("list deployments: %w", err)
    }
    fmt.Printf("Found %d deployments\n", len(deploys.Items))

    // DELETE
    err = clientset.AppsV1().Deployments("default").Delete(
        ctx, "web-server", metav1.DeleteOptions{},
    )
    return err
}
```

### 3.2 Status Subresource

Many resources have a `/status` subresource that is updated separately from the spec:

```go
func updateDeploymentStatus(
    clientset kubernetes.Interface,
    name, namespace string,
) error {
    ctx := context.TODO()

    deploy, err := clientset.AppsV1().Deployments(namespace).Get(
        ctx, name, metav1.GetOptions{},
    )
    if err != nil {
        return err
    }

    // Modify status fields
    deploy.Status.Conditions = append(deploy.Status.Conditions,
        appsv1.DeploymentCondition{
            Type:               appsv1.DeploymentProgressing,
            Status:             corev1.ConditionTrue,
            LastTransitionTime: metav1.Now(),
            Reason:             "NewReplicaSetAvailable",
            Message:            "Deployment has minimum availability",
        },
    )

    // Use UpdateStatus - only writes the status subresource
    _, err = clientset.AppsV1().Deployments(namespace).UpdateStatus(
        ctx, deploy, metav1.UpdateOptions{},
    )
    return err
}
```

### 3.3 Server-Side Apply

Server-Side Apply (SSA) allows multiple managers to own different fields of the same object safely:

```go
import (
    "encoding/json"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/types"
)

func serverSideApply(clientset kubernetes.Interface) error {
    ctx := context.TODO()

    // Define the partial object you want to apply
    patch := map[string]interface{}{
        "apiVersion": "apps/v1",
        "kind":       "Deployment",
        "metadata": map[string]interface{}{
            "name":      "web-server",
            "namespace": "default",
        },
        "spec": map[string]interface{}{
            "replicas": 5,
        },
    }

    patchBytes, err := json.Marshal(patch)
    if err != nil {
        return err
    }

    // Apply with a unique field manager name
    _, err = clientset.AppsV1().Deployments("default").Patch(
        ctx,
        "web-server",
        types.ApplyPatchType,
        patchBytes,
        metav1.PatchOptions{
            FieldManager: "my-controller",
        },
    )
    return err
}
```

---

## 4. Dynamic Client and Unstructured Objects

### 4.1 When to Use Dynamic Client

The dynamic client works with any resource type, including Custom Resources, without requiring generated Go types. It uses `unstructured.Unstructured` objects — essentially `map[string]interface{}` wrappers.

```go
import (
    "context"
    "fmt"

    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
    "k8s.io/apimachinery/pkg/runtime/schema"
    "k8s.io/client-go/dynamic"
    "k8s.io/client-go/rest"
)

func dynamicClientExample(config *rest.Config) error {
    dynClient, err := dynamic.NewForConfig(config)
    if err != nil {
        return err
    }

    ctx := context.TODO()

    // Define the GVR for a custom resource
    certificateGVR := schema.GroupVersionResource{
        Group:    "cert-manager.io",
        Version:  "v1",
        Resource: "certificates",
    }

    // Create an unstructured custom resource
    cert := &unstructured.Unstructured{
        Object: map[string]interface{}{
            "apiVersion": "cert-manager.io/v1",
            "kind":       "Certificate",
            "metadata": map[string]interface{}{
                "name":      "my-tls-cert",
                "namespace": "default",
            },
            "spec": map[string]interface{}{
                "secretName": "my-tls-secret",
                "issuerRef": map[string]interface{}{
                    "name": "letsencrypt-prod",
                    "kind": "ClusterIssuer",
                },
                "dnsNames": []interface{}{
                    "example.com",
                    "www.example.com",
                },
            },
        },
    }

    // Create the resource
    created, err := dynClient.Resource(certificateGVR).Namespace("default").Create(
        ctx, cert, metav1.CreateOptions{},
    )
    if err != nil {
        return err
    }

    // Read nested fields safely
    secretName, found, err := unstructured.NestedString(
        created.Object, "spec", "secretName",
    )
    if err != nil || !found {
        return fmt.Errorf("secretName not found")
    }
    fmt.Printf("Certificate created, secretName: %s\n", secretName)

    // List all certificates
    certs, err := dynClient.Resource(certificateGVR).Namespace("").List(
        ctx, metav1.ListOptions{},
    )
    if err != nil {
        return err
    }

    for _, c := range certs.Items {
        ns := c.GetNamespace()
        name := c.GetName()
        fmt.Printf("  %s/%s\n", ns, name)
    }

    return nil
}
```

### 4.2 Unstructured Helpers

```go
import "k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"

obj := &unstructured.Unstructured{}

// Read nested fields
replicas, found, err := unstructured.NestedInt64(obj.Object, "spec", "replicas")
labels, found, err := unstructured.NestedStringMap(obj.Object, "metadata", "labels")
containers, found, err := unstructured.NestedSlice(obj.Object, "spec", "template", "spec", "containers")

// Write nested fields
err = unstructured.SetNestedField(obj.Object, int64(3), "spec", "replicas")
err = unstructured.SetNestedStringMap(obj.Object,
    map[string]string{"app": "web"},
    "metadata", "labels",
)

// Standard metadata accessors
name := obj.GetName()
ns := obj.GetNamespace()
rv := obj.GetResourceVersion()
obj.SetLabels(map[string]string{"managed-by": "my-controller"})
```

---

## 5. Informers and Caching

### 5.1 The Watch Problem

Polling the API server for changes is inefficient. Watches use HTTP/2 streaming but reconnection and bookmark handling are complex. Informers solve this by maintaining a local cache that stays in sync with the API server.

```
Informer Architecture:
                                                  ┌─────────────┐
                                                  │  Indexer     │
                                           ┌─────►│  (Cache)     │
                                           │      └─────────────┘
┌──────────┐    LIST + WATCH    ┌──────────┤
│ API      │ ─────────────────► │ Reflector│      ┌─────────────┐
│ Server   │    HTTP/2 stream   │          │─────►│  DeltaFIFO  │
└──────────┘                    └──────────┘      │  Queue       │
                                                  └──────┬──────┘
                                                         │
                                                  ┌──────▼──────┐
                                                  │  Event       │
                                                  │  Handlers    │
                                                  │  (Add/       │
                                                  │   Update/    │
                                                  │   Delete)    │
                                                  └─────────────┘
```

### 5.2 Using a Single Informer

```go
import (
    "fmt"
    "time"

    corev1 "k8s.io/api/core/v1"
    "k8s.io/client-go/informers"
    "k8s.io/client-go/kubernetes"
    "k8s.io/client-go/tools/cache"
)

func singleInformerExample(clientset kubernetes.Interface) {
    // Create a SharedInformerFactory with 30-second resync
    factory := informers.NewSharedInformerFactory(clientset, 30*time.Second)

    // Get an informer for Pods
    podInformer := factory.Core().V1().Pods().Informer()

    // Register event handlers
    podInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
        AddFunc: func(obj interface{}) {
            pod := obj.(*corev1.Pod)
            fmt.Printf("[ADD] Pod %s/%s\n", pod.Namespace, pod.Name)
        },
        UpdateFunc: func(oldObj, newObj interface{}) {
            oldPod := oldObj.(*corev1.Pod)
            newPod := newObj.(*corev1.Pod)
            if oldPod.ResourceVersion != newPod.ResourceVersion {
                fmt.Printf("[UPDATE] Pod %s/%s (rv: %s -> %s)\n",
                    newPod.Namespace, newPod.Name,
                    oldPod.ResourceVersion, newPod.ResourceVersion)
            }
        },
        DeleteFunc: func(obj interface{}) {
            pod := obj.(*corev1.Pod)
            fmt.Printf("[DELETE] Pod %s/%s\n", pod.Namespace, pod.Name)
        },
    })

    // Start the informer (non-blocking)
    stopCh := make(chan struct{})
    factory.Start(stopCh)

    // Wait for the initial cache sync
    factory.WaitForCacheSync(stopCh)
    fmt.Println("Cache synced, watching for changes...")

    // Read from the cache (no API call)
    lister := factory.Core().V1().Pods().Lister()
    pods, err := lister.Pods("default").List(labels.Everything())
    if err == nil {
        fmt.Printf("Cached pods in default: %d\n", len(pods))
    }

    // Block until stopped
    <-stopCh
}
```

### 5.3 SharedInformerFactory

The factory ensures that only one watch per resource type is created, even if multiple components need to observe the same resource:

```go
// Namespace-scoped factory (only watches resources in "production")
factory := informers.NewSharedInformerFactoryWithOptions(
    clientset,
    30*time.Second,
    informers.WithNamespace("production"),
)

// Use multiple informers from the same factory
deployInformer := factory.Apps().V1().Deployments().Informer()
svcInformer := factory.Core().V1().Services().Informer()
nodeInformer := factory.Core().V1().Nodes().Informer()

// All share the same stop channel
stopCh := make(chan struct{})
factory.Start(stopCh)
factory.WaitForCacheSync(stopCh)
```

### 5.4 Indexers

Informer caches support custom indexes for fast lookups:

```go
const byNodeIndex = "byNode"

podInformer := factory.Core().V1().Pods().Informer()

// Add a custom index: index pods by their node name
podInformer.AddIndexers(cache.Indexers{
    byNodeIndex: func(obj interface{}) ([]string, error) {
        pod := obj.(*corev1.Pod)
        if pod.Spec.NodeName == "" {
            return nil, nil
        }
        return []string{pod.Spec.NodeName}, nil
    },
})

// After cache sync, look up pods by node
indexer := podInformer.GetIndexer()
items, err := indexer.ByIndex(byNodeIndex, "worker-node-1")
if err == nil {
    fmt.Printf("Pods on worker-node-1: %d\n", len(items))
}
```

---

## 6. Work Queues

### 6.1 Why Use Work Queues

Event handlers in informers should be lightweight — they run on the informer goroutine. Heavy processing must be offloaded to workers that consume from a queue.

```
Event Flow:
                                     ┌──────────────┐
Informer ──► AddFunc ─────────┐      │              │
             UpdateFunc ──────┼─────►│  Work Queue  │──► Worker 1 ──► Reconcile
             DeleteFunc ──────┘      │              │──► Worker 2 ──► Reconcile
                                     └──────────────┘──► Worker N ──► Reconcile
```

### 6.2 Rate-Limiting Work Queue

```go
import (
    "time"

    "k8s.io/client-go/util/workqueue"
)

// Create a rate-limiting queue with exponential backoff
queue := workqueue.NewRateLimitingQueueWithConfig(
    workqueue.NewItemExponentialFailureRateLimiter(
        200*time.Millisecond, // base delay
        5*time.Minute,        // max delay
    ),
    workqueue.RateLimitingQueueConfig{
        Name: "my-controller",
    },
)
defer queue.ShutDown()

// Enqueue an item (typically namespace/name key)
key := "default/my-pod"
queue.Add(key)

// Enqueue after a delay
queue.AddAfter("default/retry-me", 30*time.Second)

// Worker loop
for {
    item, shutdown := queue.Get()
    if shutdown {
        break
    }

    // Process the item
    key := item.(string)
    err := processItem(key)

    if err != nil {
        // Re-enqueue with rate limiting on failure
        if queue.NumRequeues(item) < 5 {
            queue.AddRateLimited(item)
        } else {
            // Give up after too many retries
            queue.Forget(item)
        }
    } else {
        // Remove from rate limiter tracking on success
        queue.Forget(item)
    }

    // Mark item as done (allows re-enqueueing)
    queue.Done(item)
}
```

### 6.3 Deriving Queue Keys

```go
import "k8s.io/client-go/tools/cache"

// In event handlers, enqueue the object's key
handler := cache.ResourceEventHandlerFuncs{
    AddFunc: func(obj interface{}) {
        key, err := cache.MetaNamespaceKeyFunc(obj)
        if err == nil {
            queue.Add(key)
        }
    },
    UpdateFunc: func(oldObj, newObj interface{}) {
        key, err := cache.MetaNamespaceKeyFunc(newObj)
        if err == nil {
            queue.Add(key)
        }
    },
    DeleteFunc: func(obj interface{}) {
        // Handle "DeletedFinalStateUnknown" tombstones
        key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
        if err == nil {
            queue.Add(key)
        }
    },
}

// In the worker, split the key back
namespace, name, err := cache.SplitMetaNamespaceKey(key)
```

---

## 7. Building a Custom Controller

### 7.1 Controller Architecture

A complete custom controller ties together informers, a work queue, and a reconciliation loop. Here is a controller that watches ConfigMaps and logs changes:

```go
package main

import (
    "context"
    "fmt"
    "os"
    "os/signal"
    "syscall"
    "time"

    corev1 "k8s.io/api/core/v1"
    "k8s.io/apimachinery/pkg/api/errors"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    utilruntime "k8s.io/apimachinery/pkg/util/runtime"
    "k8s.io/apimachinery/pkg/util/wait"
    "k8s.io/client-go/informers"
    "k8s.io/client-go/kubernetes"
    corev1listers "k8s.io/client-go/listers/core/v1"
    "k8s.io/client-go/tools/cache"
    "k8s.io/client-go/tools/clientcmd"
    "k8s.io/client-go/util/workqueue"
    "k8s.io/klog/v2"
)

// ConfigMapController watches ConfigMaps and annotates them
// with a last-seen timestamp.
type ConfigMapController struct {
    clientset kubernetes.Interface
    lister    corev1listers.ConfigMapLister
    hasSynced cache.InformerSynced
    queue     workqueue.RateLimitingInterface
}

func NewConfigMapController(
    clientset kubernetes.Interface,
    factory informers.SharedInformerFactory,
) *ConfigMapController {
    informer := factory.Core().V1().ConfigMaps()

    c := &ConfigMapController{
        clientset: clientset,
        lister:    informer.Lister(),
        hasSynced: informer.Informer().HasSynced,
        queue: workqueue.NewRateLimitingQueueWithConfig(
            workqueue.DefaultControllerRateLimiter(),
            workqueue.RateLimitingQueueConfig{Name: "configmap-controller"},
        ),
    }

    informer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
        AddFunc: func(obj interface{}) {
            c.enqueue(obj)
        },
        UpdateFunc: func(oldObj, newObj interface{}) {
            c.enqueue(newObj)
        },
    })

    return c
}

func (c *ConfigMapController) enqueue(obj interface{}) {
    key, err := cache.MetaNamespaceKeyFunc(obj)
    if err != nil {
        utilruntime.HandleError(err)
        return
    }
    c.queue.Add(key)
}

func (c *ConfigMapController) Run(ctx context.Context, workers int) error {
    defer utilruntime.HandleCrash()
    defer c.queue.ShutDown()

    klog.Info("Starting ConfigMap controller")

    klog.Info("Waiting for informer caches to sync")
    if !cache.WaitForCacheSync(ctx.Done(), c.hasSynced) {
        return fmt.Errorf("failed to sync caches")
    }

    klog.Infof("Starting %d workers", workers)
    for i := 0; i < workers; i++ {
        go wait.UntilWithContext(ctx, c.runWorker, time.Second)
    }

    klog.Info("Controller started")
    <-ctx.Done()
    klog.Info("Shutting down controller")
    return nil
}

func (c *ConfigMapController) runWorker(ctx context.Context) {
    for c.processNextItem(ctx) {
    }
}

func (c *ConfigMapController) processNextItem(ctx context.Context) bool {
    item, shutdown := c.queue.Get()
    if shutdown {
        return false
    }
    defer c.queue.Done(item)

    key := item.(string)
    err := c.reconcile(ctx, key)
    if err == nil {
        c.queue.Forget(item)
        return true
    }

    utilruntime.HandleError(fmt.Errorf("reconcile %s: %w", key, err))
    if c.queue.NumRequeues(item) < 5 {
        c.queue.AddRateLimited(item)
    } else {
        c.queue.Forget(item)
        klog.Errorf("Dropping %s after too many retries", key)
    }
    return true
}

func (c *ConfigMapController) reconcile(ctx context.Context, key string) error {
    namespace, name, err := cache.SplitMetaNamespaceKey(key)
    if err != nil {
        return err
    }

    cm, err := c.lister.ConfigMaps(namespace).Get(name)
    if errors.IsNotFound(err) {
        klog.Infof("ConfigMap %s deleted", key)
        return nil
    }
    if err != nil {
        return err
    }

    // Skip system namespaces
    if namespace == "kube-system" || namespace == "kube-public" {
        return nil
    }

    // Add last-reconciled annotation
    cmCopy := cm.DeepCopy()
    if cmCopy.Annotations == nil {
        cmCopy.Annotations = make(map[string]string)
    }
    timestamp := time.Now().UTC().Format(time.RFC3339)
    if cmCopy.Annotations["controller/last-reconciled"] == timestamp {
        return nil // Already up to date
    }
    cmCopy.Annotations["controller/last-reconciled"] = timestamp

    _, err = c.clientset.CoreV1().ConfigMaps(namespace).Update(
        ctx, cmCopy, metav1.UpdateOptions{},
    )
    if err != nil {
        return fmt.Errorf("update configmap: %w", err)
    }
    klog.Infof("Reconciled ConfigMap %s", key)
    return nil
}

func main() {
    klog.InitFlags(nil)

    config, err := clientcmd.BuildConfigFromFlags("",
        os.Getenv("HOME")+"/.kube/config")
    if err != nil {
        klog.Fatalf("Error building config: %v", err)
    }

    clientset, err := kubernetes.NewForConfig(config)
    if err != nil {
        klog.Fatalf("Error creating clientset: %v", err)
    }

    factory := informers.NewSharedInformerFactory(clientset, 30*time.Second)
    controller := NewConfigMapController(clientset, factory)

    ctx, cancel := signal.NotifyContext(context.Background(),
        syscall.SIGINT, syscall.SIGTERM)
    defer cancel()

    factory.Start(ctx.Done())

    if err := controller.Run(ctx, 2); err != nil {
        klog.Fatalf("Error running controller: %v", err)
    }
}
```

### 7.2 Owner References

Controllers should set owner references to enable garbage collection:

```go
import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// The child object will be deleted when the owner is deleted
ownerRef := metav1.OwnerReference{
    APIVersion:         "apps/v1",
    Kind:               "Deployment",
    Name:               parentDeploy.Name,
    UID:                parentDeploy.UID,
    Controller:         ptr.To(true),
    BlockOwnerDeletion: ptr.To(true),
}

childService := &corev1.Service{
    ObjectMeta: metav1.ObjectMeta{
        Name:            "my-service",
        Namespace:       parentDeploy.Namespace,
        OwnerReferences: []metav1.OwnerReference{ownerRef},
    },
    // ...
}
```

---

## 8. Controller-Runtime Library

### 8.1 Overview

`controller-runtime` (used by Kubebuilder and Operator SDK) provides a higher-level abstraction over client-go:

```
controller-runtime Architecture:
┌───────────────────────────────────────────────────────────┐
│                     Manager                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐    │
│  │ Cache     │  │ Client   │  │ Controller(s)         │    │
│  │ (shared   │  │ (cached  │  │ ┌──────────────────┐ │    │
│  │  informer │  │  reads,  │  │ │ Reconciler       │ │    │
│  │  cache)   │  │  direct  │  │ │ (your logic)     │ │    │
│  │          │  │  writes) │  │ └──────────────────┘ │    │
│  └──────────┘  └──────────┘  └──────────────────────┘    │
│  ┌──────────┐  ┌──────────┐                               │
│  │ Webhook  │  │ Health   │                               │
│  │ Server   │  │ Checks   │                               │
│  └──────────┘  └──────────┘                               │
└───────────────────────────────────────────────────────────┘
```

### 8.2 Simple Reconciler

```go
package controller

import (
    "context"
    "fmt"

    appsv1 "k8s.io/api/apps/v1"
    corev1 "k8s.io/api/core/v1"
    "k8s.io/apimachinery/pkg/api/errors"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    ctrl "sigs.k8s.io/controller-runtime"
    "sigs.k8s.io/controller-runtime/pkg/client"
    "sigs.k8s.io/controller-runtime/pkg/log"
)

// DeploymentReconciler ensures every Deployment has a corresponding Service
type DeploymentReconciler struct {
    client.Client
}

func (r *DeploymentReconciler) Reconcile(
    ctx context.Context,
    req ctrl.Request,
) (ctrl.Result, error) {
    logger := log.FromContext(ctx)

    // Fetch the Deployment
    var deploy appsv1.Deployment
    if err := r.Get(ctx, req.NamespacedName, &deploy); err != nil {
        if errors.IsNotFound(err) {
            return ctrl.Result{}, nil
        }
        return ctrl.Result{}, err
    }

    // Skip if the Deployment does not have our annotation
    if deploy.Annotations["auto-service"] != "true" {
        return ctrl.Result{}, nil
    }

    // Check if a Service already exists
    var svc corev1.Service
    svcName := client.ObjectKey{
        Namespace: deploy.Namespace,
        Name:      deploy.Name + "-auto",
    }

    err := r.Get(ctx, svcName, &svc)
    if errors.IsNotFound(err) {
        // Create the Service
        newSvc := &corev1.Service{
            ObjectMeta: metav1.ObjectMeta{
                Name:      svcName.Name,
                Namespace: svcName.Namespace,
            },
            Spec: corev1.ServiceSpec{
                Selector: deploy.Spec.Selector.MatchLabels,
                Ports: []corev1.ServicePort{
                    {Port: 80, Protocol: corev1.ProtocolTCP},
                },
            },
        }

        // Set owner reference for garbage collection
        if err := ctrl.SetControllerReference(&deploy, newSvc, r.Scheme()); err != nil {
            return ctrl.Result{}, err
        }

        if err := r.Create(ctx, newSvc); err != nil {
            return ctrl.Result{}, fmt.Errorf("create service: %w", err)
        }
        logger.Info("Created auto-service", "service", svcName.Name)
    } else if err != nil {
        return ctrl.Result{}, err
    }

    return ctrl.Result{}, nil
}

// SetupWithManager registers this reconciler with the manager
func (r *DeploymentReconciler) SetupWithManager(mgr ctrl.Manager) error {
    return ctrl.NewControllerManagedBy(mgr).
        For(&appsv1.Deployment{}).        // Watch Deployments
        Owns(&corev1.Service{}).           // Watch owned Services
        Complete(r)
}
```

### 8.3 Manager Setup

```go
package main

import (
    "os"

    ctrl "sigs.k8s.io/controller-runtime"
    "sigs.k8s.io/controller-runtime/pkg/healthz"
    "sigs.k8s.io/controller-runtime/pkg/log/zap"

    "github.com/example/k8s-controller/controller"
)

func main() {
    ctrl.SetLogger(zap.New(zap.UseDevMode(true)))

    mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
        LeaderElection:          true,
        LeaderElectionID:        "my-controller-leader",
        LeaderElectionNamespace: "kube-system",
        HealthProbeBindAddress:  ":8081",
        MetricsBindAddress:      ":8080",
    })
    if err != nil {
        os.Exit(1)
    }

    // Register health checks
    mgr.AddHealthzCheck("healthz", healthz.Ping)
    mgr.AddReadyzCheck("readyz", healthz.Ping)

    // Setup reconciler
    reconciler := &controller.DeploymentReconciler{
        Client: mgr.GetClient(),
    }
    if err := reconciler.SetupWithManager(mgr); err != nil {
        os.Exit(1)
    }

    // Start the manager (blocks)
    if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
        os.Exit(1)
    }
}
```

### 8.4 Predicates and Filtering

```go
import (
    "sigs.k8s.io/controller-runtime/pkg/event"
    "sigs.k8s.io/controller-runtime/pkg/predicate"
)

// Only reconcile on generation changes (spec changes, not status)
generationChangedPredicate := predicate.GenerationChangedPredicate{}

// Custom predicate: only reconcile labeled resources
labelPredicate := predicate.Funcs{
    CreateFunc: func(e event.CreateEvent) bool {
        return e.Object.GetLabels()["managed-by"] == "my-controller"
    },
    UpdateFunc: func(e event.UpdateEvent) bool {
        return e.ObjectNew.GetLabels()["managed-by"] == "my-controller"
    },
    DeleteFunc: func(e event.DeleteEvent) bool {
        return e.Object.GetLabels()["managed-by"] == "my-controller"
    },
}

ctrl.NewControllerManagedBy(mgr).
    For(&appsv1.Deployment{}).
    WithEventFilter(predicate.And(generationChangedPredicate, labelPredicate)).
    Complete(r)
```

---

## 9. Watching Resources and Handling Events

### 9.1 Cross-Resource Watching

Controllers often need to watch related resources. For example, watching both Deployments and the ConfigMaps they reference:

```go
import (
    "context"

    appsv1 "k8s.io/api/apps/v1"
    corev1 "k8s.io/api/core/v1"
    "k8s.io/apimachinery/pkg/types"
    ctrl "sigs.k8s.io/controller-runtime"
    "sigs.k8s.io/controller-runtime/pkg/client"
    "sigs.k8s.io/controller-runtime/pkg/handler"
    "sigs.k8s.io/controller-runtime/pkg/reconcile"
)

func (r *DeploymentReconciler) SetupWithManager(mgr ctrl.Manager) error {
    return ctrl.NewControllerManagedBy(mgr).
        For(&appsv1.Deployment{}).
        // When a ConfigMap changes, find and reconcile
        // all Deployments that reference it
        Watches(
            &corev1.ConfigMap{},
            handler.EnqueueRequestsFromMapFunc(
                func(ctx context.Context, obj client.Object) []reconcile.Request {
                    // Find deployments that reference this ConfigMap
                    var deployList appsv1.DeploymentList
                    if err := r.List(ctx, &deployList,
                        client.InNamespace(obj.GetNamespace()),
                    ); err != nil {
                        return nil
                    }

                    var requests []reconcile.Request
                    for _, deploy := range deployList.Items {
                        for _, vol := range deploy.Spec.Template.Spec.Volumes {
                            if vol.ConfigMap != nil &&
                                vol.ConfigMap.Name == obj.GetName() {
                                requests = append(requests, reconcile.Request{
                                    NamespacedName: types.NamespacedName{
                                        Name:      deploy.Name,
                                        Namespace: deploy.Namespace,
                                    },
                                })
                            }
                        }
                    }
                    return requests
                },
            ),
        ).
        Complete(r)
}
```

### 9.2 Finalizers

Finalizers let controllers perform cleanup before a resource is deleted:

```go
import (
    "context"

    "sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

const finalizerName = "example.com/cleanup"

func (r *MyReconciler) Reconcile(
    ctx context.Context,
    req ctrl.Request,
) (ctrl.Result, error) {
    var obj MyCustomResource
    if err := r.Get(ctx, req.NamespacedName, &obj); err != nil {
        return ctrl.Result{}, client.IgnoreNotFound(err)
    }

    // Check if the object is being deleted
    if !obj.DeletionTimestamp.IsZero() {
        if controllerutil.ContainsFinalizer(&obj, finalizerName) {
            // Perform cleanup logic
            if err := r.cleanupExternalResources(ctx, &obj); err != nil {
                return ctrl.Result{}, err
            }

            // Remove the finalizer to allow deletion
            controllerutil.RemoveFinalizer(&obj, finalizerName)
            if err := r.Update(ctx, &obj); err != nil {
                return ctrl.Result{}, err
            }
        }
        return ctrl.Result{}, nil
    }

    // Add finalizer if not present
    if !controllerutil.ContainsFinalizer(&obj, finalizerName) {
        controllerutil.AddFinalizer(&obj, finalizerName)
        if err := r.Update(ctx, &obj); err != nil {
            return ctrl.Result{}, err
        }
    }

    // Normal reconciliation logic here...
    return ctrl.Result{}, nil
}
```

### 9.3 Requeueing Strategies

```go
// Immediate requeue (process again as soon as possible)
return ctrl.Result{Requeue: true}, nil

// Requeue after a delay (check back later)
return ctrl.Result{RequeueAfter: 30 * time.Second}, nil

// No requeue (done processing)
return ctrl.Result{}, nil

// Error triggers automatic requeue with backoff
return ctrl.Result{}, fmt.Errorf("external API unavailable")
```

---

## 10. Testing Controllers

### 10.1 Unit Testing with Fake Client

```go
package controller_test

import (
    "context"
    "testing"

    appsv1 "k8s.io/api/apps/v1"
    corev1 "k8s.io/api/core/v1"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/runtime"
    "k8s.io/apimachinery/pkg/types"
    "k8s.io/utils/ptr"
    ctrl "sigs.k8s.io/controller-runtime"
    "sigs.k8s.io/controller-runtime/pkg/client/fake"

    "github.com/example/k8s-controller/controller"
)

func TestReconcile_CreatesService(t *testing.T) {
    // Setup: a Deployment with auto-service annotation
    deploy := &appsv1.Deployment{
        ObjectMeta: metav1.ObjectMeta{
            Name:      "web",
            Namespace: "default",
            Annotations: map[string]string{
                "auto-service": "true",
            },
        },
        Spec: appsv1.DeploymentSpec{
            Replicas: ptr.To(int32(1)),
            Selector: &metav1.LabelSelector{
                MatchLabels: map[string]string{"app": "web"},
            },
            Template: corev1.PodTemplateSpec{
                ObjectMeta: metav1.ObjectMeta{
                    Labels: map[string]string{"app": "web"},
                },
                Spec: corev1.PodSpec{
                    Containers: []corev1.Container{
                        {Name: "app", Image: "nginx:1.25"},
                    },
                },
            },
        },
    }

    scheme := runtime.NewScheme()
    _ = appsv1.AddToScheme(scheme)
    _ = corev1.AddToScheme(scheme)

    fakeClient := fake.NewClientBuilder().
        WithScheme(scheme).
        WithObjects(deploy).
        Build()

    reconciler := &controller.DeploymentReconciler{
        Client: fakeClient,
    }

    // Act
    result, err := reconciler.Reconcile(context.TODO(), ctrl.Request{
        NamespacedName: types.NamespacedName{
            Name:      "web",
            Namespace: "default",
        },
    })

    // Assert
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }
    if result.Requeue {
        t.Error("expected no requeue")
    }

    // Verify the Service was created
    var svc corev1.Service
    err = fakeClient.Get(context.TODO(), types.NamespacedName{
        Name:      "web-auto",
        Namespace: "default",
    }, &svc)
    if err != nil {
        t.Fatalf("expected service to exist: %v", err)
    }
    if svc.Spec.Selector["app"] != "web" {
        t.Errorf("expected selector app=web, got %v", svc.Spec.Selector)
    }
}
```

### 10.2 Integration Testing with envtest

`envtest` runs a real API server and etcd locally for integration tests:

```go
package controller_test

import (
    "context"
    "testing"
    "time"

    . "github.com/onsi/ginkgo/v2"
    . "github.com/onsi/gomega"

    appsv1 "k8s.io/api/apps/v1"
    corev1 "k8s.io/api/core/v1"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/types"
    "k8s.io/client-go/kubernetes/scheme"
    ctrl "sigs.k8s.io/controller-runtime"
    "sigs.k8s.io/controller-runtime/pkg/client"
    "sigs.k8s.io/controller-runtime/pkg/envtest"

    "github.com/example/k8s-controller/controller"
)

var (
    testEnv   *envtest.Environment
    k8sClient client.Client
    ctx       context.Context
    cancel    context.CancelFunc
)

func TestControllers(t *testing.T) {
    RegisterFailHandler(Fail)
    RunSpecs(t, "Controller Suite")
}

var _ = BeforeSuite(func() {
    ctx, cancel = context.WithCancel(context.TODO())

    testEnv = &envtest.Environment{}

    cfg, err := testEnv.Start()
    Expect(err).NotTo(HaveOccurred())

    k8sClient, err = client.New(cfg, client.Options{
        Scheme: scheme.Scheme,
    })
    Expect(err).NotTo(HaveOccurred())

    // Start the controller manager
    mgr, err := ctrl.NewManager(cfg, ctrl.Options{
        Scheme: scheme.Scheme,
    })
    Expect(err).NotTo(HaveOccurred())

    reconciler := &controller.DeploymentReconciler{
        Client: mgr.GetClient(),
    }
    err = reconciler.SetupWithManager(mgr)
    Expect(err).NotTo(HaveOccurred())

    go func() {
        err := mgr.Start(ctx)
        Expect(err).NotTo(HaveOccurred())
    }()
})

var _ = AfterSuite(func() {
    cancel()
    err := testEnv.Stop()
    Expect(err).NotTo(HaveOccurred())
})

var _ = Describe("DeploymentReconciler", func() {
    It("should create a Service for annotated Deployments", func() {
        deploy := &appsv1.Deployment{
            ObjectMeta: metav1.ObjectMeta{
                Name:      "integration-test",
                Namespace: "default",
                Annotations: map[string]string{
                    "auto-service": "true",
                },
            },
            Spec: appsv1.DeploymentSpec{
                Selector: &metav1.LabelSelector{
                    MatchLabels: map[string]string{"app": "test"},
                },
                Template: corev1.PodTemplateSpec{
                    ObjectMeta: metav1.ObjectMeta{
                        Labels: map[string]string{"app": "test"},
                    },
                    Spec: corev1.PodSpec{
                        Containers: []corev1.Container{
                            {Name: "app", Image: "nginx:1.25"},
                        },
                    },
                },
            },
        }

        err := k8sClient.Create(ctx, deploy)
        Expect(err).NotTo(HaveOccurred())

        // Wait for the controller to reconcile
        var svc corev1.Service
        Eventually(func() error {
            return k8sClient.Get(ctx, types.NamespacedName{
                Name:      "integration-test-auto",
                Namespace: "default",
            }, &svc)
        }, 10*time.Second, 250*time.Millisecond).Should(Succeed())

        Expect(svc.Spec.Selector).To(HaveKeyWithValue("app", "test"))
    })
})
```

### 10.3 Deploying Controllers

Package the controller as a container image and deploy it to the cluster:

```yaml
# controller-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: configmap-controller
  namespace: kube-system
  labels:
    app: configmap-controller
spec:
  replicas: 2       # HA with leader election
  selector:
    matchLabels:
      app: configmap-controller
  template:
    metadata:
      labels:
        app: configmap-controller
    spec:
      serviceAccountName: configmap-controller
      containers:
        - name: controller
          image: registry.example.com/configmap-controller:v1.0.0
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
          readinessProbe:
            httpGet:
              path: /readyz
              port: 8081
            initialDelaySeconds: 5
          resources:
            requests:
              cpu: 50m
              memory: 64Mi
            limits:
              cpu: 200m
              memory: 128Mi
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: configmap-controller
  namespace: kube-system
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: configmap-controller
rules:
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list", "watch", "update", "patch"]
  - apiGroups: ["coordination.k8s.io"]
    resources: ["leases"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: configmap-controller
subjects:
  - kind: ServiceAccount
    name: configmap-controller
    namespace: kube-system
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: configmap-controller
```

---

## Exercises

### Exercise 1: List Pods by Node

Write a Go program that uses client-go to list all pods in the cluster, grouped by the node they are running on. Display the pod name, namespace, and phase for each pod.

<details><summary>Show Answer</summary>

```go
package main

import (
    "context"
    "fmt"
    "os"
    "path/filepath"

    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/client-go/kubernetes"
    "k8s.io/client-go/tools/clientcmd"
)

func main() {
    kubeconfig := filepath.Join(os.Getenv("HOME"), ".kube", "config")
    config, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
    if err != nil {
        panic(err)
    }

    clientset, err := kubernetes.NewForConfig(config)
    if err != nil {
        panic(err)
    }

    pods, err := clientset.CoreV1().Pods("").List(
        context.TODO(), metav1.ListOptions{},
    )
    if err != nil {
        panic(err)
    }

    // Group by node
    byNode := make(map[string][]string)
    for _, pod := range pods.Items {
        node := pod.Spec.NodeName
        if node == "" {
            node = "<unscheduled>"
        }
        entry := fmt.Sprintf("  %s/%s (Phase: %s)",
            pod.Namespace, pod.Name, pod.Status.Phase)
        byNode[node] = append(byNode[node], entry)
    }

    for node, entries := range byNode {
        fmt.Printf("Node: %s (%d pods)\n", node, len(entries))
        for _, e := range entries {
            fmt.Println(e)
        }
        fmt.Println()
    }
}
```

</details>

### Exercise 2: Dynamic Client for Custom Resources

Write a Go program using the dynamic client to list all CustomResourceDefinitions in the cluster and print their group, version, kind, and scope.

<details><summary>Show Answer</summary>

```go
package main

import (
    "context"
    "fmt"
    "os"
    "path/filepath"

    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
    "k8s.io/apimachinery/pkg/runtime/schema"
    "k8s.io/client-go/dynamic"
    "k8s.io/client-go/tools/clientcmd"
)

func main() {
    kubeconfig := filepath.Join(os.Getenv("HOME"), ".kube", "config")
    config, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
    if err != nil {
        panic(err)
    }

    dynClient, err := dynamic.NewForConfig(config)
    if err != nil {
        panic(err)
    }

    crdGVR := schema.GroupVersionResource{
        Group:    "apiextensions.k8s.io",
        Version:  "v1",
        Resource: "customresourcedefinitions",
    }

    crds, err := dynClient.Resource(crdGVR).List(
        context.TODO(), metav1.ListOptions{},
    )
    if err != nil {
        panic(err)
    }

    for _, crd := range crds.Items {
        group, _, _ := unstructured.NestedString(crd.Object, "spec", "group")
        scope, _, _ := unstructured.NestedString(crd.Object, "spec", "scope")

        names, _, _ := unstructured.NestedMap(crd.Object, "spec", "names")
        kind := names["kind"]

        versions, _, _ := unstructured.NestedSlice(crd.Object, "spec", "versions")
        var versionNames []string
        for _, v := range versions {
            if vm, ok := v.(map[string]interface{}); ok {
                if name, ok := vm["name"].(string); ok {
                    versionNames = append(versionNames, name)
                }
            }
        }

        fmt.Printf("CRD: %s\n  Group: %s\n  Versions: %v\n  Kind: %v\n  Scope: %s\n\n",
            crd.GetName(), group, versionNames, kind, scope)
    }
}
```

</details>

### Exercise 3: Informer with Custom Indexer

Build a program that uses a SharedInformerFactory with a custom indexer to quickly look up Services by their type (ClusterIP, NodePort, LoadBalancer). After the cache syncs, query the index and print a count per service type.

<details><summary>Show Answer</summary>

```go
package main

import (
    "fmt"
    "os"
    "os/signal"
    "path/filepath"
    "syscall"
    "time"

    corev1 "k8s.io/api/core/v1"
    "k8s.io/client-go/informers"
    "k8s.io/client-go/kubernetes"
    "k8s.io/client-go/tools/cache"
    "k8s.io/client-go/tools/clientcmd"
)

const byTypeIndex = "byServiceType"

func main() {
    kubeconfig := filepath.Join(os.Getenv("HOME"), ".kube", "config")
    config, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
    if err != nil {
        panic(err)
    }

    clientset, err := kubernetes.NewForConfig(config)
    if err != nil {
        panic(err)
    }

    factory := informers.NewSharedInformerFactory(clientset, 30*time.Second)
    svcInformer := factory.Core().V1().Services().Informer()

    // Add custom indexer
    svcInformer.AddIndexers(cache.Indexers{
        byTypeIndex: func(obj interface{}) ([]string, error) {
            svc := obj.(*corev1.Service)
            return []string{string(svc.Spec.Type)}, nil
        },
    })

    stopCh := make(chan struct{})
    defer close(stopCh)

    factory.Start(stopCh)
    factory.WaitForCacheSync(stopCh)

    indexer := svcInformer.GetIndexer()

    for _, svcType := range []string{"ClusterIP", "NodePort", "LoadBalancer"} {
        items, err := indexer.ByIndex(byTypeIndex, svcType)
        if err != nil {
            fmt.Printf("Error querying index for %s: %v\n", svcType, err)
            continue
        }
        fmt.Printf("%s services: %d\n", svcType, len(items))
        for _, item := range items {
            svc := item.(*corev1.Service)
            fmt.Printf("  %s/%s\n", svc.Namespace, svc.Name)
        }
    }

    // Keep running to watch for changes
    fmt.Println("\nWatching for changes (Ctrl+C to stop)...")
    sig := make(chan os.Signal, 1)
    signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
    <-sig
}
```

</details>

### Exercise 4: Controller with Requeue

Write a controller-runtime reconciler for Namespaces that checks whether a `ResourceQuota` named `default-quota` exists in each namespace. If it does not, create one with 10 pods and 4Gi memory limit. If creation fails, requeue after 30 seconds.

<details><summary>Show Answer</summary>

```go
package controller

import (
    "context"
    "time"

    corev1 "k8s.io/api/core/v1"
    "k8s.io/apimachinery/pkg/api/errors"
    "k8s.io/apimachinery/pkg/api/resource"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/types"
    ctrl "sigs.k8s.io/controller-runtime"
    "sigs.k8s.io/controller-runtime/pkg/client"
    "sigs.k8s.io/controller-runtime/pkg/log"
)

type QuotaReconciler struct {
    client.Client
}

func (r *QuotaReconciler) Reconcile(
    ctx context.Context,
    req ctrl.Request,
) (ctrl.Result, error) {
    logger := log.FromContext(ctx)

    // Skip system namespaces
    systemNS := map[string]bool{
        "kube-system": true, "kube-public": true,
        "kube-node-lease": true, "default": true,
    }
    if systemNS[req.Name] {
        return ctrl.Result{}, nil
    }

    // Check if the namespace still exists
    var ns corev1.Namespace
    if err := r.Get(ctx, req.NamespacedName, &ns); err != nil {
        return ctrl.Result{}, client.IgnoreNotFound(err)
    }

    // Check if ResourceQuota exists
    var quota corev1.ResourceQuota
    err := r.Get(ctx, types.NamespacedName{
        Name:      "default-quota",
        Namespace: req.Name,
    }, &quota)

    if errors.IsNotFound(err) {
        newQuota := &corev1.ResourceQuota{
            ObjectMeta: metav1.ObjectMeta{
                Name:      "default-quota",
                Namespace: req.Name,
            },
            Spec: corev1.ResourceQuotaSpec{
                Hard: corev1.ResourceList{
                    corev1.ResourcePods:           resource.MustParse("10"),
                    corev1.ResourceLimitsMemory:    resource.MustParse("4Gi"),
                },
            },
        }

        if err := r.Create(ctx, newQuota); err != nil {
            logger.Error(err, "Failed to create quota, requeuing",
                "namespace", req.Name)
            return ctrl.Result{RequeueAfter: 30 * time.Second}, nil
        }
        logger.Info("Created default-quota", "namespace", req.Name)
    } else if err != nil {
        return ctrl.Result{}, err
    }

    return ctrl.Result{}, nil
}

func (r *QuotaReconciler) SetupWithManager(mgr ctrl.Manager) error {
    return ctrl.NewControllerManagedBy(mgr).
        For(&corev1.Namespace{}).
        Owns(&corev1.ResourceQuota{}).
        Complete(r)
}
```

</details>

### Exercise 5: Testing with Fake Client

Write a unit test that verifies the QuotaReconciler from Exercise 4 creates a ResourceQuota when one does not exist, and does not create a duplicate when one already exists.

<details><summary>Show Answer</summary>

```go
package controller_test

import (
    "context"
    "testing"

    corev1 "k8s.io/api/core/v1"
    "k8s.io/apimachinery/pkg/api/resource"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/runtime"
    "k8s.io/apimachinery/pkg/types"
    ctrl "sigs.k8s.io/controller-runtime"
    "sigs.k8s.io/controller-runtime/pkg/client/fake"

    "github.com/example/k8s-controller/controller"
)

func TestQuotaReconciler_CreatesQuota(t *testing.T) {
    ns := &corev1.Namespace{
        ObjectMeta: metav1.ObjectMeta{Name: "team-alpha"},
    }

    scheme := runtime.NewScheme()
    _ = corev1.AddToScheme(scheme)

    fakeClient := fake.NewClientBuilder().
        WithScheme(scheme).
        WithObjects(ns).
        Build()

    reconciler := &controller.QuotaReconciler{Client: fakeClient}

    result, err := reconciler.Reconcile(context.TODO(), ctrl.Request{
        NamespacedName: types.NamespacedName{Name: "team-alpha"},
    })
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }
    if result.Requeue || result.RequeueAfter != 0 {
        t.Error("expected no requeue")
    }

    var quota corev1.ResourceQuota
    err = fakeClient.Get(context.TODO(), types.NamespacedName{
        Name:      "default-quota",
        Namespace: "team-alpha",
    }, &quota)
    if err != nil {
        t.Fatalf("expected quota to exist: %v", err)
    }

    pods := quota.Spec.Hard[corev1.ResourcePods]
    if pods.Cmp(resource.MustParse("10")) != 0 {
        t.Errorf("expected 10 pods limit, got %s", pods.String())
    }
}

func TestQuotaReconciler_SkipsExisting(t *testing.T) {
    ns := &corev1.Namespace{
        ObjectMeta: metav1.ObjectMeta{Name: "team-beta"},
    }
    existingQuota := &corev1.ResourceQuota{
        ObjectMeta: metav1.ObjectMeta{
            Name:      "default-quota",
            Namespace: "team-beta",
        },
        Spec: corev1.ResourceQuotaSpec{
            Hard: corev1.ResourceList{
                corev1.ResourcePods:        resource.MustParse("20"),
                corev1.ResourceLimitsMemory: resource.MustParse("8Gi"),
            },
        },
    }

    scheme := runtime.NewScheme()
    _ = corev1.AddToScheme(scheme)

    fakeClient := fake.NewClientBuilder().
        WithScheme(scheme).
        WithObjects(ns, existingQuota).
        Build()

    reconciler := &controller.QuotaReconciler{Client: fakeClient}

    _, err := reconciler.Reconcile(context.TODO(), ctrl.Request{
        NamespacedName: types.NamespacedName{Name: "team-beta"},
    })
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }

    // Verify quota was not modified
    var quota corev1.ResourceQuota
    _ = fakeClient.Get(context.TODO(), types.NamespacedName{
        Name:      "default-quota",
        Namespace: "team-beta",
    }, &quota)

    pods := quota.Spec.Hard[corev1.ResourcePods]
    if pods.Cmp(resource.MustParse("20")) != 0 {
        t.Errorf("expected original 20 pods limit, got %s", pods.String())
    }
}

func TestQuotaReconciler_SkipsSystemNamespace(t *testing.T) {
    ns := &corev1.Namespace{
        ObjectMeta: metav1.ObjectMeta{Name: "kube-system"},
    }

    scheme := runtime.NewScheme()
    _ = corev1.AddToScheme(scheme)

    fakeClient := fake.NewClientBuilder().
        WithScheme(scheme).
        WithObjects(ns).
        Build()

    reconciler := &controller.QuotaReconciler{Client: fakeClient}

    _, err := reconciler.Reconcile(context.TODO(), ctrl.Request{
        NamespacedName: types.NamespacedName{Name: "kube-system"},
    })
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }

    // Verify no quota was created
    var quota corev1.ResourceQuota
    err = fakeClient.Get(context.TODO(), types.NamespacedName{
        Name:      "default-quota",
        Namespace: "kube-system",
    }, &quota)
    if err == nil {
        t.Error("expected no quota in kube-system")
    }
}
```

</details>

---

**Previous**: [15. Multi-Cluster](./15_Multi_Cluster.md) | **Next**: [17. Production Operations](./17_Production_Operations.md)

# 12. Admission Controllers

**Previous**: [Operators](./11_Operators.md) | **Next**: [Autoscaling](./13_Autoscaling.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the Kubernetes admission controller pipeline and how requests flow through it
2. Configure and deploy validating and mutating webhook servers
3. Implement policy enforcement using OPA Gatekeeper constraints and constraint templates
4. Use Kyverno as an alternative policy engine with declarative YAML policies
5. Test and debug admission policies in development and production environments

---

Every request to the Kubernetes API server passes through a chain of admission controllers before it is persisted to etcd. This chain is one of the most powerful extension points in Kubernetes -- it lets you enforce security policies, inject sidecars, set defaults, validate configurations, and prevent misconfigurations before they reach the cluster. This lesson covers both the built-in admission controllers and the dynamic admission control system that lets you plug in your own logic.

Before the webhook setup, read [**Theory & Principles**](#theory--principles) — where admission sits in the request pipeline (after authn/authz, before persistence), the Mutating-then-Validating ordering that lets you inject defaults safely, why webhooks must be fast, idempotent, and fail-closed-or-open by design, and how policy engines (OPA, Kyverno) layer rules on top of the same webhook plumbing.

## Table of Contents

- [Theory & Principles](#theory--principles)
- [1. The Admission Controller Pipeline](#1-the-admission-controller-pipeline)
- [2. Built-in Admission Controllers](#2-built-in-admission-controllers)
- [3. Dynamic Admission Control](#3-dynamic-admission-control)
- [4. Validating Webhooks](#4-validating-webhooks)
- [5. Mutating Webhooks](#5-mutating-webhooks)
- [6. Webhook Configuration](#6-webhook-configuration)
- [7. OPA Gatekeeper](#7-opa-gatekeeper)
- [8. Kyverno](#8-kyverno)
- [9. Testing Admission Policies](#9-testing-admission-policies)
- [10. Admission Controller Performance](#10-admission-controller-performance)
- [Exercises](#exercises)

---

## Theory & Principles

Admission control is the third gate of the API request pipeline (lesson 06 §A). After authentication says *who* and authorization says *if*, admission says *should this exact object, as written, be persisted?* This is where policies live: "no privileged pods," "every container must have a CPU limit," "images must come from our registry," "namespaces must have a cost-center label." Anything you can express as "look at the object, decide allow/deny/mutate" goes here. This section explains the place of admission in the pipeline, the mutating-then-validating two-pass design, the operational constraints on webhooks (latency, fail-policy, idempotence), and how OPA Gatekeeper and Kyverno are simply policy engines that plug into this same machinery.

### A. Where Admission Sits and Why That Matters

Recall the four-gate pipeline from lesson 06: **authn → authz → admission → schema validation → persist to etcd**. Each stage answers a specific question:

- **Authn**: who are you?
- **Authz** (RBAC etc.): are you allowed to do this verb on this resource?
- **Admission**: should this *specific* request go through?
- **Schema/CEL validation**: does the object match the registered shape?
- **Persist**: write to etcd.

Admission is *the* extension point for everything that depends on the *content* of the object, not just on permissions. RBAC can grant "create pods" but cannot say "only non-privileged pods." Schema validation can require a field but cannot enforce "this field's value matches our registry's hostname." Both gaps are admission's job.

Admission also runs **before persistence**, so a rejected request never makes it into etcd, never produces audit-log noise about partial state, never confuses a controller. This is why "policy as admission" is fundamentally different from "policy as a controller that deletes bad objects after the fact": the latter creates a window where bad state exists and can be observed by other controllers; the former makes bad state literally impossible to create.

The trade-off: admission is on the hot path. Every API request — every kubectl apply, every controller create — pays the admission cost. So webhooks have hard performance requirements (§C).

### B. Built-in Admission Plugins and the Two-Pass Webhook Design

Admission has two kinds of plugins: **built-in** (compiled into the API server) and **dynamic** (webhooks you register). Built-in plugins handle the universal cases:

- `LimitRanger` injects default CPU/memory limits if a namespace has a `LimitRange`.
- `ResourceQuota` rejects requests that would exceed a namespace's `ResourceQuota`.
- `ServiceAccount` injects the namespace's default ServiceAccount and its token volume.
- `NamespaceLifecycle` rejects creates in namespaces that don't exist or are terminating.
- `PodSecurity` (lesson 06) enforces Pod Security Standards.
- `MutatingAdmissionWebhook` and `ValidatingAdmissionWebhook` are the entry points to dynamic admission.

The two webhook types run in distinct phases:

**Phase 1: Mutating webhooks.** Each registered `MutatingWebhookConfiguration` matched by the request is called with the object. Each may return a JSON patch that the API server applies. Mutating webhooks chain — webhook A's output is webhook B's input — so order can matter (the API server processes them in a non-deterministic order, with `reinvocationPolicy: IfNeeded` to re-run after later mutations). Typical uses: inject sidecars (Istio, Linkerd), add labels/annotations, set defaults the API author forgot.

**Phase 2: Validating webhooks.** After all mutations finish, validating webhooks see the *final* object and return allow or deny (no patches). Multiple validating webhooks all run; if any denies, the request is rejected. Typical uses: enforce policies (no privileged pods, image registries, label requirements).

This ordering is deliberate: validation runs on the final state, so a mutator can add defaults that a validator then verifies, and the user only sees one error message about the final form. Doing them in the other order would let a validator approve a partial object that a later mutator breaks.

A subtlety: mutating webhooks must be careful about idempotence and conflicts. If two webhooks try to inject the same annotation with different values, the API server's reinvocation logic resolves it but the user gets unpredictable behavior. The operational rule is: each mutating webhook should own a non-overlapping concern.

### C. Operational Constraints: Latency, Fail-Policy, Side Effects

Webhooks live on the API hot path. The constraints are nontrivial:

- **Latency**: every API request waits for every matched webhook. The API server has a default 10-second timeout (configurable down). A slow webhook makes every kubectl apply slow. **Recommendation: webhooks must respond within 100ms, p99.**
- **Fail policy** is `Fail` (default) or `Ignore`. With `Fail`, if the webhook is unreachable, the request is rejected — strict but means a webhook outage breaks deployments. With `Ignore`, the webhook is bypassed on error — graceful but lets policy violations through during outages. Most security webhooks should use `Fail` plus high availability (multiple replicas, PDB, tested rollouts).
- **Idempotence**: webhooks may be retried, especially mutating ones with `reinvocationPolicy: IfNeeded`. A webhook that sets `metadata.labels.foo = bar` is idempotent. A webhook that *appends* to a list ("add this sidecar to the containers list") needs to first check if it's already there — otherwise duplicate sidecars on retry.
- **Side effects**: webhooks should not have external side effects (don't post to Slack from a webhook). The API server retries on errors and can call your webhook many times for one logical request. Use `sideEffects: None` (or `NoneOnDryRun`) so the API server knows it can be called freely.
- **Scope**: configure `rules` precisely so the webhook is only invoked for what it cares about. A webhook that watches `pods` and runs on every CRD apply just adds latency.

These constraints are why production webhooks are usually written with frameworks (kubebuilder for Go, kubewarden for WebAssembly) that handle TLS, request parsing, and the AdmissionReview schema, leaving you with just the policy logic.

### D. Policy Engines: OPA Gatekeeper and Kyverno

Writing webhooks in Go for every policy gets tedious. **Policy engines** are pre-built validating (and sometimes mutating) webhooks that read declarative policies as Kubernetes resources. Two dominant choices:

**OPA Gatekeeper**. Built on the Open Policy Agent runtime; policies are written in **Rego**, a declarative logic language. Two CRDs:

- `ConstraintTemplate`: defines a parameterized policy in Rego. Like a function definition.
- `Constraint` (a custom kind generated from the template): an instance of the template with parameters. Like a function call.

Example: a `RequiredLabels` template + a constraint that says "all namespaces must have labels `cost-center` and `team`." Gatekeeper compiles these into webhook decisions. Strong audit features (continuous re-evaluation against existing objects, not just admission), and the same Rego policies are reusable outside Kubernetes (Envoy authz, Terraform, custom apps).

**Kyverno**. Native Kubernetes — policies are CRDs (`ClusterPolicy`, `Policy`) written in YAML, no DSL to learn. Three rule types: validate (allow/deny), mutate (set defaults, add labels), generate (create child resources from a template).

```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-labels
spec:
  validationFailureAction: enforce
  rules:
    - name: check-team-label
      match:
        any:
          - resources:
              kinds: [Namespace]
      validate:
        message: "Namespace must have a 'team' label"
        pattern:
          metadata:
            labels:
              team: "?*"
```

Kyverno wins on accessibility — your security team can write policies without learning Rego. OPA wins on power — Rego can express policies that are awkward in Kyverno's declarative syntax.

Both engines plug into the same Mutating/ValidatingWebhookConfiguration machinery from §B; you don't choose between webhooks and policy engines, you choose what runs *as* the webhook.

### From Theory to the Code Below

The lesson now applies these abstractions:

- **Section 1 (The Admission Controller Pipeline)** is §A — the place of admission with the full request flow.
- **Section 2 (Built-in Admission Controllers)** is the §B baseline of plugins that are always on.
- **Section 3 (Dynamic Admission Control)** introduces the webhook concept.
- **Sections 4–5 (Validating Webhooks, Mutating Webhooks)** are §B's two phases with concrete YAML and Go code.
- **Section 6 (Webhook Configuration)** is the `MutatingWebhookConfiguration`/`ValidatingWebhookConfiguration` spec — rules, fail policy, side effects, namespace selectors.
- **Section 7 (OPA Gatekeeper)** is §D's Rego-based policy engine.
- **Section 8 (Kyverno)** is §D's YAML-based policy engine.
- **Section 9 (Testing Admission Policies)** is the dry-run + audit pattern that lets you roll out enforce mode safely.
- **Section 10 (Performance)** is §C made operational — measuring webhook latency, scaling replicas.

Once you see admission as "the third gate, with webhooks as the extension API and policy engines as preconfigured webhooks," every cluster security/compliance feature reduces to "what policy lives in admission?"

---

## 1. The Admission Controller Pipeline

### 1.1 Request Flow

When a client (kubectl, controller, CI pipeline) sends a request to the API server, it passes through several stages:

```
Client Request
    │
    ▼
┌──────────────────┐
│  Authentication   │  Who are you?
└────────┬─────────┘
         │
    ▼
┌──────────────────┐
│  Authorization    │  Are you allowed?
│  (RBAC)          │
└────────┬─────────┘
         │
    ▼
┌──────────────────────────────────────────────┐
│           Admission Controllers               │
│                                               │
│  ┌─────────────────┐  ┌────────────────────┐  │
│  │    Mutating      │  │    Validating      │  │
│  │    Admission     │──▶    Admission       │  │
│  │    Webhooks      │  │    Webhooks        │  │
│  └─────────────────┘  └────────────────────┘  │
│                                               │
│  (Object schema validation happens between)   │
└────────┬─────────────────────────────────────┘
         │
    ▼
┌──────────────────┐
│  Persist to etcd  │
└──────────────────┘
```

### 1.2 Mutating vs Validating

| Phase | Purpose | Can Modify Object? | Runs |
|---|---|---|---|
| Mutating | Set defaults, inject sidecars, add labels | Yes | First |
| Object Schema Validation | Check against OpenAPI schema | No | Between phases |
| Validating | Enforce policies, reject bad configs | No | Second |

Mutating webhooks run first because validators need to see the final form of the object. Mutating webhooks can also be called multiple times if one mutator's changes trigger re-evaluation.

---

## 2. Built-in Admission Controllers

### 2.1 Commonly Used Controllers

Kubernetes ships with ~30 compiled-in admission controllers. Key ones:

| Controller | Purpose |
|---|---|
| `NamespaceLifecycle` | Prevents operations in non-existent or terminating namespaces |
| `LimitRanger` | Enforces LimitRange constraints on pods |
| `ServiceAccount` | Injects the default service account and token |
| `DefaultStorageClass` | Sets the default StorageClass on PVCs with no class |
| `ResourceQuota` | Enforces namespace resource quotas |
| `PodSecurity` | Enforces Pod Security Standards (replaced PodSecurityPolicy) |
| `MutatingAdmissionWebhook` | Calls external mutating webhooks |
| `ValidatingAdmissionWebhook` | Calls external validating webhooks |
| `ValidatingAdmissionPolicy` | In-cluster CEL-based validation (v1.28+ stable) |

### 2.2 Checking Enabled Controllers

```bash
# Check which admission plugins are enabled
kubectl exec -n kube-system kube-apiserver-<node> -- \
  kube-apiserver --help 2>&1 | grep enable-admission-plugins

# Or check the API server manifest
cat /etc/kubernetes/manifests/kube-apiserver.yaml | grep admission

# Default enabled list (Kubernetes 1.29+):
# CertificateApproval, CertificateSigning, CertificateSubjectRestriction,
# DefaultIngressClass, DefaultStorageClass, DefaultTolerationSeconds,
# LimitRanger, MutatingAdmissionWebhook, NamespaceLifecycle,
# PersistentVolumeClaimResize, PodSecurity, Priority,
# ResourceQuota, RuntimeClass, ServiceAccount, StorageObjectInUseProtection,
# TaintNodesByCondition, ValidatingAdmissionPolicy, ValidatingAdmissionWebhook
```

### 2.3 ValidatingAdmissionPolicy (CEL-based)

Kubernetes 1.28+ provides in-cluster validation using Common Expression Language (CEL), without needing an external webhook:

```yaml
# Define a policy
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
metadata:
  name: require-resource-limits
spec:
  failurePolicy: Fail
  matchConstraints:
    resourceRules:
    - apiGroups: [""]
      apiVersions: ["v1"]
      operations: ["CREATE", "UPDATE"]
      resources: ["pods"]
  validations:
  - expression: >-
      object.spec.containers.all(c,
        has(c.resources) &&
        has(c.resources.limits) &&
        has(c.resources.limits.cpu) &&
        has(c.resources.limits.memory)
      )
    message: "All containers must have CPU and memory limits set"
    reason: Invalid
---
# Bind the policy to a namespace
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicyBinding
metadata:
  name: require-resource-limits-binding
spec:
  policyName: require-resource-limits
  validationActions:
  - Deny
  matchResources:
    namespaceSelector:
      matchLabels:
        environment: production
```

---

## 3. Dynamic Admission Control

### 3.1 Architecture

Dynamic admission control allows you to register external HTTPS servers (webhooks) that the API server calls during the admission phase.

```
                  API Server
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
  ┌───────────────┐   ┌────────────────┐
  │   Mutating    │   │  Validating    │
  │   Webhook     │   │  Webhook       │
  │   Config      │   │  Config        │
  └───────┬───────┘   └────────┬───────┘
          │                     │
          ▼                     ▼
  ┌───────────────┐   ┌────────────────┐
  │  Webhook Pod  │   │  Webhook Pod   │
  │  (HTTPS)      │   │  (HTTPS)       │
  │  /mutate      │   │  /validate     │
  └───────────────┘   └────────────────┘
```

### 3.2 AdmissionReview API

Webhooks communicate using the `AdmissionReview` object:

```json
{
  "apiVersion": "admission.k8s.io/v1",
  "kind": "AdmissionReview",
  "request": {
    "uid": "705ab4f5-6393-11e8-b7cc-42010a800002",
    "kind": {"group": "", "version": "v1", "kind": "Pod"},
    "resource": {"group": "", "version": "v1", "resource": "pods"},
    "namespace": "default",
    "operation": "CREATE",
    "userInfo": {
      "username": "system:serviceaccount:default:my-sa",
      "groups": ["system:serviceaccounts"]
    },
    "object": {
      "metadata": {"name": "my-pod", "namespace": "default"},
      "spec": {
        "containers": [{"name": "app", "image": "nginx:latest"}]
      }
    },
    "oldObject": null,
    "dryRun": false
  }
}
```

---

## 4. Validating Webhooks

### 4.1 Webhook Server Implementation (Go)

```go
package main

import (
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "strings"

    admissionv1 "k8s.io/api/admission/v1"
    corev1 "k8s.io/api/core/v1"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/runtime"
    "k8s.io/apimachinery/pkg/runtime/serializer"
)

var (
    runtimeScheme = runtime.NewScheme()
    codecs        = serializer.NewCodecFactory(runtimeScheme)
    deserializer  = codecs.UniversalDeserializer()
)

// validatePod checks that no container uses the :latest tag
func validatePod(pod *corev1.Pod) (bool, string) {
    for _, container := range pod.Spec.Containers {
        if strings.HasSuffix(container.Image, ":latest") || !strings.Contains(container.Image, ":") {
            return false, fmt.Sprintf(
                "container %q uses image %q: images must have an explicit tag, not :latest",
                container.Name, container.Image,
            )
        }
    }
    for _, container := range pod.Spec.InitContainers {
        if strings.HasSuffix(container.Image, ":latest") || !strings.Contains(container.Image, ":") {
            return false, fmt.Sprintf(
                "init container %q uses image %q: images must have an explicit tag, not :latest",
                container.Name, container.Image,
            )
        }
    }
    return true, ""
}

func handleValidate(w http.ResponseWriter, r *http.Request) {
    body, err := io.ReadAll(r.Body)
    if err != nil {
        http.Error(w, "could not read body", http.StatusBadRequest)
        return
    }

    var admissionReview admissionv1.AdmissionReview
    if _, _, err := deserializer.Decode(body, nil, &admissionReview); err != nil {
        http.Error(w, "could not decode body", http.StatusBadRequest)
        return
    }

    var pod corev1.Pod
    if err := json.Unmarshal(admissionReview.Request.Object.Raw, &pod); err != nil {
        http.Error(w, "could not unmarshal pod", http.StatusBadRequest)
        return
    }

    allowed, reason := validatePod(&pod)

    response := &admissionv1.AdmissionReview{
        TypeMeta: metav1.TypeMeta{
            APIVersion: "admission.k8s.io/v1",
            Kind:       "AdmissionReview",
        },
        Response: &admissionv1.AdmissionResponse{
            UID:     admissionReview.Request.UID,
            Allowed: allowed,
        },
    }
    if !allowed {
        response.Response.Result = &metav1.Status{
            Message: reason,
            Code:    http.StatusForbidden,
        }
    }

    respBytes, _ := json.Marshal(response)
    w.Header().Set("Content-Type", "application/json")
    w.Write(respBytes)
}

func main() {
    http.HandleFunc("/validate", handleValidate)
    fmt.Println("Starting webhook server on :8443")
    err := http.ListenAndServeTLS(":8443", "/certs/tls.crt", "/certs/tls.key", nil)
    if err != nil {
        panic(err)
    }
}
```

### 4.2 Deploying the Webhook Server

```yaml
# Deployment for the webhook server
apiVersion: apps/v1
kind: Deployment
metadata:
  name: image-policy-webhook
  namespace: webhook-system
spec:
  replicas: 2
  selector:
    matchLabels:
      app: image-policy-webhook
  template:
    metadata:
      labels:
        app: image-policy-webhook
    spec:
      containers:
      - name: webhook
        image: example.com/image-policy-webhook:v1.0.0
        ports:
        - containerPort: 8443
          protocol: TCP
        volumeMounts:
        - name: tls-certs
          mountPath: /certs
          readOnly: true
        resources:
          requests:
            cpu: 50m
            memory: 64Mi
          limits:
            cpu: 200m
            memory: 128Mi
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8443
            scheme: HTTPS
          initialDelaySeconds: 5
        readinessProbe:
          httpGet:
            path: /readyz
            port: 8443
            scheme: HTTPS
          initialDelaySeconds: 3
      volumes:
      - name: tls-certs
        secret:
          secretName: image-policy-webhook-tls
---
apiVersion: v1
kind: Service
metadata:
  name: image-policy-webhook
  namespace: webhook-system
spec:
  selector:
    app: image-policy-webhook
  ports:
  - port: 443
    targetPort: 8443
    protocol: TCP
```

### 4.3 TLS Certificate Setup

```bash
# Generate CA and server certificate using OpenSSL
# The SAN must match the webhook service DNS name
SERVICE_NAME=image-policy-webhook
NAMESPACE=webhook-system

# Generate CA key and certificate
openssl genrsa -out ca.key 2048
openssl req -new -x509 -days 365 -key ca.key -subj "/CN=Webhook CA" -out ca.crt

# Generate server key and CSR
openssl genrsa -out server.key 2048
openssl req -new -key server.key \
  -subj "/CN=${SERVICE_NAME}.${NAMESPACE}.svc" \
  -out server.csr

# Create SAN config
cat > san.cnf <<EOF
[req]
req_extensions = v3_req
[v3_req]
subjectAltName = DNS:${SERVICE_NAME}.${NAMESPACE}.svc, DNS:${SERVICE_NAME}.${NAMESPACE}.svc.cluster.local
EOF

# Sign the server certificate
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key \
  -CAcreateserial -out server.crt -days 365 \
  -extfile san.cnf -extensions v3_req

# Create the TLS secret
kubectl create secret tls image-policy-webhook-tls \
  --cert=server.crt --key=server.key \
  -n webhook-system

# The CA bundle (base64-encoded) goes in the webhook configuration
CA_BUNDLE=$(cat ca.crt | base64 | tr -d '\n')
echo $CA_BUNDLE
```

Alternatively, use cert-manager for automatic certificate management:

```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: image-policy-webhook-cert
  namespace: webhook-system
spec:
  secretName: image-policy-webhook-tls
  dnsNames:
  - image-policy-webhook.webhook-system.svc
  - image-policy-webhook.webhook-system.svc.cluster.local
  issuerRef:
    name: selfsigned-issuer
    kind: ClusterIssuer
  duration: 8760h  # 1 year
  renewBefore: 720h  # 30 days
```

---

## 5. Mutating Webhooks

### 5.1 Mutation via JSON Patch

Mutating webhooks return a JSON Patch (RFC 6902) in the response to modify the incoming object:

```go
// handleMutate injects a sidecar container into every pod
func handleMutate(w http.ResponseWriter, r *http.Request) {
    body, _ := io.ReadAll(r.Body)
    var admissionReview admissionv1.AdmissionReview
    deserializer.Decode(body, nil, &admissionReview)

    var pod corev1.Pod
    json.Unmarshal(admissionReview.Request.Object.Raw, &pod)

    // Skip if annotation says no injection
    if pod.Annotations["sidecar-injector/inject"] == "false" {
        sendAllowed(w, admissionReview.Request.UID)
        return
    }

    // Build JSON Patch to add a sidecar container
    patches := []map[string]interface{}{}

    sidecar := map[string]interface{}{
        "name":  "log-collector",
        "image": "fluent/fluent-bit:2.2",
        "resources": map[string]interface{}{
            "requests": map[string]string{"cpu": "25m", "memory": "32Mi"},
            "limits":   map[string]string{"cpu": "100m", "memory": "64Mi"},
        },
        "volumeMounts": []map[string]string{
            {"name": "shared-logs", "mountPath": "/var/log/app"},
        },
    }

    patches = append(patches, map[string]interface{}{
        "op":    "add",
        "path":  "/spec/containers/-",
        "value": sidecar,
    })

    // Add shared volume if no volumes exist
    if len(pod.Spec.Volumes) == 0 {
        patches = append(patches, map[string]interface{}{
            "op":   "add",
            "path": "/spec/volumes",
            "value": []map[string]interface{}{
                {"name": "shared-logs", "emptyDir": map[string]interface{}{}},
            },
        })
    } else {
        patches = append(patches, map[string]interface{}{
            "op":   "add",
            "path": "/spec/volumes/-",
            "value": map[string]interface{}{
                "name": "shared-logs", "emptyDir": map[string]interface{}{},
            },
        })
    }

    // Add a label to track injection
    if pod.Labels == nil {
        patches = append(patches, map[string]interface{}{
            "op":    "add",
            "path":  "/metadata/labels",
            "value": map[string]string{"sidecar-injected": "true"},
        })
    } else {
        patches = append(patches, map[string]interface{}{
            "op":    "add",
            "path":  "/metadata/labels/sidecar-injected",
            "value": "true",
        })
    }

    patchBytes, _ := json.Marshal(patches)
    patchType := admissionv1.PatchTypeJSONPatch

    response := &admissionv1.AdmissionReview{
        TypeMeta: metav1.TypeMeta{
            APIVersion: "admission.k8s.io/v1",
            Kind:       "AdmissionReview",
        },
        Response: &admissionv1.AdmissionResponse{
            UID:       admissionReview.Request.UID,
            Allowed:   true,
            Patch:     patchBytes,
            PatchType: &patchType,
        },
    }

    respBytes, _ := json.Marshal(response)
    w.Header().Set("Content-Type", "application/json")
    w.Write(respBytes)
}
```

### 5.2 Common Mutation Use Cases

| Use Case | What Gets Mutated |
|---|---|
| Sidecar injection (Istio, Linkerd) | Add containers and volumes |
| Default resource limits | Set requests/limits if missing |
| Image registry rewriting | Replace `nginx` with `registry.internal/nginx` |
| Label/annotation injection | Add org-standard labels |
| Node affinity injection | Add tolerations or nodeSelector based on namespace |
| Environment variable injection | Add common env vars (region, cluster name) |

---

## 6. Webhook Configuration

### 6.1 ValidatingWebhookConfiguration

```yaml
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: image-policy-validator
  annotations:
    cert-manager.io/inject-ca-from: webhook-system/image-policy-webhook-cert
webhooks:
- name: validate.image-policy.example.com
  admissionReviewVersions: ["v1"]
  sideEffects: None
  timeoutSeconds: 5
  failurePolicy: Fail      # Fail or Ignore
  matchPolicy: Equivalent   # Exact or Equivalent
  clientConfig:
    service:
      name: image-policy-webhook
      namespace: webhook-system
      path: /validate
      port: 443
    # caBundle: <base64-encoded CA cert>  # Use if not using cert-manager annotation
  rules:
  - operations: ["CREATE", "UPDATE"]
    apiGroups: [""]
    apiVersions: ["v1"]
    resources: ["pods"]
    scope: "Namespaced"
  namespaceSelector:
    matchExpressions:
    - key: kubernetes.io/metadata.name
      operator: NotIn
      values: ["kube-system", "kube-public", "webhook-system"]
  objectSelector:
    matchExpressions:
    - key: skip-validation
      operator: DoesNotExist
```

### 6.2 MutatingWebhookConfiguration

```yaml
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingWebhookConfiguration
metadata:
  name: sidecar-injector
webhooks:
- name: mutate.sidecar-injector.example.com
  admissionReviewVersions: ["v1"]
  sideEffects: None
  reinvocationPolicy: IfNeeded  # Re-invoke if another mutator changes the object
  timeoutSeconds: 10
  failurePolicy: Ignore          # Do not block pod creation if webhook is down
  clientConfig:
    service:
      name: sidecar-injector
      namespace: webhook-system
      path: /mutate
      port: 443
    caBundle: "${CA_BUNDLE}"
  rules:
  - operations: ["CREATE"]
    apiGroups: [""]
    apiVersions: ["v1"]
    resources: ["pods"]
  namespaceSelector:
    matchLabels:
      sidecar-injection: enabled
```

### 6.3 Configuration Fields Reference

| Field | Description | Options |
|---|---|---|
| `failurePolicy` | What to do if the webhook is unreachable | `Fail` (reject) or `Ignore` (allow) |
| `sideEffects` | Whether the webhook has side effects | `None`, `NoneOnDryRun`, `Unknown` |
| `timeoutSeconds` | Maximum time to wait for webhook response | 1-30 (default: 10) |
| `reinvocationPolicy` | Whether to re-invoke after other mutations | `Never` or `IfNeeded` |
| `matchPolicy` | How to match API versions | `Exact` or `Equivalent` |
| `namespaceSelector` | Label selector to filter namespaces | Standard label selector |
| `objectSelector` | Label selector to filter objects | Standard label selector |

---

## 7. OPA Gatekeeper

### 7.1 What is Gatekeeper?

OPA (Open Policy Agent) Gatekeeper is a validating admission webhook that lets you define policies using Rego, OPA's purpose-built policy language. It provides a Kubernetes-native way to express and enforce policies through CRDs.

### 7.2 Installation

```bash
# Install Gatekeeper using Helm
helm repo add gatekeeper https://open-policy-agent.github.io/gatekeeper/charts
helm install gatekeeper gatekeeper/gatekeeper \
  --namespace gatekeeper-system \
  --create-namespace \
  --set replicas=3 \
  --set audit.replicas=1 \
  --set audit.logLevel=INFO

# Verify installation
kubectl get pods -n gatekeeper-system
kubectl get crd | grep gatekeeper
```

### 7.3 Constraint Templates

A ConstraintTemplate defines the Rego policy logic and the parameters it accepts:

```yaml
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: k8srequiredlabels
spec:
  crd:
    spec:
      names:
        kind: K8sRequiredLabels
      validation:
        openAPIV3Schema:
          type: object
          properties:
            labels:
              type: array
              description: "List of required label keys"
              items:
                type: string
  targets:
  - target: admission.k8s.gatekeeper.sh
    rego: |
      package k8srequiredlabels

      violation[{"msg": msg}] {
        provided := {label | input.review.object.metadata.labels[label]}
        required := {label | label := input.parameters.labels[_]}
        missing := required - provided
        count(missing) > 0
        msg := sprintf("Missing required labels: %v", [missing])
      }
```

### 7.4 Constraints

A Constraint instantiates a ConstraintTemplate with specific parameters:

```yaml
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sRequiredLabels
metadata:
  name: require-team-label
spec:
  enforcementAction: deny  # deny, dryrun, or warn
  match:
    kinds:
    - apiGroups: [""]
      kinds: ["Namespace"]
    - apiGroups: ["apps"]
      kinds: ["Deployment", "StatefulSet", "DaemonSet"]
    excludedNamespaces:
    - kube-system
    - gatekeeper-system
  parameters:
    labels:
    - "team"
    - "cost-center"
```

### 7.5 Advanced Rego Policies

**Deny privileged containers:**

```yaml
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: k8sdenyprivileged
spec:
  crd:
    spec:
      names:
        kind: K8sDenyPrivileged
  targets:
  - target: admission.k8s.gatekeeper.sh
    rego: |
      package k8sdenyprivileged

      violation[{"msg": msg}] {
        container := input.review.object.spec.containers[_]
        container.securityContext.privileged == true
        msg := sprintf("Privileged container is not allowed: %v", [container.name])
      }

      violation[{"msg": msg}] {
        container := input.review.object.spec.initContainers[_]
        container.securityContext.privileged == true
        msg := sprintf("Privileged init container is not allowed: %v", [container.name])
      }
```

**Restrict allowed registries:**

```yaml
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: k8sallowedregistries
spec:
  crd:
    spec:
      names:
        kind: K8sAllowedRegistries
      validation:
        openAPIV3Schema:
          type: object
          properties:
            registries:
              type: array
              items:
                type: string
  targets:
  - target: admission.k8s.gatekeeper.sh
    rego: |
      package k8sallowedregistries

      violation[{"msg": msg}] {
        container := input.review.object.spec.containers[_]
        not registry_allowed(container.image)
        msg := sprintf("Container %v uses image %v from a disallowed registry. Allowed: %v",
          [container.name, container.image, input.parameters.registries])
      }

      violation[{"msg": msg}] {
        container := input.review.object.spec.initContainers[_]
        not registry_allowed(container.image)
        msg := sprintf("Init container %v uses image %v from a disallowed registry. Allowed: %v",
          [container.name, container.image, input.parameters.registries])
      }

      registry_allowed(image) {
        registry := input.parameters.registries[_]
        startswith(image, registry)
      }
---
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sAllowedRegistries
metadata:
  name: allowed-registries
spec:
  enforcementAction: deny
  match:
    kinds:
    - apiGroups: [""]
      kinds: ["Pod"]
    - apiGroups: ["apps"]
      kinds: ["Deployment", "StatefulSet", "DaemonSet"]
  parameters:
    registries:
    - "gcr.io/my-project/"
    - "docker.io/library/"
    - "ghcr.io/my-org/"
```

### 7.6 Gatekeeper Audit

Gatekeeper runs periodic audits to find existing resources that violate constraints:

```bash
# Check constraint violations
kubectl get k8srequiredlabels require-team-label -o yaml

# The status section shows violations
# status:
#   totalViolations: 5
#   violations:
#   - enforcementAction: deny
#     kind: Deployment
#     name: frontend
#     namespace: default
#     message: 'Missing required labels: {"team"}'
```

---

## 8. Kyverno

### 8.1 Kyverno vs Gatekeeper

| Feature | Gatekeeper | Kyverno |
|---|---|---|
| Policy language | Rego (learning curve) | YAML (Kubernetes-native) |
| Mutation support | Limited (assign/modify) | Full JSON Patch and strategic merge |
| Generation | No | Yes (create resources from policies) |
| Image verification | Via external data | Built-in (cosign, Notary) |
| Policy reports | Via constraint status | Dedicated PolicyReport CRD |
| CLI testing | `opa test` + `gator` | `kyverno test` |

### 8.2 Installation

```bash
# Install Kyverno using Helm
helm repo add kyverno https://kyverno.github.io/kyverno/
helm install kyverno kyverno/kyverno \
  --namespace kyverno \
  --create-namespace \
  --set replicaCount=3

# Verify
kubectl get pods -n kyverno
```

### 8.3 Kyverno Validation Policy

```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-resource-limits
  annotations:
    policies.kyverno.io/title: Require Resource Limits
    policies.kyverno.io/category: Best Practices
    policies.kyverno.io/severity: medium
spec:
  validationFailureAction: Enforce  # Enforce or Audit
  background: true  # Scan existing resources
  rules:
  - name: check-container-limits
    match:
      any:
      - resources:
          kinds:
          - Pod
    exclude:
      any:
      - resources:
          namespaces:
          - kube-system
    validate:
      message: "CPU and memory limits are required for container {{request.object.spec.containers[*].name}}"
      pattern:
        spec:
          containers:
          - resources:
              limits:
                memory: "?*"
                cpu: "?*"
```

### 8.4 Kyverno Mutation Policy

```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: add-default-labels
spec:
  rules:
  - name: add-labels
    match:
      any:
      - resources:
          kinds:
          - Deployment
          - StatefulSet
    mutate:
      patchStrategicMerge:
        metadata:
          labels:
            +(managed-by): "platform-team"
            +(environment): "{{request.namespace}}"
        spec:
          template:
            metadata:
              labels:
                +(managed-by): "platform-team"
```

### 8.5 Kyverno Generate Policy

Kyverno can create resources automatically when triggering events occur:

```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: generate-network-policy
spec:
  rules:
  - name: default-deny-ingress
    match:
      any:
      - resources:
          kinds:
          - Namespace
    exclude:
      any:
      - resources:
          names:
          - kube-system
          - kube-public
    generate:
      synchronize: true  # Keep in sync if policy changes
      apiVersion: networking.k8s.io/v1
      kind: NetworkPolicy
      name: default-deny-ingress
      namespace: "{{request.object.metadata.name}}"
      data:
        spec:
          podSelector: {}
          policyTypes:
          - Ingress
```

---

## 9. Testing Admission Policies

### 9.1 Testing Gatekeeper Policies with gator

```bash
# Install gator CLI
go install github.com/open-policy-agent/gatekeeper/v3/cmd/gator@latest

# Create a test suite
cat > tests/require-labels-test.yaml <<EOF
apiVersion: test.gatekeeper.sh/v1alpha1
kind: Suite
metadata:
  name: require-labels-test
tests:
- name: deployment-without-labels
  template: template.yaml
  constraint: constraint.yaml
  cases:
  - name: should-reject-missing-labels
    object: testdata/deployment-no-labels.yaml
    assertions:
    - violations: 1
      message: "Missing required labels"
  - name: should-allow-with-labels
    object: testdata/deployment-with-labels.yaml
    assertions:
    - violations: 0
EOF

# Run the tests
gator verify tests/
```

### 9.2 Testing Kyverno Policies with CLI

```bash
# Install Kyverno CLI
brew install kyverno  # macOS
# or
kubectl krew install kyverno

# Test a policy against a resource
kyverno apply policy.yaml --resource deployment.yaml

# Run a test suite
cat > kyverno-test.yaml <<EOF
name: require-resource-limits-test
policies:
- require-resource-limits.yaml
resources:
- testdata/pod-with-limits.yaml
- testdata/pod-without-limits.yaml
results:
- policy: require-resource-limits
  rule: check-container-limits
  resource: pod-with-limits
  kind: Pod
  result: pass
- policy: require-resource-limits
  rule: check-container-limits
  resource: pod-without-limits
  kind: Pod
  result: fail
EOF

kyverno test .
```

### 9.3 Integration Testing Webhooks

```go
// Test webhook with a mock HTTP server
func TestValidateHandler(t *testing.T) {
    pod := corev1.Pod{
        ObjectMeta: metav1.ObjectMeta{Name: "test-pod"},
        Spec: corev1.PodSpec{
            Containers: []corev1.Container{
                {Name: "app", Image: "nginx:latest"},
            },
        },
    }
    podBytes, _ := json.Marshal(pod)

    review := admissionv1.AdmissionReview{
        TypeMeta: metav1.TypeMeta{APIVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"},
        Request: &admissionv1.AdmissionRequest{
            UID: "test-uid",
            Object: runtime.RawExtension{
                Raw: podBytes,
            },
        },
    }
    reviewBytes, _ := json.Marshal(review)

    req := httptest.NewRequest("POST", "/validate", bytes.NewReader(reviewBytes))
    req.Header.Set("Content-Type", "application/json")
    rec := httptest.NewRecorder()

    handleValidate(rec, req)

    var response admissionv1.AdmissionReview
    json.Unmarshal(rec.Body.Bytes(), &response)

    if response.Response.Allowed {
        t.Error("expected request to be denied for :latest tag")
    }
}
```

### 9.4 Dry-Run Testing

```bash
# Test admission with dry-run (does not persist, but still runs webhooks)
kubectl apply --dry-run=server -f pod.yaml

# Test Gatekeeper constraint in dryrun mode first
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sRequiredLabels
metadata:
  name: require-team-label
spec:
  enforcementAction: dryrun  # Will not block, only audit
```

---

## 10. Admission Controller Performance

### 10.1 Performance Considerations

Admission webhooks add latency to every API request that matches their rules. Poor performance can slow down the entire cluster.

| Factor | Impact | Mitigation |
|---|---|---|
| Webhook latency | Added to every matched request | Keep webhook logic simple, cache data |
| Network hops | Webhook in different node adds RTT | Co-locate or use in-cluster webhooks |
| TLS handshake | Per-connection overhead | Enable HTTP/2, connection pooling |
| Failure mode | `Fail` policy blocks all matched requests | Use `Ignore` for non-critical webhooks |
| Match scope | Broad rules process more requests | Narrow `rules`, use namespace/object selectors |

### 10.2 Performance Optimization

```yaml
# Narrow the scope as much as possible
webhooks:
- name: validate.example.com
  rules:
  - operations: ["CREATE"]      # Only CREATE, not UPDATE
    apiGroups: ["apps"]          # Only apps group, not "*"
    apiVersions: ["v1"]          # Specific version
    resources: ["deployments"]   # Specific resource, not "*"
    scope: "Namespaced"          # Skip cluster-scoped resources
  namespaceSelector:
    matchLabels:
      policy-enforcement: enabled   # Only labeled namespaces
  timeoutSeconds: 3                 # Fail fast
  failurePolicy: Ignore            # Do not block on webhook failure
```

### 10.3 Monitoring Webhook Performance

```bash
# Check API server metrics for webhook latency
kubectl get --raw /metrics | grep apiserver_admission_webhook_admission_duration_seconds

# Prometheus query for webhook latency (p99)
# histogram_quantile(0.99,
#   rate(apiserver_admission_webhook_admission_duration_seconds_bucket{
#     name="validate.image-policy.example.com"
#   }[5m])
# )

# Check webhook rejection rate
# sum(rate(apiserver_admission_webhook_rejection_count{
#   name="validate.image-policy.example.com"
# }[5m]))
```

### 10.4 High Availability

```yaml
# Run multiple webhook replicas with pod anti-affinity
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webhook-server
spec:
  replicas: 3
  template:
    spec:
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: kubernetes.io/hostname
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: webhook-server
      containers:
      - name: webhook
        image: example.com/webhook:v1
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 256Mi
```

---

## Exercises

### Exercise 1: Build a Validating Webhook

Write a validating webhook server in Go that rejects any Pod creation if: (a) the pod does not have a `team` label, (b) any container runs as root (securityContext.runAsNonRoot is false or not set), (c) any container requests more than 4 CPU cores. Include the ValidatingWebhookConfiguration YAML.

<details>
<summary>Show Answer</summary>

```go
func validatePodSecurity(pod *corev1.Pod) (bool, string) {
    // Check team label
    if _, ok := pod.Labels["team"]; !ok {
        return false, "pod must have a 'team' label"
    }

    for _, c := range pod.Spec.Containers {
        // Check runAsNonRoot
        if c.SecurityContext == nil || c.SecurityContext.RunAsNonRoot == nil || !*c.SecurityContext.RunAsNonRoot {
            return false, fmt.Sprintf("container %q must set securityContext.runAsNonRoot=true", c.Name)
        }

        // Check CPU limit
        if cpuLimit, ok := c.Resources.Limits[corev1.ResourceCPU]; ok {
            if cpuLimit.Value() > 4 {
                return false, fmt.Sprintf("container %q requests %v CPU, max allowed is 4", c.Name, cpuLimit.String())
            }
        }
    }
    return true, ""
}
```

```yaml
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: pod-security-validator
webhooks:
- name: validate.pod-security.example.com
  admissionReviewVersions: ["v1"]
  sideEffects: None
  timeoutSeconds: 5
  failurePolicy: Fail
  clientConfig:
    service:
      name: pod-security-webhook
      namespace: webhook-system
      path: /validate
      port: 443
  rules:
  - operations: ["CREATE"]
    apiGroups: [""]
    apiVersions: ["v1"]
    resources: ["pods"]
  namespaceSelector:
    matchExpressions:
    - key: kubernetes.io/metadata.name
      operator: NotIn
      values: ["kube-system", "webhook-system"]
```

</details>

### Exercise 2: Write a Mutating Webhook

Write a mutating webhook that: (a) adds a `prometheus.io/scrape: "true"` annotation to all Pods, (b) sets `automountServiceAccountToken: false` if not explicitly set, (c) adds a `toleration` for the key `dedicated=monitoring:NoSchedule` to pods in the `monitoring` namespace. Return the JSON Patch array.

<details>
<summary>Show Answer</summary>

```go
func buildMutationPatches(pod *corev1.Pod, namespace string) []map[string]interface{} {
    patches := []map[string]interface{}{}

    // (a) Add Prometheus scrape annotation
    if pod.Annotations == nil {
        patches = append(patches, map[string]interface{}{
            "op":    "add",
            "path":  "/metadata/annotations",
            "value": map[string]string{"prometheus.io/scrape": "true"},
        })
    } else if _, ok := pod.Annotations["prometheus.io/scrape"]; !ok {
        patches = append(patches, map[string]interface{}{
            "op":    "add",
            "path":  "/metadata/annotations/prometheus.io~1scrape",
            "value": "true",
        })
    }

    // (b) Set automountServiceAccountToken to false if not set
    if pod.Spec.AutomountServiceAccountToken == nil {
        falseVal := false
        _ = falseVal
        patches = append(patches, map[string]interface{}{
            "op":    "add",
            "path":  "/spec/automountServiceAccountToken",
            "value": false,
        })
    }

    // (c) Add monitoring toleration for monitoring namespace
    if namespace == "monitoring" {
        toleration := map[string]interface{}{
            "key":      "dedicated",
            "operator": "Equal",
            "value":    "monitoring",
            "effect":   "NoSchedule",
        }
        if len(pod.Spec.Tolerations) == 0 {
            patches = append(patches, map[string]interface{}{
                "op":    "add",
                "path":  "/spec/tolerations",
                "value": []map[string]interface{}{toleration},
            })
        } else {
            patches = append(patches, map[string]interface{}{
                "op":    "add",
                "path":  "/spec/tolerations/-",
                "value": toleration,
            })
        }
    }

    return patches
}
```

The resulting JSON Patch for a pod in the `monitoring` namespace with no annotations or tolerations:

```json
[
  {"op": "add", "path": "/metadata/annotations", "value": {"prometheus.io/scrape": "true"}},
  {"op": "add", "path": "/spec/automountServiceAccountToken", "value": false},
  {"op": "add", "path": "/spec/tolerations", "value": [{"key": "dedicated", "operator": "Equal", "value": "monitoring", "effect": "NoSchedule"}]}
]
```

</details>

### Exercise 3: OPA Gatekeeper Policy

Write a Gatekeeper ConstraintTemplate and Constraint that enforces the following: (a) all Deployments must have at least 2 replicas in production namespaces, (b) the constraint should only apply to namespaces labeled `env: production`, (c) Deployments in the `kube-system` namespace are exempt.

<details>
<summary>Show Answer</summary>

```yaml
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: k8sminreplicas
spec:
  crd:
    spec:
      names:
        kind: K8sMinReplicas
      validation:
        openAPIV3Schema:
          type: object
          properties:
            minReplicas:
              type: integer
              description: "Minimum number of replicas required"
  targets:
  - target: admission.k8s.gatekeeper.sh
    rego: |
      package k8sminreplicas

      violation[{"msg": msg}] {
        input.review.object.kind == "Deployment"
        replicas := object.get(input.review.object.spec, "replicas", 1)
        replicas < input.parameters.minReplicas
        msg := sprintf(
          "Deployment %v has %v replicas, minimum required is %v",
          [input.review.object.metadata.name, replicas, input.parameters.minReplicas]
        )
      }
---
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sMinReplicas
metadata:
  name: production-min-replicas
spec:
  enforcementAction: deny
  match:
    kinds:
    - apiGroups: ["apps"]
      kinds: ["Deployment"]
    namespaceSelector:
      matchLabels:
        env: production
    excludedNamespaces:
    - kube-system
  parameters:
    minReplicas: 2
```

Test with:

```bash
# This should be rejected (1 replica in production namespace)
kubectl -n production-ns create deployment test --image=nginx --replicas=1

# This should be allowed (2+ replicas)
kubectl -n production-ns create deployment test --image=nginx --replicas=3
```

</details>

### Exercise 4: Kyverno Policy Suite

Write three Kyverno ClusterPolicies: (a) a validation policy that prevents Pods from using `hostNetwork: true`, (b) a mutation policy that adds `readOnlyRootFilesystem: true` to all containers that do not explicitly set it, (c) a generate policy that creates a ResourceQuota (10 CPU, 20Gi memory) in every new namespace.

<details>
<summary>Show Answer</summary>

```yaml
# (a) Deny hostNetwork
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: deny-host-network
spec:
  validationFailureAction: Enforce
  background: true
  rules:
  - name: deny-host-network
    match:
      any:
      - resources:
          kinds:
          - Pod
    exclude:
      any:
      - resources:
          namespaces:
          - kube-system
    validate:
      message: "Using hostNetwork is not allowed. Pod {{request.object.metadata.name}} sets hostNetwork: true"
      pattern:
        spec:
          =(hostNetwork): false
---
# (b) Mutate readOnlyRootFilesystem
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: set-readonly-root-fs
spec:
  rules:
  - name: set-readonly-root
    match:
      any:
      - resources:
          kinds:
          - Pod
    mutate:
      foreach:
      - list: "request.object.spec.containers"
        patchStrategicMerge:
          spec:
            containers:
            - name: "{{element.name}}"
              securityContext:
                +(readOnlyRootFilesystem): true
---
# (c) Generate ResourceQuota
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: generate-resource-quota
spec:
  rules:
  - name: create-default-quota
    match:
      any:
      - resources:
          kinds:
          - Namespace
    exclude:
      any:
      - resources:
          names:
          - kube-system
          - kube-public
          - kube-node-lease
          - kyverno
    generate:
      synchronize: true
      apiVersion: v1
      kind: ResourceQuota
      name: default-quota
      namespace: "{{request.object.metadata.name}}"
      data:
        spec:
          hard:
            requests.cpu: "10"
            requests.memory: 20Gi
            limits.cpu: "20"
            limits.memory: 40Gi
```

</details>

### Exercise 5: Webhook Failure Modes

Your production cluster has a validating webhook that is experiencing intermittent timeouts. Describe: (a) the difference between `failurePolicy: Fail` and `failurePolicy: Ignore` and when to use each, (b) how to configure the webhook so critical namespaces (`kube-system`, `monitoring`) are never blocked, (c) how to monitor webhook latency using Prometheus metrics, (d) write the updated WebhookConfiguration with proper resilience settings.

<details>
<summary>Show Answer</summary>

**(a) Failure Policy:**

- `Fail`: If the webhook is unreachable or times out, the API request is rejected. Use for security-critical policies where you prefer to block operations rather than allow potentially unsafe changes (e.g., image provenance verification).
- `Ignore`: If the webhook is unreachable or times out, the API request proceeds without webhook validation. Use for non-critical policies where availability is more important than enforcement (e.g., label recommendations, cost tracking).

**(b) Namespace exclusions** are configured using `namespaceSelector`:

**(c) Prometheus metrics** to monitor:

```promql
# Webhook call latency (p99)
histogram_quantile(0.99,
  rate(apiserver_admission_webhook_admission_duration_seconds_bucket{
    name="validate.example.com",
    type="validating"
  }[5m])
)

# Webhook rejection rate
sum(rate(apiserver_admission_webhook_rejection_count{name="validate.example.com"}[5m]))

# Webhook failure/timeout rate
sum(rate(apiserver_admission_webhook_fail_open_count{name="validate.example.com"}[5m]))
```

**(d) Resilient configuration:**

```yaml
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: resilient-validator
webhooks:
- name: validate.example.com
  admissionReviewVersions: ["v1", "v1beta1"]
  sideEffects: None
  timeoutSeconds: 3
  failurePolicy: Ignore
  clientConfig:
    service:
      name: validator
      namespace: webhook-system
      path: /validate
      port: 443
  rules:
  - operations: ["CREATE", "UPDATE"]
    apiGroups: ["apps"]
    apiVersions: ["v1"]
    resources: ["deployments"]
    scope: "Namespaced"
  namespaceSelector:
    matchExpressions:
    - key: kubernetes.io/metadata.name
      operator: NotIn
      values:
      - kube-system
      - kube-public
      - kube-node-lease
      - monitoring
      - webhook-system
    - key: webhook-validation
      operator: In
      values: ["enabled"]
  objectSelector:
    matchExpressions:
    - key: skip-webhook
      operator: DoesNotExist
```

</details>

---

**Previous**: [Operators](./11_Operators.md) | **Next**: [Autoscaling](./13_Autoscaling.md)

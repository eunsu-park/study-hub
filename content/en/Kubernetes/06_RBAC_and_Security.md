# 06. RBAC and Security

**Previous**: [Configuration and Secrets](./05_Configuration_and_Secrets.md) | **Next**: [Ingress and Gateway API](./07_Ingress_and_Gateway_API.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain Kubernetes authentication methods and how the API server verifies identity
2. Configure Role-Based Access Control (RBAC) with Roles, ClusterRoles, and Bindings
3. Apply Pod Security Standards and Pod Security Admission to enforce workload isolation
4. Implement OPA/Gatekeeper policies for custom admission control
5. Harden pods with security contexts, seccomp profiles, and network policies

---

Kubernetes clusters are multi-tenant platforms that run workloads from different teams, services, and trust levels. Without proper security, a single compromised pod can escalate privileges, exfiltrate secrets, and move laterally across the entire cluster. This lesson covers the full Kubernetes security stack -- from authenticating users and authorizing API requests with RBAC, to hardening workloads with Pod Security Standards, and enforcing organizational policies with OPA/Gatekeeper.

> **Defense in Depth:** Kubernetes security is not a single feature but a layered approach. Authentication verifies identity, authorization controls access, admission control enforces policies, and runtime security restricts what pods can do. Each layer compensates for failures in other layers.

Before the manifests, read [**Theory & Principles**](#theory--principles) — the four-stage gate every API request passes (authn → authz → admission → schema validation), why RBAC is purely additive, the Role/ClusterRole + Binding combinatorial model, and why Pod Security Standards live in admission rather than RBAC.

## Table of Contents

- [Theory & Principles](#theory--principles)
- [1. Authentication Methods](#1-authentication-methods)
  - [1.1 X.509 Client Certificates](#11-x509-client-certificates)
  - [1.2 Bearer Tokens](#12-bearer-tokens)
  - [1.3 OpenID Connect (OIDC)](#13-openid-connect-oidc)
  - [1.4 Service Account Tokens](#14-service-account-tokens)
- [2. Authorization Modes](#2-authorization-modes)
  - [2.1 RBAC (Role-Based Access Control)](#21-rbac-role-based-access-control)
  - [2.2 ABAC (Attribute-Based Access Control)](#22-abac-attribute-based-access-control)
  - [2.3 Webhook Authorization](#23-webhook-authorization)
- [3. RBAC Deep Dive](#3-rbac-deep-dive)
  - [3.1 Roles and ClusterRoles](#31-roles-and-clusterroles)
  - [3.2 RoleBindings and ClusterRoleBindings](#32-rolebindings-and-clusterrolebindings)
  - [3.3 Aggregated ClusterRoles](#33-aggregated-clusterroles)
  - [3.4 Common RBAC Patterns](#34-common-rbac-patterns)
- [4. Service Accounts](#4-service-accounts)
  - [4.1 Service Account Basics](#41-service-account-basics)
  - [4.2 Bound Service Account Tokens](#42-bound-service-account-tokens)
  - [4.3 Disabling Auto-Mount](#43-disabling-auto-mount)
- [5. Pod Security Standards](#5-pod-security-standards)
  - [5.1 The Three Profiles](#51-the-three-profiles)
  - [5.2 Pod Security Admission](#52-pod-security-admission)
  - [5.3 Namespace-Level Enforcement](#53-namespace-level-enforcement)
- [6. Security Contexts](#6-security-contexts)
  - [6.1 Pod-Level Security Context](#61-pod-level-security-context)
  - [6.2 Container-Level Security Context](#62-container-level-security-context)
- [7. Seccomp and AppArmor](#7-seccomp-and-apparmor)
  - [7.1 Seccomp Profiles](#71-seccomp-profiles)
  - [7.2 AppArmor Profiles](#72-apparmor-profiles)
- [8. OPA/Gatekeeper Policies](#8-opagatekeeper-policies)
  - [8.1 Architecture](#81-architecture)
  - [8.2 ConstraintTemplates and Constraints](#82-constrainttemplates-and-constraints)
- [9. Network Policies for Security](#9-network-policies-for-security)
  - [9.1 Default Deny](#91-default-deny)
  - [9.2 Allow Specific Traffic](#92-allow-specific-traffic)
- [Exercises](#exercises)

---

## Theory & Principles

Kubernetes security is best understood as a **pipeline of independent gates**: a request must pass authentication, then authorization, then admission, then schema validation, and only then is it persisted to etcd. Each gate has a different job, a different failure mode, and a different extension point. Confusing them ("RBAC denied my pod") is the most common debugging mistake. This section explains the four-gate pipeline, the additive model of RBAC, and why deeper security controls (Pod Security Standards, OPA, network policies) live in admission and runtime rather than in the authorization layer.

### A. The Four-Stage Request Pipeline

Every API request — `kubectl apply`, controller call, sidecar GET, dashboard click — passes through the same chain inside the API server:

1. **Authentication (authn): "Who are you?"** Validates client credentials (X.509 cert, bearer token, OIDC ID token, ServiceAccount JWT). Output is a `user.Info` struct (username, groups, extra). If no authenticator recognizes the credential, the request is anonymous (or rejected if anonymous is disabled).
2. **Authorization (authz): "Are you allowed to do this?"** Given `(user, verb, resource, namespace, name)`, ask each enabled authorizer (RBAC, ABAC, Webhook, Node) — they return Allow, Deny, or NoOpinion. The first explicit answer wins; **if everyone says NoOpinion, the request is denied** (default deny).
3. **Admission control: "Should this request, as written, actually happen?"** Mutating webhooks can rewrite the object (inject sidecars, set defaults). Validating webhooks can reject it (block privileged pods, enforce labels). Built-in admission controllers handle quotas, defaults, namespace existence checks, and Pod Security Standards.
4. **Schema validation and persistence.** OpenAPI/CEL schema check on the object, then write to etcd.

Only after all four gates pass does the object exist. A failure at any gate means the request never lands. Knowing which gate is rejecting you ("forbidden" → authz; "denied by webhook" → admission; "invalid schema" → step 4) makes debugging tractable.

This pipeline is also why "I gave my user all permissions but they still cannot create privileged pods" is not a bug — RBAC said yes, but Pod Security Admission said no.

### B. RBAC as a Pure Allow List

RBAC has four object kinds, organized by scope:

| Kind | Defines | Scope |
|------|---------|-------|
| `Role` | a set of `(verb, resource)` permissions | namespaced |
| `ClusterRole` | same, but cluster-wide | cluster |
| `RoleBinding` | grants a `Role` (or `ClusterRole`) to subjects in one namespace | namespaced |
| `ClusterRoleBinding` | grants a `ClusterRole` to subjects cluster-wide | cluster |

Subjects are users, groups, or ServiceAccounts.

The model has three crucial properties:

- **Additive only.** RBAC has no "deny" rule. You build up permissions by binding roles. To take permissions away, you remove a binding — there is no way to write "user X cannot read secrets" once they have a role granting it. This makes the system simple to reason about (no rule-precedence puzzles) but means least-privilege requires careful role construction, not patching.
- **Verb-resource granularity.** Permissions are at the `(verb, resource)` level: `get pods`, `list deployments`, `create configmaps/myconfig` (resource name optional). Subresources are separate (`pods/exec`, `pods/log`). Wildcards exist (`verbs: ["*"]`, `resources: ["*"]`) but should be avoided in production roles.
- **No data filtering.** RBAC controls *whether* you can list secrets in a namespace, not *which* secrets you see. If you can list secrets, you can list all of them. Per-row filtering requires admission webhooks or external policy engines.

The Role/ClusterRole + Binding split lets the same `ClusterRole` ("view") be reused: bind it to user A in namespace `dev`, user B in namespace `prod`, and group `oncall` cluster-wide — three bindings, one role definition.

### C. Identity for Workloads: ServiceAccounts and Token Projection

Human users authenticate with certs or OIDC. Pods authenticate with **ServiceAccount tokens** — JWTs signed by the API server, automatically mounted into the Pod at `/var/run/secrets/kubernetes.io/serviceaccount/token`. The `default` ServiceAccount in every namespace is what unconfigured Pods get.

Modern clusters use **bound ServiceAccount tokens** (BoundServiceAccountTokenVolume, on by default since 1.21):

- The token is generated per Pod (not stored in a Secret object).
- It is bound to the specific Pod's UID and audience — when the Pod is deleted, the token becomes invalid.
- The kubelet rotates the token before expiry (default 1h).

This eliminates the old "permanent ServiceAccount token sitting in etcd that anyone with read-Secret can grab" failure mode. For external systems, you can issue your own tokens with TokenRequest API, scoped to specific audiences (e.g., a token valid only against an internal Vault).

**Auto-mounting** is on by default but should be off for Pods that don't need the API (`automountServiceAccountToken: false` at Pod or SA level). A pod that does not need to talk to the API server has no business carrying an API credential.

### D. Defense in Depth: Pod Security Standards, NetworkPolicy, OPA

RBAC controls API access; it does *not* control what a running container can do. A container with `privileged: true` and `hostNetwork: true` can read every secret on the node and pivot the cluster — even if its ServiceAccount has zero RBAC permissions. So Kubernetes layers additional controls:

**Pod Security Standards (PSS)** define three profiles enforced by the **Pod Security Admission** controller:

| Profile | What it allows | When to use |
|---------|----------------|-------------|
| `privileged` | everything | trusted system workloads (CNI, storage drivers) |
| `baseline` | minimum container hygiene; blocks `privileged`, hostNetwork, hostPath | most user workloads |
| `restricted` | strict hardening; non-root, drop ALL capabilities, seccomp RuntimeDefault | apps with no special needs |

Enforcement is per-namespace via labels:

```yaml
metadata:
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: v1.29
```

This is **admission-time** enforcement — pods that violate are rejected at gate 3 of §A.

**NetworkPolicy** restricts pod-to-pod traffic at L3/L4. Default Kubernetes networking is "any pod can talk to any pod" (lesson 03 §A). NetworkPolicy lets you say "pods labeled `tier=backend` only accept traffic from pods labeled `tier=frontend` on port 8080." Implementation is delegated to the CNI plugin (lesson 08).

**OPA Gatekeeper / Kyverno** are policy engines that plug into admission (gate 3) via webhooks. They let you write rules like "every namespace must have a cost-center label" or "container images must come from our internal registry" in a domain-specific language (Rego for OPA, YAML for Kyverno). When Pod Security Standards are not enough, these are the next step.

The mental model: **RBAC controls API access; PSS + NetworkPolicy + OPA control workload behavior.** Both are needed. A misconfigured cluster with strict RBAC but `privileged` admission allowed is one container away from a full compromise.

### From Theory to the YAML Below

The lesson now concretizes these abstractions:

- **Section 1 (Authentication)** is gate 1 of §A — the four credential types and how the API server validates each.
- **Section 2 (Authorization Modes)** introduces the gate 2 plug-ins; you'll usually use RBAC (§B) and Webhook for advanced cases.
- **Section 3 (RBAC Deep Dive)** is the four-object model from §B with concrete YAML and the aggregation pattern that lets you compose roles.
- **Section 4 (Service Accounts)** is §C — how Pods get identities and how to lock down auto-mount.
- **Section 5 (Pod Security Standards)** is the §D PSS layer, enforced via admission labels.
- **Sections 6–7 (Security Contexts, Seccomp/AppArmor)** are workload-side hardening that PSS enforces.
- **Sections 8 (OPA Gatekeeper)** is the policy-engine extension of admission control from §D.
- **Section 9 (Network Policies)** is §D's L3/L4 layer.

Once you see the four-gate pipeline in §A, every "why was this denied?" question maps to a specific gate and a specific extension point.

---

## 1. Authentication Methods

Every request to the Kubernetes API server must be authenticated. Kubernetes does not have a built-in user database -- instead, it delegates authentication to external systems through a plugin architecture.

```
┌──────────┐     ┌─────────────────────────┐     ┌───────────────┐
│  kubectl  │────▶│    API Server            │────▶│ Authorization │
│  or SDK   │     │  Authentication Plugins: │     │   (RBAC)      │
└──────────┘     │  - X.509 Certs          │     └───────────────┘
                  │  - Bearer Tokens         │
                  │  - OIDC                  │
                  │  - Service Accounts      │
                  └─────────────────────────┘
```

### 1.1 X.509 Client Certificates

The most common method for cluster administrators. The API server validates the client certificate against its configured Certificate Authority (CA).

```bash
# Generate a private key for a new user
openssl genrsa -out developer.key 2048

# Create a Certificate Signing Request (CSR)
# The CN (Common Name) becomes the username
# The O (Organization) becomes the group
openssl req -new -key developer.key \
  -out developer.csr \
  -subj "/CN=jane/O=dev-team"

# Create a Kubernetes CertificateSigningRequest
cat <<EOF | kubectl apply -f -
apiVersion: certificates.k8s.io/v1
kind: CertificateSigningRequest
metadata:
  name: jane-csr
spec:
  request: $(cat developer.csr | base64 | tr -d '\n')
  signerName: kubernetes.io/kube-apiserver-client
  usages:
    - client auth
EOF

# Approve the CSR
kubectl certificate approve jane-csr

# Extract the signed certificate
kubectl get csr jane-csr -o jsonpath='{.status.certificate}' | base64 -d > developer.crt

# Configure kubectl context for the new user
kubectl config set-credentials jane \
  --client-certificate=developer.crt \
  --client-key=developer.key

kubectl config set-context jane-context \
  --cluster=minikube \
  --user=jane \
  --namespace=dev

kubectl config use-context jane-context
```

### 1.2 Bearer Tokens

Bearer tokens are sent in the `Authorization` header. Kubernetes supports static token files and bootstrap tokens, though these are mostly used in automated setups.

```bash
# Static token file format (one token per line)
# token,user,uid,"group1,group2"
# Not recommended for production -- use OIDC instead

# Using a bearer token with kubectl
kubectl --token="eyJhbGciOiJSUzI1NiIs..." get pods

# Using a bearer token with curl
curl -k https://API_SERVER:6443/api/v1/pods \
  -H "Authorization: Bearer eyJhbGciOiJSUzI1NiIs..."
```

### 1.3 OpenID Connect (OIDC)

OIDC is the recommended authentication method for production clusters. It delegates authentication to an external identity provider (Dex, Keycloak, Google, Azure AD).

```
┌───────┐     ┌──────────────┐     ┌───────────────┐
│ User  │────▶│ OIDC Provider│────▶│   API Server  │
│       │◀────│ (Keycloak)   │     │ validates JWT │
│       │     │              │     │ id_token      │
│ gets  │     └──────────────┘     └───────────────┘
│ token │
└───────┘
```

The API server configuration for OIDC:

```yaml
# kube-apiserver flags (in the static pod manifest or kubeadm config)
apiVersion: kubeadm.k8s.io/v1beta3
kind: ClusterConfiguration
apiServer:
  extraArgs:
    oidc-issuer-url: "https://keycloak.example.com/realms/k8s"
    oidc-client-id: "kubernetes"
    oidc-username-claim: "email"
    oidc-groups-claim: "groups"
    oidc-username-prefix: "oidc:"
    oidc-groups-prefix: "oidc:"
```

### 1.4 Service Account Tokens

Service accounts authenticate workloads running inside the cluster. Since Kubernetes 1.24, bound service account tokens (projected volumes) are the default -- they are time-limited, audience-bound, and automatically rotated.

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: my-app
  namespace: production
automountServiceAccountToken: false  # Explicitly opt out if not needed
```

```bash
# Create a short-lived token for a service account
kubectl create token my-app --duration=1h --namespace=production

# Inspect the token (it is a JWT)
kubectl create token my-app | jwt decode -
```

---

## 2. Authorization Modes

After authentication, the API server checks whether the authenticated identity is allowed to perform the requested action. Kubernetes supports multiple authorization modes that are evaluated in order.

### 2.1 RBAC (Role-Based Access Control)

RBAC is the standard authorization mode in all production clusters. It grants permissions based on roles assigned to users, groups, or service accounts.

```bash
# Check which authorization modes are enabled
kubectl api-versions | grep rbac
# rbac.authorization.k8s.io/v1

# Test whether a user can perform an action
kubectl auth can-i create deployments --namespace=dev --as=jane
# yes

kubectl auth can-i delete nodes --as=jane
# no
```

### 2.2 ABAC (Attribute-Based Access Control)

ABAC uses a static policy file and requires an API server restart to update. It is rarely used in modern clusters.

```json
{
  "apiVersion": "abac.authorization.kubernetes.io/v1beta1",
  "kind": "Policy",
  "spec": {
    "user": "jane",
    "namespace": "dev",
    "resource": "pods",
    "readonly": true
  }
}
```

### 2.3 Webhook Authorization

Webhook authorization delegates authorization decisions to an external HTTP service. This is useful for integrating with existing enterprise authorization systems.

```yaml
# Webhook authorization configuration
apiVersion: v1
kind: Config
clusters:
  - name: authz-webhook
    cluster:
      server: https://authz.example.com/authorize
      certificate-authority: /etc/kubernetes/pki/authz-ca.pem
users:
  - name: kube-apiserver
    user:
      client-certificate: /etc/kubernetes/pki/authz-client.pem
      client-key: /etc/kubernetes/pki/authz-client-key.pem
current-context: webhook
contexts:
  - context:
      cluster: authz-webhook
      user: kube-apiserver
    name: webhook
```

---

## 3. RBAC Deep Dive

RBAC has four resource types that work together:

```
                 Namespaced                    Cluster-wide
               ┌──────────┐                 ┌──────────────┐
  Permissions  │   Role   │                 │ ClusterRole  │
               └────┬─────┘                 └──────┬───────┘
                    │ binds to                      │ binds to
               ┌────▼──────────┐             ┌─────▼────────────┐
  Binding      │ RoleBinding   │             │ClusterRoleBinding│
               └───────────────┘             └──────────────────┘
```

### 3.1 Roles and ClusterRoles

A **Role** defines permissions within a specific namespace. A **ClusterRole** defines permissions cluster-wide or across all namespaces.

```yaml
# Role: allow reading pods in the "dev" namespace
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-reader
  namespace: dev
rules:
  - apiGroups: [""]           # core API group
    resources: ["pods"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["pods/log"]   # subresource
    verbs: ["get"]
```

```yaml
# ClusterRole: allow managing deployments in any namespace
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: deployment-manager
rules:
  - apiGroups: ["apps"]
    resources: ["deployments"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: ["apps"]
    resources: ["deployments/scale"]
    verbs: ["update", "patch"]
  - apiGroups: ["apps"]
    resources: ["deployments/status"]
    verbs: ["get"]
```

```yaml
# ClusterRole for non-namespaced resources (nodes)
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: node-viewer
rules:
  - apiGroups: [""]
    resources: ["nodes"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["metrics.k8s.io"]
    resources: ["nodes"]
    verbs: ["get", "list"]
```

### 3.2 RoleBindings and ClusterRoleBindings

A **RoleBinding** grants a Role (or ClusterRole) to subjects within a namespace. A **ClusterRoleBinding** grants a ClusterRole across the entire cluster.

```yaml
# RoleBinding: grant pod-reader role to user "jane" in "dev" namespace
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-pods-dev
  namespace: dev
subjects:
  - kind: User
    name: jane
    apiGroup: rbac.authorization.k8s.io
  - kind: Group
    name: dev-team
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
```

```yaml
# ClusterRoleBinding: grant cluster-wide admin to a group
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: platform-admins
subjects:
  - kind: Group
    name: oidc:platform-admins    # OIDC group with prefix
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: cluster-admin
  apiGroup: rbac.authorization.k8s.io
```

```yaml
# RoleBinding referencing a ClusterRole (scoped to namespace)
# This is a powerful pattern: define permissions once, bind per-namespace
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: deploy-manager-dev
  namespace: dev
subjects:
  - kind: ServiceAccount
    name: ci-pipeline
    namespace: ci
roleRef:
  kind: ClusterRole          # ClusterRole, not Role
  name: deployment-manager
  apiGroup: rbac.authorization.k8s.io
```

### 3.3 Aggregated ClusterRoles

Aggregated ClusterRoles combine multiple ClusterRoles using label selectors. The built-in `admin`, `edit`, and `view` roles are aggregated.

```yaml
# Custom ClusterRole that gets aggregated into the "admin" role
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: custom-resource-admin
  labels:
    rbac.authorization.k8s.io/aggregate-to-admin: "true"
rules:
  - apiGroups: ["example.com"]
    resources: ["widgets"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
```

```bash
# Verify aggregation -- the admin role now includes widget permissions
kubectl describe clusterrole admin | grep widgets
```

### 3.4 Common RBAC Patterns

```bash
# Debugging RBAC -- check what a user/service account can do
kubectl auth can-i --list --as=system:serviceaccount:dev:my-app

# Check which roles are bound in a namespace
kubectl get rolebindings -n dev -o wide

# Check cluster-level bindings
kubectl get clusterrolebindings -o wide | grep dev-team

# Impersonate a user to test permissions
kubectl get pods -n dev --as=jane --as-group=dev-team
```

**Principle of least privilege checklist:**

| Guideline | Example |
|-----------|---------|
| Use namespace-scoped Roles over ClusterRoles | `Role` in `dev` instead of `ClusterRole` everywhere |
| Avoid wildcard verbs (`*`) | Specify exact verbs: `get`, `list`, `watch` |
| Avoid wildcard resources (`*`) | List specific resources: `pods`, `services` |
| Prefer Group bindings over User bindings | Bind to `dev-team` group, not individual users |
| Audit regularly | `kubectl auth can-i --list --as=...` |

---

## 4. Service Accounts

Service accounts are the identity mechanism for pods. Every namespace has a `default` service account, and pods use it unless specified otherwise.

### 4.1 Service Account Basics

```yaml
# Create a dedicated service account
apiVersion: v1
kind: ServiceAccount
metadata:
  name: log-collector
  namespace: monitoring
```

```yaml
# Assign the service account to a pod
apiVersion: v1
kind: Pod
metadata:
  name: log-agent
  namespace: monitoring
spec:
  serviceAccountName: log-collector
  containers:
    - name: agent
      image: fluent/fluent-bit:latest
```

### 4.2 Bound Service Account Tokens

Since Kubernetes 1.24, service account tokens are projected volumes with expiration.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: api-client
spec:
  serviceAccountName: my-app
  containers:
    - name: client
      image: my-api-client:v1
      volumeMounts:
        - name: token
          mountPath: /var/run/secrets/tokens
          readOnly: true
  volumes:
    - name: token
      projected:
        sources:
          - serviceAccountToken:
              path: api-token
              expirationSeconds: 3600      # 1 hour
              audience: "https://my-api.example.com"
```

### 4.3 Disabling Auto-Mount

For pods that do not need to call the Kubernetes API, disable the automatic token mount to reduce attack surface.

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: static-web
  namespace: frontend
automountServiceAccountToken: false
---
apiVersion: v1
kind: Pod
metadata:
  name: web-server
  namespace: frontend
spec:
  serviceAccountName: static-web
  automountServiceAccountToken: false   # Also set at pod level for clarity
  containers:
    - name: nginx
      image: nginx:1.27
```

---

## 5. Pod Security Standards

Pod Security Standards (PSS) define three progressive profiles that control what pods are allowed to do. They replace the deprecated PodSecurityPolicy.

### 5.1 The Three Profiles

```
Privileged          Baseline              Restricted
─────────           ────────              ──────────
No restrictions     Prevents known        Heavily restricted
                    privilege             Best practices
                    escalations           enforced

Examples:           Blocks:               Requires:
- System daemons    - hostNetwork         - Non-root user
- Node agents       - hostPID             - Read-only root FS
- CNI plugins       - privileged          - Drop ALL capabilities
                    - hostPath            - Seccomp profile
                                          - No privilege escalation
```

### 5.2 Pod Security Admission

Pod Security Admission (PSA) is the built-in admission controller that enforces Pod Security Standards at the namespace level.

Three enforcement modes:

| Mode | Behavior |
|------|----------|
| `enforce` | Reject pods that violate the policy |
| `audit` | Allow but log violations in the audit log |
| `warn` | Allow but return a warning to the user |

### 5.3 Namespace-Level Enforcement

```yaml
# Apply Pod Security Standards to a namespace via labels
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    # Enforce the restricted profile
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: v1.30
    # Audit and warn with the same profile
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/audit-version: v1.30
    pod-security.kubernetes.io/warn: restricted
    pod-security.kubernetes.io/warn-version: v1.30
```

```yaml
# A pod that complies with the restricted profile
apiVersion: v1
kind: Pod
metadata:
  name: secure-app
  namespace: production
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: app
      image: my-app:v1
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop: ["ALL"]
      resources:
        limits:
          memory: "256Mi"
          cpu: "500m"
        requests:
          memory: "128Mi"
          cpu: "250m"
```

```bash
# Test what happens when you violate the policy
kubectl run test-privileged \
  --image=nginx \
  --namespace=production \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "nginx",
        "image": "nginx",
        "securityContext": {"privileged": true}
      }]
    }
  }'
# Error: pods "test-privileged" is forbidden: violates PodSecurity "restricted:v1.30"

# Dry-run to check compliance without creating the pod
kubectl label namespace staging \
  pod-security.kubernetes.io/enforce=restricted \
  --dry-run=server --overwrite
```

---

## 6. Security Contexts

Security contexts configure privilege and access control settings for pods and containers.

### 6.1 Pod-Level Security Context

Pod-level settings apply to all containers in the pod, including init containers.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 10000
    runAsGroup: 10000
    fsGroup: 20000           # GID for volume mounts
    fsGroupChangePolicy: "OnRootMismatch"  # Faster volume mounts
    supplementalGroups: [30000]
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: app
      image: my-app:v2
      # Container-level settings override pod-level
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop: ["ALL"]
          add: ["NET_BIND_SERVICE"]  # Only if binding to ports < 1024
```

### 6.2 Container-Level Security Context

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hardened-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hardened
  template:
    metadata:
      labels:
        app: hardened
    spec:
      securityContext:
        runAsNonRoot: true
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: app
          image: my-app:v2
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop: ["ALL"]
          # Writable directories must be tmpfs
          volumeMounts:
            - name: tmp
              mountPath: /tmp
            - name: cache
              mountPath: /app/cache
      volumes:
        - name: tmp
          emptyDir:
            sizeLimit: 100Mi
        - name: cache
          emptyDir:
            sizeLimit: 500Mi
```

**Linux capability reference:**

| Capability | Purpose | Needed by |
|-----------|---------|-----------|
| `NET_BIND_SERVICE` | Bind to ports < 1024 | Web servers on port 80/443 |
| `NET_RAW` | Raw sockets | Network diagnostics |
| `SYS_PTRACE` | Process tracing | Debuggers, profilers |
| `DAC_OVERRIDE` | Bypass file permission checks | Legacy apps |
| `SETUID` / `SETGID` | Change UID/GID | su, sudo |

---

## 7. Seccomp and AppArmor

### 7.1 Seccomp Profiles

Seccomp (secure computing) restricts which system calls a container can make. The `RuntimeDefault` profile blocks dangerous syscalls like `reboot`, `mount`, and `ptrace`.

```yaml
# Using the RuntimeDefault seccomp profile (recommended)
apiVersion: v1
kind: Pod
metadata:
  name: seccomp-default
spec:
  securityContext:
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: app
      image: my-app:v1
```

```yaml
# Custom seccomp profile (loaded from node filesystem)
apiVersion: v1
kind: Pod
metadata:
  name: seccomp-custom
spec:
  securityContext:
    seccompProfile:
      type: Localhost
      localhostProfile: profiles/custom-audit.json
  containers:
    - name: app
      image: my-app:v1
```

Custom seccomp profile example (placed in `/var/lib/kubelet/seccomp/profiles/`):

```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": ["SCMP_ARCH_X86_64"],
  "syscalls": [
    {
      "names": [
        "read", "write", "open", "close", "stat", "fstat",
        "mmap", "mprotect", "munmap", "brk",
        "rt_sigaction", "rt_sigprocmask", "ioctl",
        "access", "pipe", "select", "sched_yield",
        "clone", "execve", "exit_group", "futex",
        "epoll_ctl", "epoll_wait", "accept4",
        "socket", "connect", "bind", "listen",
        "getpid", "gettid", "nanosleep"
      ],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
```

### 7.2 AppArmor Profiles

AppArmor confines programs with per-program security profiles. Profiles must be loaded on the node before pods can reference them.

```yaml
# Pod with AppArmor profile (Kubernetes 1.30+ annotation-free)
apiVersion: v1
kind: Pod
metadata:
  name: apparmor-pod
spec:
  securityContext:
    appArmorProfile:
      type: Localhost
      localhostProfile: k8s-custom-deny-write
  containers:
    - name: app
      image: my-app:v1
```

```bash
# Check loaded AppArmor profiles on a node
ssh node01 "sudo aa-status"

# Load a custom profile
ssh node01 "sudo apparmor_parser -r /etc/apparmor.d/k8s-custom-deny-write"
```

---

## 8. OPA/Gatekeeper Policies

Open Policy Agent (OPA) Gatekeeper is a policy engine that acts as a validating admission webhook. It enables custom policy enforcement beyond what Pod Security Standards provide.

### 8.1 Architecture

```
                    ┌────────────────┐
  kubectl apply ──▶ │   API Server   │
                    │                │
                    │  Admission     │
                    │  Webhooks:     │
                    │  ┌───────────┐ │
                    │  │Gatekeeper │ │──▶ Evaluate Rego policies
                    │  │ Webhook   │ │    against constraints
                    │  └───────────┘ │
                    └────────────────┘
```

```bash
# Install Gatekeeper on minikube
helm repo add gatekeeper https://open-policy-agent.github.io/gatekeeper/charts
helm install gatekeeper gatekeeper/gatekeeper \
  --namespace gatekeeper-system \
  --create-namespace
```

### 8.2 ConstraintTemplates and Constraints

A **ConstraintTemplate** defines reusable policy logic in Rego. A **Constraint** applies that template with specific parameters.

```yaml
# ConstraintTemplate: require specific labels on all resources
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

```yaml
# Constraint: require "team" and "env" labels on all Deployments
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sRequiredLabels
metadata:
  name: deployment-must-have-labels
spec:
  match:
    kinds:
      - apiGroups: ["apps"]
        kinds: ["Deployment"]
  parameters:
    labels:
      - "team"
      - "env"
```

```yaml
# ConstraintTemplate: block use of latest tag
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: k8sblocklatesttag
spec:
  crd:
    spec:
      names:
        kind: K8sBlockLatestTag
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package k8sblocklatesttag

        violation[{"msg": msg}] {
          container := input.review.object.spec.containers[_]
          endswith(container.image, ":latest")
          msg := sprintf("Container '%v' uses ':latest' tag. Use a specific version.", [container.name])
        }

        violation[{"msg": msg}] {
          container := input.review.object.spec.containers[_]
          not contains(container.image, ":")
          msg := sprintf("Container '%v' has no tag. Use a specific version.", [container.name])
        }
```

```bash
# Check Gatekeeper audit results
kubectl get k8srequiredlabels deployment-must-have-labels -o yaml

# List all violations
kubectl get constraints -o wide
```

---

## 9. Network Policies for Security

Network policies are a critical security layer that controls pod-to-pod communication. They are enforced by the CNI plugin (Calico, Cilium, etc.).

### 9.1 Default Deny

Start with a default-deny policy and explicitly allow required traffic.

```yaml
# Default deny all ingress traffic in a namespace
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-ingress
  namespace: production
spec:
  podSelector: {}       # Selects ALL pods in the namespace
  policyTypes:
    - Ingress
---
# Default deny all egress traffic in a namespace
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-egress
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Egress
```

### 9.2 Allow Specific Traffic

```yaml
# Allow frontend pods to reach backend pods on port 8080
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend-to-backend
  namespace: production
spec:
  podSelector:
    matchLabels:
      role: backend
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              role: frontend
      ports:
        - protocol: TCP
          port: 8080
---
# Allow all pods to reach DNS (kube-dns)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dns
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Egress
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: kube-system
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53
        - protocol: TCP
          port: 53
```

```bash
# Test network policy enforcement
# Deploy a test pod
kubectl run test-client --image=busybox --rm -it --restart=Never \
  -n production -- wget -qO- --timeout=2 http://backend:8080/health

# Label the test pod as frontend to allow access
kubectl run test-client --image=busybox --rm -it --restart=Never \
  -n production --labels="role=frontend" \
  -- wget -qO- --timeout=2 http://backend:8080/health
```

---

## Exercises

### Exercise 1: Create a Read-Only RBAC Policy

Create a Role and RoleBinding that grants user `alice` read-only access to pods, services, and deployments in the `staging` namespace. Then verify the permissions using `kubectl auth can-i`.

<details><summary>Show Answer</summary>

```yaml
# role-readonly.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: staging
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: readonly
  namespace: staging
rules:
  - apiGroups: [""]
    resources: ["pods", "services"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["apps"]
    resources: ["deployments"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: alice-readonly
  namespace: staging
subjects:
  - kind: User
    name: alice
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: readonly
  apiGroup: rbac.authorization.k8s.io
```

```bash
kubectl apply -f role-readonly.yaml

# Verify permissions
kubectl auth can-i get pods -n staging --as=alice
# yes

kubectl auth can-i create pods -n staging --as=alice
# no

kubectl auth can-i delete deployments -n staging --as=alice
# no

kubectl auth can-i list services -n staging --as=alice
# yes
```

</details>

### Exercise 2: Enforce Pod Security Standards

Create a namespace called `secure-ns` with the `restricted` Pod Security Standard enforced. Then attempt to deploy a privileged pod and verify it is rejected. Finally, deploy a compliant pod.

<details><summary>Show Answer</summary>

```yaml
# secure-namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: secure-ns
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: v1.30
    pod-security.kubernetes.io/warn: restricted
```

```bash
kubectl apply -f secure-namespace.yaml

# This will be rejected
kubectl run bad-pod --image=nginx --namespace=secure-ns \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "nginx",
        "image": "nginx",
        "securityContext": {"privileged": true}
      }]
    }
  }'
# Error: violates PodSecurity "restricted:v1.30"
```

```yaml
# compliant-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: good-pod
  namespace: secure-ns
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: app
      image: busybox:1.36
      command: ["sleep", "3600"]
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop: ["ALL"]
```

```bash
kubectl apply -f compliant-pod.yaml
kubectl get pod good-pod -n secure-ns
# good-pod   1/1     Running   0          5s
```

</details>

### Exercise 3: Create a Gatekeeper Policy

Write a Gatekeeper ConstraintTemplate and Constraint that requires all pods to have resource limits (memory and cpu) defined. Test with a pod that has no resource limits.

<details><summary>Show Answer</summary>

```yaml
# constraint-template.yaml
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: k8srequireresourcelimits
spec:
  crd:
    spec:
      names:
        kind: K8sRequireResourceLimits
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package k8srequireresourcelimits

        violation[{"msg": msg}] {
          container := input.review.object.spec.containers[_]
          not container.resources.limits.memory
          msg := sprintf("Container '%v' must have memory limits", [container.name])
        }

        violation[{"msg": msg}] {
          container := input.review.object.spec.containers[_]
          not container.resources.limits.cpu
          msg := sprintf("Container '%v' must have CPU limits", [container.name])
        }
---
# constraint.yaml
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sRequireResourceLimits
metadata:
  name: must-have-limits
spec:
  match:
    kinds:
      - apiGroups: [""]
        kinds: ["Pod"]
    excludedNamespaces:
      - kube-system
      - gatekeeper-system
```

```bash
kubectl apply -f constraint-template.yaml
# Wait for template to be ready
kubectl apply -f constraint.yaml

# This should be rejected
kubectl run no-limits --image=nginx
# Error: Container 'no-limits' must have memory limits

# This should succeed
kubectl run with-limits --image=nginx \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "with-limits",
        "image": "nginx",
        "resources": {
          "limits": {"memory": "128Mi", "cpu": "500m"},
          "requests": {"memory": "64Mi", "cpu": "250m"}
        }
      }]
    }
  }'
```

</details>

### Exercise 4: Network Policy Microsegmentation

In the `microservices` namespace, create network policies that implement the following rules:
- `frontend` pods can only talk to `api` pods on port 8080
- `api` pods can only talk to `database` pods on port 5432
- `database` pods accept traffic only from `api` pods
- All pods can reach DNS

<details><summary>Show Answer</summary>

```yaml
# network-policies.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: microservices
---
# Default deny all traffic
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: microservices
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
---
# Allow DNS for all pods
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dns
  namespace: microservices
spec:
  podSelector: {}
  policyTypes:
    - Egress
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: kube-system
      ports:
        - protocol: UDP
          port: 53
        - protocol: TCP
          port: 53
---
# Frontend egress to API on 8080
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: frontend-to-api
  namespace: microservices
spec:
  podSelector:
    matchLabels:
      tier: frontend
  policyTypes:
    - Egress
  egress:
    - to:
        - podSelector:
            matchLabels:
              tier: api
      ports:
        - protocol: TCP
          port: 8080
---
# API ingress from frontend on 8080
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-ingress-from-frontend
  namespace: microservices
spec:
  podSelector:
    matchLabels:
      tier: api
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              tier: frontend
      ports:
        - protocol: TCP
          port: 8080
---
# API egress to database on 5432
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-to-database
  namespace: microservices
spec:
  podSelector:
    matchLabels:
      tier: api
  policyTypes:
    - Egress
  egress:
    - to:
        - podSelector:
            matchLabels:
              tier: database
      ports:
        - protocol: TCP
          port: 5432
---
# Database ingress from API on 5432
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: database-ingress-from-api
  namespace: microservices
spec:
  podSelector:
    matchLabels:
      tier: database
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              tier: api
      ports:
        - protocol: TCP
          port: 5432
```

```bash
kubectl apply -f network-policies.yaml

# Verify policies
kubectl get networkpolicies -n microservices
# NAME                        POD-SELECTOR     AGE
# default-deny-all            <none>           5s
# allow-dns                   <none>           5s
# frontend-to-api             tier=frontend    5s
# api-ingress-from-frontend   tier=api         5s
# api-to-database             tier=api         5s
# database-ingress-from-api   tier=database    5s
```

</details>

### Exercise 5: Service Account with Minimal Permissions

Create a service account `log-reader` in the `monitoring` namespace that can only read pod logs across all namespaces. Deploy a pod using this service account and verify it can read logs but cannot list secrets.

<details><summary>Show Answer</summary>

```yaml
# log-reader-rbac.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: monitoring
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: log-reader
  namespace: monitoring
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: pod-log-reader
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list"]
  - apiGroups: [""]
    resources: ["pods/log"]
    verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: log-reader-binding
subjects:
  - kind: ServiceAccount
    name: log-reader
    namespace: monitoring
roleRef:
  kind: ClusterRole
  name: pod-log-reader
  apiGroup: rbac.authorization.k8s.io
```

```yaml
# log-reader-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: log-reader-pod
  namespace: monitoring
spec:
  serviceAccountName: log-reader
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: kubectl
      image: bitnami/kubectl:latest
      command: ["sleep", "3600"]
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop: ["ALL"]
```

```bash
kubectl apply -f log-reader-rbac.yaml
kubectl apply -f log-reader-pod.yaml

# Verify permissions
kubectl auth can-i get pods/log --all-namespaces \
  --as=system:serviceaccount:monitoring:log-reader
# yes

kubectl auth can-i list secrets --all-namespaces \
  --as=system:serviceaccount:monitoring:log-reader
# no

kubectl auth can-i create pods \
  --as=system:serviceaccount:monitoring:log-reader
# no

# Test from inside the pod
kubectl exec -it log-reader-pod -n monitoring -- \
  kubectl logs -n kube-system kube-apiserver-minikube --tail=5

kubectl exec -it log-reader-pod -n monitoring -- \
  kubectl get secrets -n kube-system
# Error from server (Forbidden)
```

</details>

---

**Previous**: [Configuration and Secrets](./05_Configuration_and_Secrets.md) | **Next**: [Ingress and Gateway API](./07_Ingress_and_Gateway_API.md)

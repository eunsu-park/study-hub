# 07. Kubernetes Security

**Previous**: [Kubernetes Introduction](./06_Kubernetes_Intro.md) | **Next**: [Kubernetes Advanced](./08_Kubernetes_Advanced.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the Kubernetes 4C security model and its layered defense approach
2. Implement Role-Based Access Control (RBAC) with Roles, ClusterRoles, and Bindings
3. Configure ServiceAccounts to control Pod-level API access
4. Write NetworkPolicy manifests to enforce network isolation between Pods
5. Manage Secrets securely and distinguish them from ConfigMaps
6. Apply Pod security policies including SecurityContext and Pod Security Standards

---

As Kubernetes clusters grow to host production workloads, security becomes a critical concern. A misconfigured RBAC policy can grant unintended access, an open network can allow lateral movement between services, and exposed secrets can compromise entire systems. This lesson covers the essential security primitives built into Kubernetes -- from access control and network isolation to secret management and Pod hardening -- giving you the tools to defend your cluster at every layer.

## Table of Contents
1. [Kubernetes Security Overview](#1-kubernetes-security-overview)
2. [RBAC (Role-Based Access Control)](#2-rbac-role-based-access-control)
3. [ServiceAccount](#3-serviceaccount)
4. [NetworkPolicy](#4-networkpolicy)
5. [Secrets Management](#5-secrets-management)
6. [Pod Security](#6-pod-security)
7. [Practice Exercises](#7-practice-exercises)

---

## 1. Kubernetes Security Overview

### 1.1 4C Security Model

```
┌─────────────────────────────────────────────────────────────┐
│                     Cloud                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 Cluster                              │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │              Container                       │   │   │
│  │  │  ┌─────────────────────────────────────┐   │   │   │
│  │  │  │            Code                      │   │   │   │
│  │  │  │  - Vulnerability scanning            │   │   │   │
│  │  │  │  - Dependency management             │   │   │   │
│  │  │  │  - Secure coding                     │   │   │   │
│  │  │  └─────────────────────────────────────┘   │   │   │
│  │  │  - Image security                           │   │   │
│  │  │  - Runtime security                         │   │   │
│  │  │  - Resource limits                          │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  │  - RBAC, NetworkPolicy                            │   │
│  │  - Secrets management                             │   │
│  │  - Pod security                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│  - Network security                                        │
│  - IAM, firewall                                          │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Authentication and Authorization

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│    User     │────▶│   API Server │────▶│  Resources  │
│  (kubectl)  │     │              │     │   (Pods)    │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  AuthN   │ │  AuthZ   │ │ Admission│
        │          │ │          │ │ Control  │
        ├──────────┤ ├──────────┤ ├──────────┤
        │• Certs   │ │• RBAC    │ │• Validate│
        │• Tokens  │ │• ABAC    │ │• Mutate  │
        │• OIDC    │ │• Webhook │ │• Policy  │
        └──────────┘ └──────────┘ └──────────┘
```

### 1.3 Security Components

```yaml
# Check current cluster security status
# Check API server settings
kubectl describe pod kube-apiserver-<master-node> -n kube-system

# Check authentication mode
kubectl api-versions | grep rbac
# rbac.authorization.k8s.io/v1

# Check cluster permissions
kubectl auth can-i --list
```

---

## 2. RBAC (Role-Based Access Control)

### 2.1 RBAC Core Concepts

```
┌─────────────────────────────────────────────────────────────┐
│                      RBAC Components                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────┐                  ┌───────────────┐      │
│  │     Role      │                  │  ClusterRole  │      │
│  │  (Namespace)  │                  │   (Cluster)   │      │
│  └───────┬───────┘                  └───────┬───────┘      │
│          │                                  │               │
│          │ Binding                          │ Binding       │
│          ▼                                  ▼               │
│  ┌───────────────┐                  ┌───────────────┐      │
│  │ RoleBinding   │                  │ClusterRole    │      │
│  │               │                  │   Binding     │      │
│  └───────┬───────┘                  └───────┬───────┘      │
│          │                                  │               │
│          └──────────────┬───────────────────┘               │
│                         ▼                                   │
│                 ┌───────────────┐                           │
│                 │   Subjects    │                           │
│                 │ • User        │                           │
│                 │ • Group       │                           │
│                 │ • ServiceAcc  │                           │
│                 └───────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Role Definition

```yaml
# role-pod-reader.yaml
# Pod read permission in specific namespace
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: development
  name: pod-reader
rules:
  # Principle of least privilege — grant only the permissions this service actually needs
- apiGroups: [""]          # "" = core API group
  resources: ["pods"]
  verbs: ["get", "watch", "list"]  # Read-only: no create/delete prevents accidental or malicious changes

---
# role-deployment-manager.yaml
# Deployment management permission
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: development
  name: deployment-manager
rules:
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods/log"]
  verbs: ["get"]

---
# role-secret-reader.yaml
# Read specific Secrets only (using resourceNames)
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: production
  name: specific-secret-reader
rules:
- apiGroups: [""]
  resources: ["secrets"]
  resourceNames: ["app-config", "db-credentials"]  # Specific resources only
  verbs: ["get"]  # resourceNames narrows scope — even if the Role is compromised, only these two Secrets are exposed
```

### 2.3 ClusterRole Definition

```yaml
# clusterrole-node-reader.yaml
# Read node information across cluster
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: node-reader
rules:
- apiGroups: [""]
  resources: ["nodes"]
  verbs: ["get", "watch", "list"]

---
# clusterrole-pv-manager.yaml
# PersistentVolume management (cluster-scoped resource)
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: pv-manager
rules:
- apiGroups: [""]
  resources: ["persistentvolumes"]
  verbs: ["get", "list", "watch", "create", "delete"]
- apiGroups: [""]
  resources: ["persistentvolumeclaims"]
  verbs: ["get", "list", "watch"]

---
# clusterrole-namespace-admin.yaml
# Admin role across all namespaces
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: namespace-admin
rules:
- apiGroups: [""]
  resources: ["namespaces"]
  verbs: ["get", "list", "watch", "create", "delete"]
- apiGroups: [""]
  resources: ["*"]  # Wildcard grants access to ALL resources — use sparingly and audit regularly
  verbs: ["*"]

---
# Aggregated ClusterRole
# clusterrole-monitoring.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: monitoring
  labels:
    rbac.example.com/aggregate-to-monitoring: "true"
aggregationRule:
  clusterRoleSelectors:
  - matchLabels:
      rbac.example.com/aggregate-to-monitoring: "true"
rules: []  # Rules are automatically aggregated — keeps individual roles small and composable
```

### 2.4 RoleBinding & ClusterRoleBinding

```yaml
# rolebinding-pod-reader.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-pods
  namespace: development
subjects:
- kind: User
  name: jane
  apiGroup: rbac.authorization.k8s.io
- kind: Group
  name: developers
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io

---
# rolebinding-sa.yaml
# Bind role to ServiceAccount
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: app-deployment-binding
  namespace: development
subjects:
- kind: ServiceAccount
  name: app-deployer
  namespace: development
roleRef:
  kind: Role
  name: deployment-manager
  apiGroup: rbac.authorization.k8s.io

---
# clusterrolebinding.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: node-reader-binding
subjects:
- kind: Group
  name: ops-team
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: node-reader
  apiGroup: rbac.authorization.k8s.io

---
# Bind ClusterRole to specific namespace
# (Reuse ClusterRole)
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: admin-binding
  namespace: staging
subjects:
- kind: User
  name: admin-user
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole      # ClusterRole but
  name: admin            # Scope limited by RoleBinding — reuse one ClusterRole across namespaces without granting cluster-wide access
  apiGroup: rbac.authorization.k8s.io
```

### 2.5 RBAC Testing and Debugging

```bash
# Check permissions
kubectl auth can-i create pods --namespace development
# yes

kubectl auth can-i delete pods --namespace production --as jane
# no

kubectl auth can-i '*' '*' --all-namespaces --as system:serviceaccount:default:admin
# yes

# Check all permissions for specific user
kubectl auth can-i --list --as jane --namespace development

# View RBAC resources
kubectl get roles -n development
kubectl get rolebindings -n development
kubectl get clusterroles
kubectl get clusterrolebindings

# Detailed information
kubectl describe role pod-reader -n development
kubectl describe rolebinding read-pods -n development
```

---

## 3. ServiceAccount

### 3.1 ServiceAccount Basics

```yaml
# serviceaccount.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: app-service-account
  namespace: production
  annotations:
    description: "Application service account for production"
# Tokens are not automatically created in Kubernetes 1.24+

---
# Token creation (Kubernetes 1.24+)
apiVersion: v1
kind: Secret
metadata:
  name: app-sa-token
  namespace: production
  annotations:
    kubernetes.io/service-account.name: app-service-account
type: kubernetes.io/service-account-token
```

### 3.2 Using ServiceAccount in Pods

```yaml
# pod-with-sa.yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
  namespace: production
spec:
  serviceAccountName: app-service-account
  automountServiceAccountToken: true  # Auto-mount token — only enable when the app calls the K8s API
  containers:
  - name: app
    image: myapp:latest
    # Token mounted at /var/run/secrets/kubernetes.io/serviceaccount/

---
# Disable token mount (security hardening)
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
spec:
  serviceAccountName: restricted-sa
  automountServiceAccountToken: false  # Do not mount token — reduces attack surface if the container is compromised
  containers:
  - name: app
    image: myapp:latest
```

### 3.3 RBAC for ServiceAccount

```yaml
# ServiceAccount for CI/CD pipeline example
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: cicd-deployer
  namespace: cicd

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: cicd-deployer-role
rules:
# Deployment management — CI/CD needs full lifecycle control to roll out new versions
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
# Service management — deployer may need to create/update Services for new endpoints
- apiGroups: [""]
  resources: ["services"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
# ConfigMap, Secret read — read-only prevents CI/CD from overwriting production secrets
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
# Pod status check — needed for deployment verification, not modification
- apiGroups: [""]
  resources: ["pods", "pods/log"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: cicd-deployer-binding
subjects:
- kind: ServiceAccount
  name: cicd-deployer
  namespace: cicd
roleRef:
  kind: ClusterRole
  name: cicd-deployer-role
  apiGroup: rbac.authorization.k8s.io
```

### 3.4 Using ServiceAccount Tokens

```bash
# Get ServiceAccount token
TOKEN=$(kubectl create token app-service-account -n production)

# Or get from Secret
TOKEN=$(kubectl get secret app-sa-token -n production -o jsonpath='{.data.token}' | base64 -d)

# Call API with token
curl -k -H "Authorization: Bearer $TOKEN" \
  https://kubernetes.default.svc/api/v1/namespaces/production/pods

# Create kubeconfig
kubectl config set-credentials sa-user --token=$TOKEN
kubectl config set-context sa-context --cluster=my-cluster --user=sa-user
```

---

## 4. NetworkPolicy

### 4.1 NetworkPolicy Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    NetworkPolicy Behavior                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Without NetworkPolicy:                                     │
│  ┌─────┐     ┌─────┐     ┌─────┐                           │
│  │Pod A│◀───▶│Pod B│◀───▶│Pod C│  All traffic allowed      │
│  └─────┘     └─────┘     └─────┘                           │
│                                                             │
│  With NetworkPolicy:                                        │
│  ┌─────┐     ┌─────┐     ┌─────┐                           │
│  │Pod A│────▶│Pod B│  ✗  │Pod C│  Restricted by policy    │
│  └─────┘     └─────┘     └─────┘                           │
│                                                             │
│  ⚠️  Note: CNI plugin must support NetworkPolicy           │
│      (Calico, Cilium, Weave Net, etc.)                     │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Basic NetworkPolicy

```yaml
# deny-all-ingress.yaml
# Default-deny + explicit allow — limits blast radius of a compromised pod
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all-ingress
  namespace: production
spec:
  podSelector: {}  # Apply to all Pods — empty selector means "every pod in this namespace"
  policyTypes:
  - Ingress
  # No ingress rules = deny all inbound — forces every service to declare its allowed sources

---
# deny-all-egress.yaml
# Deny all outbound traffic
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all-egress
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Egress
  # No egress rules = deny all outbound

---
# default-deny-all.yaml
# Deny all traffic (most restrictive)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
```

### 4.3 Allow Policies

```yaml
# allow-frontend-to-backend.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend-to-backend
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend  # Only frontend pods can reach the backend — blocks lateral movement from other services
    ports:
    - protocol: TCP
      port: 8080

---
# allow-backend-to-database.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-backend-to-database
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: database
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: backend  # Only backend can talk to the DB — even if frontend is compromised, the DB is unreachable
    ports:
    - protocol: TCP
      port: 5432  # Restrict to the exact port — an attacker cannot probe other services on the DB pod

---
# Allow access from another namespace
# allow-from-monitoring.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-from-monitoring
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
      podSelector:
        matchLabels:
          app: prometheus
    ports:
    - protocol: TCP
      port: 9090
```

### 4.4 Complex Policies

```yaml
# comprehensive-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-server-policy
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: api-server
  policyTypes:
  - Ingress
  - Egress
  ingress:
  # 1. Allow from frontend in same namespace
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 443
  # 2. Allow from Ingress Controller
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 443
  # 3. Allow from specific IP range
  - from:
    - ipBlock:
        cidr: 10.0.0.0/8
        except:
        - 10.0.1.0/24  # Exclude this range — carve out untrusted subnets within the broader CIDR
    ports:
    - protocol: TCP
      port: 443
  egress:
  # 1. Outbound to database
  - to:
    - podSelector:
        matchLabels:
          app: database
    ports:
    - protocol: TCP
      port: 5432
  # 2. Outbound to cache server
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  # 3. Allow DNS (required!) — without this, pods cannot resolve service names and all network calls fail
  - to:
    - namespaceSelector: {}
      podSelector:
        matchLabels:
          k8s-app: kube-dns
    ports:
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 53  # TCP fallback for large DNS responses (>512 bytes) or zone transfers
```

### 4.5 NetworkPolicy Debugging

```bash
# View NetworkPolicy
kubectl get networkpolicy -n production
kubectl describe networkpolicy api-server-policy -n production

# Check Pod labels
kubectl get pods -n production --show-labels

# Connection test
kubectl run test-pod --rm -it --image=busybox -n production -- /bin/sh
# Inside Pod
wget -qO- --timeout=2 http://backend-service:8080
nc -zv database-service 5432

# Check CNI plugin
kubectl get pods -n kube-system | grep -E "calico|cilium|weave"
```

---

## 5. Secrets Management

### 5.1 Secret Types

```yaml
# 1. Opaque (generic data)
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
  namespace: production
type: Opaque
data:
  # base64 encoding required
  username: YWRtaW4=         # admin
  password: cGFzc3dvcmQxMjM=  # password123
stringData:
  # stringData doesn't need encoding — K8s base64-encodes it automatically, reducing human error
  api-key: my-secret-api-key

---
# 2. kubernetes.io/dockerconfigjson (container registry)
apiVersion: v1
kind: Secret
metadata:
  name: docker-registry-secret
type: kubernetes.io/dockerconfigjson
data:
  .dockerconfigjson: eyJhdXRocyI6eyJodHRwczovL2luZGV4LmRvY2tlci5pby92MS8iOnsidXNlcm5hbWUiOiJ1c2VyIiwicGFzc3dvcmQiOiJwYXNzIiwiYXV0aCI6ImRYTmxjanB3WVhOeiJ9fX0=

---
# 3. kubernetes.io/tls (TLS certificate)
apiVersion: v1
kind: Secret
metadata:
  name: tls-secret
type: kubernetes.io/tls
data:
  tls.crt: LS0tLS1CRUdJTi...
  tls.key: LS0tLS1CRUdJTi...

---
# 4. kubernetes.io/basic-auth
apiVersion: v1
kind: Secret
metadata:
  name: basic-auth
type: kubernetes.io/basic-auth
stringData:
  username: admin
  password: t0p-Secret
```

### 5.2 Secret Creation Commands

```bash
# Opaque Secret (literal)
kubectl create secret generic db-credentials \
  --from-literal=username=admin \
  --from-literal=password=secret123 \
  -n production

# Create from file
kubectl create secret generic ssh-key \
  --from-file=ssh-privatekey=~/.ssh/id_rsa \
  --from-file=ssh-publickey=~/.ssh/id_rsa.pub

# Docker registry secret
kubectl create secret docker-registry regcred \
  --docker-server=ghcr.io \
  --docker-username=myuser \
  --docker-password=mytoken \
  --docker-email=user@example.com

# TLS secret
kubectl create secret tls app-tls \
  --cert=path/to/cert.pem \
  --key=path/to/key.pem
```

### 5.3 Using Secrets

```yaml
# Use as environment variables
apiVersion: v1
kind: Pod
metadata:
  name: app-with-secrets
spec:
  containers:
  - name: app
    image: myapp:latest
    env:
    # Use specific key only
    - name: DB_USERNAME
      valueFrom:
        secretKeyRef:
          name: db-credentials
          key: username
    - name: DB_PASSWORD
      valueFrom:
        secretKeyRef:
          name: db-credentials
          key: password
    # Use entire Secret as env vars
    envFrom:
    - secretRef:
        name: app-secrets

---
# Mount as volume
apiVersion: v1
kind: Pod
metadata:
  name: app-with-secret-volume
spec:
  containers:
  - name: app
    image: myapp:latest
    volumeMounts:
    - name: secret-volume
      mountPath: /etc/secrets
      readOnly: true  # Prevent the app from accidentally overwriting secret files
    - name: tls-volume
      mountPath: /etc/tls
      readOnly: true
  volumes:
  - name: secret-volume
    secret:
      secretName: app-secrets
      # Mount specific keys only — avoids exposing unrelated secrets in the same Secret object
      items:
      - key: api-key
        path: api-key.txt
        mode: 0400  # File permissions — owner-read-only prevents other processes from reading the secret
  - name: tls-volume
    secret:
      secretName: tls-secret

---
# Image Pull Secret
apiVersion: v1
kind: Pod
metadata:
  name: private-image-pod
spec:
  containers:
  - name: app
    image: ghcr.io/myorg/private-app:latest
  imagePullSecrets:
  - name: regcred
```

### 5.4 Secret Security Hardening

```yaml
# Secret encryption config (kube-apiserver)
# /etc/kubernetes/encryption-config.yaml
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
  - resources:
      - secrets
    providers:
      - aescbc:  # Encrypt Secrets at rest in etcd — without this, anyone with etcd access reads plaintext
          keys:
            - name: key1
              secret: <base64-encoded-32-byte-key>
      - identity: {}  # Fallback (unencrypted) — listed last so new writes use aescbc, but old unencrypted data is still readable

---
# Restrict Secret access with RBAC
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: secret-reader
  namespace: production
rules:
- apiGroups: [""]
  resources: ["secrets"]
  resourceNames: ["app-secrets"]  # Specific Secret only
  verbs: ["get"]
```

### 5.5 External Secret Management Tools

```yaml
# External Secrets Operator example
# Fetch from AWS Secrets Manager
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: aws-secret
  namespace: production
spec:
  refreshInterval: 1h  # Periodic sync ensures rotated secrets propagate without redeployment
  secretStoreRef:
    name: aws-secretsmanager
    kind: SecretStore
  target:
    name: db-credentials  # K8s Secret name to create
  data:
  - secretKey: username
    remoteRef:
      key: production/db-credentials
      property: username
  - secretKey: password
    remoteRef:
      key: production/db-credentials
      property: password

---
# Sealed Secrets (for GitOps)
# Encrypted with kubeseal
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: mysecret
  namespace: production
spec:
  encryptedData:
    password: AgBy8hCi...encrypted-data...
```

---

## 6. Pod Security

### 6.1 Pod Security Standards

```
┌─────────────────────────────────────────────────────────────┐
│              Pod Security Standards (PSS)                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Privileged                                                 │
│  ├── Unrestricted                                          │
│  └── For system Pods                                       │
│                                                             │
│  Baseline                                                   │
│  ├── Prevents known privilege escalation                  │
│  ├── Forbids hostNetwork, hostPID                         │
│  └── Suitable for most workloads                          │
│                                                             │
│  Restricted                                                 │
│  ├── Strong security policy                               │
│  ├── Non-root execution required                          │
│  ├── Read-only root filesystem                            │
│  └── For security-sensitive workloads                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Pod Security Admission

```yaml
# Apply security level to namespace
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    # enforce: deny violations
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: latest
    # audit: record in audit log
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/audit-version: latest
    # warn: show warning message
    pod-security.kubernetes.io/warn: restricted
    pod-security.kubernetes.io/warn-version: latest

---
# baseline level namespace
apiVersion: v1
kind: Namespace
metadata:
  name: staging
  labels:
    pod-security.kubernetes.io/enforce: baseline
    pod-security.kubernetes.io/warn: restricted
```

### 6.3 Security Context

```yaml
# secure-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
spec:
  # Pod-level security context
  securityContext:
    runAsNonRoot: true  # Prevents container from running as UID 0 even if the image defaults to root
    runAsUser: 1000
    runAsGroup: 3000
    fsGroup: 2000  # Volumes are owned by this GID — ensures the non-root user can read/write mounted data
    seccompProfile:
      type: RuntimeDefault  # Drop dangerous syscalls — defense-in-depth even if container runtime has a bug

  containers:
  - name: app
    image: myapp:latest
    # Container-level security context
    securityContext:
      allowPrivilegeEscalation: false  # Blocks setuid/setgid binaries from gaining elevated privileges
      readOnlyRootFilesystem: true  # Immutable filesystem: an attacker cannot install tools or drop malware
      capabilities:
        drop:
          - ALL  # Drop all Linux capabilities — add back only what the app truly needs
        # Add only necessary capabilities
        # add:
        #   - NET_BIND_SERVICE

    # Resource limits
    resources:
      limits:
        cpu: "500m"
        memory: "128Mi"  # limits prevent one pod from starving others on the node
      requests:
        cpu: "250m"
        memory: "64Mi"  # requests guarantee scheduling — the scheduler reserves this much capacity

    # Temporary volumes (for read-only root when writes needed)
    volumeMounts:
    - name: tmp
      mountPath: /tmp
    - name: cache
      mountPath: /app/cache

  volumes:
  - name: tmp
    emptyDir: {}
  - name: cache
    emptyDir:
      sizeLimit: 100Mi
```

### 6.4 Advanced Security Configuration

```yaml
# highly-secure-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: secure-app
  namespace: production
spec:
  replicas: 3  # Multiple replicas for high availability — if one pod crashes, others continue serving
  selector:
    matchLabels:
      app: secure-app
  template:
    metadata:
      labels:
        app: secure-app
    spec:
      # Don't mount ServiceAccount token — most apps don't call the K8s API, so the token is pure attack surface
      automountServiceAccountToken: false

      # Pod security context
      securityContext:
        runAsNonRoot: true
        runAsUser: 65534  # nobody — a well-known non-root UID with no login shell or home directory
        runAsGroup: 65534
        fsGroup: 65534
        seccompProfile:
          type: RuntimeDefault

      containers:
      - name: app
        image: myapp:latest
        imagePullPolicy: Always  # Ensures the latest digest is pulled — prevents stale cached images in production

        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true  # Immutable filesystem: an attacker cannot install tools or drop malware
          capabilities:
            drop:
              - ALL  # Drop dangerous syscalls — defense-in-depth even if container runtime has a bug

        # Ports
        ports:
        - containerPort: 8080
          protocol: TCP

        # Resource limits
        resources:
          limits:
            cpu: "1"
            memory: "512Mi"  # limits prevent one pod from starving others
          requests:
            cpu: "100m"
            memory: "128Mi"  # requests guarantee scheduling; the scheduler reserves this much

        # Health checks
        livenessProbe:  # liveness restarts the pod; separate from readiness to avoid cascading restarts
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 10

        readinessProbe:  # readiness gates traffic; a failing probe removes the pod from the Service
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: config
          mountPath: /etc/app
          readOnly: true

      volumes:
      - name: tmp
        emptyDir:
          medium: Memory  # tmpfs in RAM — faster I/O and data is automatically wiped when the pod terminates
          sizeLimit: 64Mi
      - name: config
        configMap:
          name: app-config

      # Forbid host network/PID — prevents container from seeing host processes or sniffing host traffic
      hostNetwork: false
      hostPID: false
      hostIPC: false

      # DNS policy
      dnsPolicy: ClusterFirst
```

### 6.5 Security Scanning

```bash
# Image vulnerability scanning (Trivy)
trivy image myapp:latest

# Cluster security scan (kubescape)
kubescape scan framework nsa --exclude-namespaces kube-system

# Pod security check (kube-bench)
kubectl apply -f https://raw.githubusercontent.com/aquasecurity/kube-bench/main/job.yaml
kubectl logs job/kube-bench

# OPA/Gatekeeper policy check
kubectl get constrainttemplates
kubectl get constraints
```

---

## 7. Practice Exercises

### Exercise 1: Development Team RBAC Configuration
```yaml
# Requirements:
# - Developers can manage Pods, Deployments, Services in development namespace
# - In production namespace, can only view Pods
# - No access to Secrets

# Write Role and RoleBinding
```

### Exercise 2: Microservices NetworkPolicy
```yaml
# Requirements:
# - Communication only: frontend -> api-gateway -> backend -> database
# - Allow monitoring namespace to access /metrics on all Pods
# - Only frontend accessible from outside

# Write NetworkPolicy
```

### Exercise 3: Secure Application Deployment
```yaml
# Requirements:
# - Run as non-root user
# - Read-only root filesystem
# - Drop all capabilities
# - Set resource limits
# - Mount Secrets as both env vars and volumes

# Write Deployment
```

### Exercise 4: Security Audit
```bash
# Check the following:
# 1. Find privileged Pods in cluster
# 2. Find Pods using default ServiceAccount
# 3. Find Pods with Secrets exposed as env vars
# 4. Find namespaces without NetworkPolicy

# Write commands
```

---

## References

- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/)
- [RBAC Documentation](https://kubernetes.io/docs/reference/access-authn-authz/rbac/)
- [Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
- [Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)

---

## Exercises

### Exercise 1: Create RBAC for a Development Team

Apply the principle of least privilege using Roles and RoleBindings.

1. Create a `dev` namespace: `kubectl create namespace dev`
2. Create a ServiceAccount for a developer: `kubectl create serviceaccount developer -n dev`
3. Write a Role manifest that grants the `developer` SA permission to `get`, `list`, `watch`, `create`, and `delete` Pods and Deployments in the `dev` namespace
4. Write a RoleBinding that binds the Role to the `developer` ServiceAccount
5. Apply both manifests: `kubectl apply -f role.yaml -f rolebinding.yaml`
6. Test that the SA can list Pods: `kubectl auth can-i list pods --as=system:serviceaccount:dev:developer -n dev`
7. Test that it cannot access Secrets: `kubectl auth can-i get secrets --as=system:serviceaccount:dev:developer -n dev`

### Exercise 2: Enforce Network Isolation with NetworkPolicy

Use NetworkPolicy to allow only intended traffic flows.

1. Create a namespace with three deployments: `frontend`, `backend`, and `database`
2. Apply a default-deny NetworkPolicy that blocks all ingress traffic in the namespace:
   ```yaml
   apiVersion: networking.k8s.io/v1
   kind: NetworkPolicy
   metadata:
     name: default-deny
   spec:
     podSelector: {}
     policyTypes:
       - Ingress
   ```
3. Verify that `frontend` can no longer reach `backend` (`kubectl exec` into frontend pod and try `curl backend`)
4. Write and apply a NetworkPolicy that allows `frontend` to reach `backend` on port 8080
5. Write and apply a NetworkPolicy that allows `backend` to reach `database` on port 5432, but blocks `frontend` → `database` directly
6. Confirm the allowed paths work and the denied path is blocked

### Exercise 3: Harden a Pod with SecurityContext

Apply Pod-level hardening following the principle of least privilege.

1. Create a Pod manifest with the following security constraints:
   ```yaml
   securityContext:
     runAsNonRoot: true
     runAsUser: 1000
     fsGroup: 2000
   containers:
   - name: app
     image: nginx:alpine
     securityContext:
       allowPrivilegeEscalation: false
       readOnlyRootFilesystem: true
       capabilities:
         drop: ["ALL"]
   ```
2. Apply the manifest and check if the Pod starts (`nginx` requires write access to some directories — it will fail)
3. Fix the issue by adding an `emptyDir` volume mounted at `/tmp` and `/var/cache/nginx`
4. Verify the Pod is running and exec into it: confirm `whoami` returns a non-root user
5. Attempt to write to `/` inside the container and observe the permission denial

### Exercise 4: Manage Secrets Securely

Practice secure Secret creation and consumption patterns.

1. Create a Secret from literal values:
   ```bash
   kubectl create secret generic app-credentials \
     --from-literal=DB_USER=admin \
     --from-literal=DB_PASS=s3cr3t
   ```
2. Inspect the Secret: `kubectl get secret app-credentials -o yaml`
3. Decode a value: `kubectl get secret app-credentials -o jsonpath='{.data.DB_PASS}' | base64 -d`
4. Create a Pod that mounts the Secret as a **volume** at `/run/secrets/app` instead of environment variables
5. Exec into the Pod and read the mounted files: `cat /run/secrets/app/DB_PASS`
6. Explain the security benefit of volume mounts over environment variables for secrets

### Exercise 5: Apply Pod Security Standards

Enforce workload security policies at the namespace level using Pod Security Standards.

1. Label a namespace to enforce the `restricted` Pod Security Standard:
   ```bash
   kubectl label namespace dev \
     pod-security.kubernetes.io/enforce=restricted \
     pod-security.kubernetes.io/enforce-version=latest
   ```
2. Attempt to create a Pod that runs as root in the `dev` namespace — observe the rejection
3. Fix the Pod manifest to comply with the `restricted` standard (non-root user, no privilege escalation, drop all capabilities, read-only root filesystem)
4. Successfully apply the compliant Pod
5. Change the label to `baseline` and repeat step 2 — observe which Pods are now allowed

---

**Previous**: [Kubernetes Introduction](./06_Kubernetes_Intro.md) | **Next**: [Kubernetes Advanced](./08_Kubernetes_Advanced.md)

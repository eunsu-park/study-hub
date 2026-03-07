# GitOps

**Previous**: [Deployment Strategies](./13_Deployment_Strategies.md) | **Next**: [Secrets Management](./15_Secrets_Management.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define GitOps principles and explain how they differ from traditional CI/CD push-based deployments
2. Compare pull-based (ArgoCD, Flux) and push-based (Jenkins, GitHub Actions) deployment models
3. Deploy and configure ArgoCD with Application CRDs, sync policies, and automated rollback
4. Set up Flux for continuous delivery with GitRepository and Kustomization resources
5. Implement drift detection and self-healing to ensure the cluster always matches the desired state in Git
6. Design a multi-environment GitOps workflow with promotion pipelines across dev, staging, and production

---

GitOps is an operational framework that uses Git as the single source of truth for declarative infrastructure and application configuration. Instead of running `kubectl apply` from a CI pipeline or an engineer's laptop, a GitOps operator continuously watches a Git repository and automatically reconciles the cluster state to match what is declared in Git. If someone makes a manual change to the cluster, the operator detects the drift and reverts it. This model brings the benefits of version control -- history, audit trail, peer review, rollback -- to infrastructure operations.

> **Analogy -- Thermostat vs Manual Heating**: Traditional deployment is like manually turning a heater on and off -- you push a command, hope the temperature is right, and must intervene if it drifts. GitOps is like a thermostat: you declare the desired temperature (Git), and the system continuously adjusts to maintain it. If someone opens a window (manual cluster change), the thermostat (GitOps operator) detects the drift and compensates automatically.

## 1. GitOps Principles

### 1.1 The Four Principles

| Principle | Description |
|-----------|-------------|
| **Declarative** | The entire system is described declaratively (Kubernetes manifests, Helm charts, Kustomize) |
| **Versioned and immutable** | The desired state is stored in Git, providing a complete audit trail and the ability to revert |
| **Pulled automatically** | Software agents (ArgoCD, Flux) pull the desired state from Git and apply it to the cluster |
| **Continuously reconciled** | Agents continuously compare actual state vs desired state and correct any drift |

### 1.2 Push-Based vs Pull-Based Deployment

```
Push-Based (Traditional CI/CD):
┌──────┐   ┌──────┐   ┌──────────┐   kubectl apply   ┌──────────┐
│ Dev  │──→│  CI  │──→│ CD Pipeline│─────────────────→│ Cluster  │
│      │   │Build │   │(Jenkins,  │                   │          │
│      │   │Test  │   │ GH Actions│                   │          │
└──────┘   └──────┘   └──────────┘                   └──────────┘
                       ↑ Credentials stored
                         in CI system

Pull-Based (GitOps):
┌──────┐   ┌──────┐   ┌──────────┐                   ┌──────────┐
│ Dev  │──→│  CI  │──→│ Git Repo │◄─── pull ──────── │ GitOps   │
│      │   │Build │   │(manifests│     (reconcile)   │ Operator │
│      │   │Test  │   │ + images)│                   │(ArgoCD / │
└──────┘   └──────┘   └──────────┘                   │ Flux)    │
                                                      │ IN the   │
                                                      │ cluster  │
                                                      └──────────┘
                                          ↑ Credentials stay
                                            inside the cluster
```

### 1.3 Benefits of Pull-Based GitOps

| Benefit | Explanation |
|---------|-------------|
| **Security** | CI system does not need cluster credentials. The operator runs inside the cluster with limited RBAC. |
| **Audit trail** | Every change is a Git commit with author, timestamp, and review history. |
| **Drift detection** | Operator continuously reconciles; manual `kubectl` changes are reverted. |
| **Rollback** | `git revert` a commit, and the operator restores the previous state. |
| **Disaster recovery** | Rebuild a cluster from scratch by pointing the operator at the Git repo. |
| **Compliance** | All changes go through pull requests with required reviews and CI checks. |

---

## 2. Repository Structure

### 2.1 Monorepo vs Multi-Repo

| Pattern | Structure | Pros | Cons |
|---------|-----------|------|------|
| **Monorepo** | All environments in one repo | Easy cross-env comparison, single PR for promotion | Permissions harder to scope, large repo |
| **Multi-repo** | Separate repos per env or per app | Fine-grained access control, isolated blast radius | Harder to coordinate, more repos to manage |
| **Hybrid** | App repo + environment repo | App team owns manifests, platform team owns env config | Most common in practice |

### 2.2 Recommended Directory Layout

```
gitops-repo/
├── apps/                          # Application manifests
│   ├── webapp/
│   │   ├── base/                  # Kustomize base (shared)
│   │   │   ├── kustomization.yaml
│   │   │   ├── deployment.yaml
│   │   │   ├── service.yaml
│   │   │   └── hpa.yaml
│   │   └── overlays/              # Environment-specific overrides
│   │       ├── dev/
│   │       │   ├── kustomization.yaml
│   │       │   └── patch-replicas.yaml
│   │       ├── staging/
│   │       │   ├── kustomization.yaml
│   │       │   └── patch-replicas.yaml
│   │       └── production/
│   │           ├── kustomization.yaml
│   │           └── patch-replicas.yaml
│   └── payment-service/
│       ├── base/
│       └── overlays/
├── infrastructure/                # Cluster infrastructure
│   ├── monitoring/
│   │   ├── prometheus/
│   │   └── grafana/
│   ├── ingress/
│   │   └── nginx-ingress/
│   └── cert-manager/
└── clusters/                      # Cluster-specific config
    ├── dev/
    │   ├── apps.yaml              # ArgoCD ApplicationSet or Flux Kustomization
    │   └── infrastructure.yaml
    ├── staging/
    └── production/
```

### 2.3 Kustomize Base and Overlays

```yaml
# apps/webapp/base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
  - service.yaml
  - hpa.yaml

# apps/webapp/base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: webapp
          image: webapp:latest  # Overridden per environment

# apps/webapp/overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: production
resources:
  - ../../base
patches:
  - path: patch-replicas.yaml
images:
  - name: webapp
    newTag: v2.1.0     # Pin production to specific version

# apps/webapp/overlays/production/patch-replicas.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp
spec:
  replicas: 10         # Production needs 10 replicas
```

---

## 3. ArgoCD

### 3.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ArgoCD Architecture                         │
│                                                                  │
│  ┌──────────────┐                                               │
│  │  Git Repo    │──────────────────────────────┐                │
│  │ (desired     │                              │                │
│  │  state)      │                              │                │
│  └──────────────┘                              │                │
│                                                 ▼                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  ArgoCD (runs inside the cluster)                        │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │ Repo Server  │  │ Application  │  │    API       │  │   │
│  │  │ (fetch &     │  │ Controller   │  │   Server     │  │   │
│  │  │  render      │  │ (reconcile   │  │  (UI + CLI)  │  │   │
│  │  │  manifests)  │  │  loop)       │  │              │  │   │
│  │  └──────────────┘  └──────┬───────┘  └──────────────┘  │   │
│  │                           │                              │   │
│  │                    Compare desired                        │   │
│  │                    vs actual state                        │   │
│  │                           │                              │   │
│  │                    ┌──────┴───────┐                      │   │
│  │                    │  Kubernetes  │                      │   │
│  │                    │   Cluster    │                      │   │
│  │                    │ (actual      │                      │   │
│  │                    │  state)      │                      │   │
│  │                    └──────────────┘                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Installation

```bash
# Install ArgoCD in the cluster
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Get the initial admin password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d

# Port-forward the UI
kubectl port-forward svc/argocd-server -n argocd 8080:443

# Login via CLI
argocd login localhost:8080 --username admin --password <password>

# Add a Git repository
argocd repo add https://github.com/org/gitops-repo.git \
  --username git --password <token>
```

### 3.3 Application CRD

```yaml
# ArgoCD Application: declarative app definition
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: webapp-production
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io  # Cascade delete
spec:
  project: default

  # Source: where to find the manifests
  source:
    repoURL: https://github.com/org/gitops-repo.git
    targetRevision: main
    path: apps/webapp/overlays/production

  # Destination: where to deploy
  destination:
    server: https://kubernetes.default.svc
    namespace: production

  # Sync policy
  syncPolicy:
    automated:
      prune: true          # Delete resources removed from Git
      selfHeal: true       # Revert manual changes to match Git
      allowEmpty: false    # Prevent accidental deletion of all resources
    syncOptions:
      - CreateNamespace=true
      - PrunePropagationPolicy=foreground
      - PruneLast=true     # Prune after all other sync operations
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m

  # Health checks and status
  ignoreDifferences:
    - group: apps
      kind: Deployment
      jsonPointers:
        - /spec/replicas   # Ignore HPA-managed replica count
```

### 3.4 Sync Policies

| Policy | Behavior |
|--------|----------|
| **Manual sync** | Engineer clicks "Sync" in the UI or runs `argocd app sync`. Default mode. |
| **Automated sync** | ArgoCD auto-syncs when Git changes. Set `syncPolicy.automated`. |
| **Self-heal** | Revert manual cluster changes. Set `automated.selfHeal: true`. |
| **Prune** | Delete resources that are in the cluster but not in Git. Set `automated.prune: true`. |
| **Sync waves** | Control sync order with `argocd.argoproj.io/sync-wave` annotations. Lower numbers sync first. |

### 3.5 Sync Waves (Ordering)

```yaml
# CRDs must be created before custom resources
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  annotations:
    argocd.argoproj.io/sync-wave: "-1"    # Sync first

---
# Namespace before deployments
apiVersion: v1
kind: Namespace
metadata:
  name: production
  annotations:
    argocd.argoproj.io/sync-wave: "0"

---
# ConfigMaps before deployments
apiVersion: v1
kind: ConfigMap
metadata:
  annotations:
    argocd.argoproj.io/sync-wave: "1"

---
# Deployment after dependencies
apiVersion: apps/v1
kind: Deployment
metadata:
  annotations:
    argocd.argoproj.io/sync-wave: "2"     # Sync last
```

### 3.6 ApplicationSet (Multi-Environment)

```yaml
# Generate Applications for all environments from a single template
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: webapp
  namespace: argocd
spec:
  generators:
    - list:
        elements:
          - env: dev
            cluster: https://kubernetes.default.svc
            namespace: dev
            revision: develop
          - env: staging
            cluster: https://kubernetes.default.svc
            namespace: staging
            revision: main
          - env: production
            cluster: https://prod-cluster.example.com
            namespace: production
            revision: main
  template:
    metadata:
      name: 'webapp-{{env}}'
    spec:
      project: default
      source:
        repoURL: https://github.com/org/gitops-repo.git
        targetRevision: '{{revision}}'
        path: 'apps/webapp/overlays/{{env}}'
      destination:
        server: '{{cluster}}'
        namespace: '{{namespace}}'
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
```

### 3.7 Rollback

```bash
# View application history
argocd app history webapp-production

# Rollback to a previous revision
argocd app rollback webapp-production <revision-number>

# Or simply revert the Git commit and let ArgoCD sync
git revert HEAD
git push origin main
# ArgoCD detects the change and syncs automatically
```

---

## 4. Flux

### 4.1 Architecture

Flux v2 uses a set of specialized controllers, each responsible for a specific part of the GitOps pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                       Flux Architecture                          │
│                                                                  │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │ Source        │      │ Kustomize    │      │ Helm         │  │
│  │ Controller   │─────→│ Controller   │      │ Controller   │  │
│  │ (fetch Git/  │      │ (apply       │      │ (install     │  │
│  │  Helm repos) │      │  manifests)  │      │  Helm charts)│  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│         │                      │                     │          │
│         │              ┌──────────────┐              │          │
│         └─────────────→│ Notification │◄─────────────┘          │
│                        │ Controller   │                          │
│                        │ (Slack, etc.)│                          │
│                        └──────────────┘                          │
│                                                                  │
│  ┌──────────────┐                                               │
│  │ Image        │                                               │
│  │ Automation   │  (auto-update image tags in Git)              │
│  │ Controller   │                                               │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Bootstrap Flux

```bash
# Install Flux CLI
curl -s https://fluxcd.io/install.sh | sudo bash

# Bootstrap Flux into a cluster (creates Flux resources and pushes to Git)
flux bootstrap github \
  --owner=org \
  --repository=gitops-repo \
  --branch=main \
  --path=clusters/production \
  --personal

# Check Flux status
flux get all
```

### 4.3 GitRepository and Kustomization

```yaml
# Source: tell Flux where to find manifests
apiVersion: source.toolkit.fluxcd.io/v1
kind: GitRepository
metadata:
  name: gitops-repo
  namespace: flux-system
spec:
  interval: 1m                    # Poll Git every minute
  url: https://github.com/org/gitops-repo.git
  ref:
    branch: main
  secretRef:
    name: git-credentials          # Git authentication

---
# Kustomization: tell Flux what to apply from the source
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: webapp-production
  namespace: flux-system
spec:
  interval: 5m                    # Reconcile every 5 minutes
  sourceRef:
    kind: GitRepository
    name: gitops-repo
  path: ./apps/webapp/overlays/production
  prune: true                     # Delete resources removed from Git
  targetNamespace: production
  healthChecks:
    - apiVersion: apps/v1
      kind: Deployment
      name: webapp
      namespace: production
  timeout: 3m
```

### 4.4 Image Automation (Auto-Update Tags)

```yaml
# Watch a container registry for new tags
apiVersion: image.toolkit.fluxcd.io/v1beta2
kind: ImageRepository
metadata:
  name: webapp
  namespace: flux-system
spec:
  image: registry.example.com/webapp
  interval: 5m

---
# Policy: select the latest semver tag
apiVersion: image.toolkit.fluxcd.io/v1beta2
kind: ImagePolicy
metadata:
  name: webapp
  namespace: flux-system
spec:
  imageRepositoryRef:
    name: webapp
  policy:
    semver:
      range: ">=1.0.0"    # Latest tag matching semver >= 1.0.0

---
# Auto-update the deployment manifest in Git
apiVersion: image.toolkit.fluxcd.io/v1beta1
kind: ImageUpdateAutomation
metadata:
  name: webapp
  namespace: flux-system
spec:
  interval: 5m
  sourceRef:
    kind: GitRepository
    name: gitops-repo
  git:
    checkout:
      ref:
        branch: main
    commit:
      author:
        name: flux-bot
        email: flux@example.com
      messageTemplate: "chore: update webapp to {{.NewTag}}"
    push:
      branch: main
  update:
    path: ./apps/webapp
    strategy: Setters
```

---

## 5. Drift Detection and Self-Healing

### 5.1 What is Drift

Drift occurs when the actual cluster state diverges from the desired state in Git. Common causes:

| Cause | Example | Risk |
|-------|---------|------|
| **Manual kubectl** | Engineer runs `kubectl scale` during incident | Undocumented change, may conflict with next sync |
| **Direct API access** | Dashboard or operator modifies resources | Bypasses review process |
| **Controller mutations** | HPA changes replica count, cert-manager updates secrets | Expected drift (should be ignored) |
| **Failed sync** | ArgoCD/Flux applied partially due to error | Half-deployed state |

### 5.2 ArgoCD Drift Detection

```bash
# Check if application is in sync
argocd app get webapp-production

# Status values:
# - Synced:    actual state matches desired state
# - OutOfSync: drift detected
# - Unknown:   cannot determine (connection issue)

# View diff between Git and cluster
argocd app diff webapp-production

# Force sync to resolve drift
argocd app sync webapp-production --force
```

### 5.3 Ignoring Expected Drift

```yaml
# ArgoCD: ignore HPA-managed replica count and status fields
spec:
  ignoreDifferences:
    - group: apps
      kind: Deployment
      jsonPointers:
        - /spec/replicas              # HPA manages this
    - group: autoscaling
      kind: HorizontalPodAutoscaler
      jqPathExpressions:
        - .status                      # Status is runtime-generated
    - group: ""
      kind: Secret
      name: webhook-tls
      jsonPointers:
        - /data                        # cert-manager regenerates TLS
```

---

## 6. Multi-Environment Management

### 6.1 Promotion Pipeline

```
dev → staging → production

┌────────────┐   PR merge   ┌────────────┐   PR merge   ┌────────────┐
│  dev/      │─────────────→│  staging/  │─────────────→│ production/│
│  webapp:   │              │  webapp:   │              │  webapp:   │
│  v2.1.0    │              │  v2.1.0    │              │  v2.1.0    │
└────────────┘              └────────────┘              └────────────┘
     ↑                           ↑                           ↑
  Auto-sync                   Auto-sync               Manual approval
  (ArgoCD)                    (ArgoCD)                 then auto-sync
```

### 6.2 Implementation with Kustomize Overlays

```bash
# Promote from dev to staging: update the image tag in staging overlay
cd apps/webapp/overlays/staging
kustomize edit set image webapp=webapp:v2.1.0

# Commit and push
git add .
git commit -m "promote webapp v2.1.0 to staging"
git push origin main

# ArgoCD detects the change and syncs staging

# After staging validation, promote to production
cd apps/webapp/overlays/production
kustomize edit set image webapp=webapp:v2.1.0
git add .
git commit -m "promote webapp v2.1.0 to production"
git push origin main
```

### 6.3 Automated Promotion with CI

```yaml
# .github/workflows/promote.yml
name: Promote to Staging
on:
  workflow_dispatch:
    inputs:
      version:
        description: "Version to promote"
        required: true

jobs:
  promote:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Update staging image tag
        run: |
          cd apps/webapp/overlays/staging
          kustomize edit set image webapp=webapp:${{ inputs.version }}

      - name: Commit and push
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add .
          git commit -m "promote webapp ${{ inputs.version }} to staging"
          git push origin main
```

---

## 7. ArgoCD vs Flux Comparison

| Feature | ArgoCD | Flux |
|---------|--------|------|
| **UI** | Rich web UI with visualization | No built-in UI (use Weave GitOps) |
| **Multi-cluster** | Built-in multi-cluster management | Requires additional setup |
| **Sync model** | Application CRD, ApplicationSet | GitRepository + Kustomization CRDs |
| **Helm support** | Native Helm rendering | HelmRelease CRD |
| **Image automation** | Requires Argo CD Image Updater | Built-in Image Automation Controller |
| **RBAC** | Built-in with projects and roles | Kubernetes RBAC |
| **Learning curve** | Lower (UI-driven) | Higher (CRD-driven, no UI) |
| **Community** | Larger community, CNCF graduated | CNCF graduated, strong Kubernetes-native focus |

---

## 8. Next Steps

- [15_Secrets_Management.md](./15_Secrets_Management.md) - Managing secrets in GitOps workflows
- [13_Deployment_Strategies.md](./13_Deployment_Strategies.md) - Deployment strategies that complement GitOps

---

## Exercises

### Exercise 1: GitOps Repository Design

Design a GitOps repository structure for a company with the following setup:
- 3 environments: dev, staging, production
- 5 microservices: auth, users, orders, payments, notifications
- Production runs on a separate cluster from dev/staging
- The platform team manages infrastructure (ingress, monitoring, cert-manager)
- Application teams manage their own service manifests

<details>
<summary>Show Answer</summary>

**Recommended structure (hybrid multi-repo):**

```
# Repo 1: gitops-infrastructure (owned by platform team)
gitops-infrastructure/
├── base/
│   ├── ingress-nginx/
│   ├── cert-manager/
│   ├── prometheus/
│   ├── grafana/
│   └── argocd/
├── overlays/
│   ├── dev-staging-cluster/
│   │   ├── kustomization.yaml
│   │   └── values-override.yaml
│   └── production-cluster/
│       ├── kustomization.yaml
│       └── values-override.yaml
└── clusters/
    ├── dev-staging/
    │   └── infrastructure.yaml    # ArgoCD Application
    └── production/
        └── infrastructure.yaml

# Repo 2: gitops-apps (owned by application teams)
gitops-apps/
├── apps/
│   ├── auth/
│   │   ├── base/
│   │   │   ├── kustomization.yaml
│   │   │   ├── deployment.yaml
│   │   │   └── service.yaml
│   │   └── overlays/
│   │       ├── dev/
│   │       ├── staging/
│   │       └── production/
│   ├── users/
│   │   ├── base/
│   │   └── overlays/
│   ├── orders/
│   ├── payments/
│   └── notifications/
└── clusters/
    ├── dev/
    │   └── apps.yaml              # ArgoCD ApplicationSet
    ├── staging/
    │   └── apps.yaml
    └── production/
        └── apps.yaml
```

**Design rationale:**
- **Two repos**: Platform team has write access to infrastructure repo only; app teams have write access to apps repo only. This enforces separation of concerns.
- **Kustomize overlays**: Each app uses `base/` for shared manifests and `overlays/<env>/` for environment-specific patches (replicas, resource limits, image tags).
- **Separate clusters**: Dev and staging share a cluster (different namespaces), production is isolated. ArgoCD ApplicationSets in each cluster directory generate Applications for all apps.
- **ApplicationSet**: One ApplicationSet per cluster generates Applications for all 5 services, reducing YAML duplication.

</details>

### Exercise 2: Drift Scenario

An engineer SSHs into a production Kubernetes cluster during an incident and runs:
```bash
kubectl scale deployment orders --replicas=20 -n production
```

Describe what happens next in a GitOps setup with ArgoCD (self-heal enabled), and explain the correct way to handle this situation.

<details>
<summary>Show Answer</summary>

**What happens with self-heal enabled:**

1. The engineer's `kubectl scale` command immediately sets the deployment to 20 replicas in the cluster.
2. Within ArgoCD's sync interval (default: 3 minutes), the Application Controller detects that the actual state (20 replicas) differs from the desired state in Git (e.g., 4 replicas).
3. ArgoCD marks the application as `OutOfSync`.
4. With `selfHeal: true`, ArgoCD automatically syncs the application back to the Git-declared state, scaling the deployment back down to 4 replicas.
5. The engineer's change is effectively reverted within minutes.

**Why this is problematic during an incident:**
The engineer scaled up to handle increased load, but GitOps reverted the fix. This is the correct behavior from a GitOps perspective (Git is the source of truth), but it creates friction during incidents.

**The correct approach:**

**Option 1 (recommended): Update Git**
```bash
# In the gitops-apps repo, update the production overlay:
cd apps/orders/overlays/production
# Edit patch-replicas.yaml to set replicas: 20
git add . && git commit -m "incident: scale orders to 20 replicas"
git push origin main
# ArgoCD syncs the change within minutes
```

**Option 2: Temporarily disable self-heal**
```bash
# Disable self-heal for the duration of the incident
argocd app set orders-production --self-heal=false
# Now kubectl changes persist
kubectl scale deployment orders --replicas=20 -n production
# After the incident, re-enable self-heal and commit the final state to Git
argocd app set orders-production --self-heal=true
```

**Option 3: Use `ignoreDifferences` for HPA**
If the orders service should be managed by HPA (autoscaler), add `/spec/replicas` to `ignoreDifferences` so ArgoCD ignores replica count drift. This is the best long-term solution for services that need dynamic scaling.

</details>

### Exercise 3: ArgoCD Application Configuration

Write an ArgoCD Application manifest for a Helm chart deployment with the following requirements:
- Chart: `bitnami/postgresql` version 13.4.0
- Target namespace: `database`
- Values override: `primary.persistence.size: 100Gi`, `auth.database: myapp`
- Auto-sync with self-heal and prune
- Sync wave: -1 (deploy before application services)

<details>
<summary>Show Answer</summary>

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: postgresql
  namespace: argocd
  annotations:
    argocd.argoproj.io/sync-wave: "-1"
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: default
  source:
    repoURL: https://charts.bitnami.com/bitnami
    chart: postgresql
    targetRevision: 13.4.0
    helm:
      values: |
        primary:
          persistence:
            size: 100Gi
          resources:
            requests:
              memory: 512Mi
              cpu: 250m
            limits:
              memory: 1Gi
              cpu: 1000m
        auth:
          database: myapp
  destination:
    server: https://kubernetes.default.svc
    namespace: database
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
    retry:
      limit: 3
      backoff:
        duration: 10s
        factor: 2
        maxDuration: 3m
```

**Key points:**
- `source.chart` and `source.repoURL` point to the Helm chart repository (not a Git repo).
- `targetRevision` pins the chart version to `13.4.0` for reproducibility.
- `helm.values` provides inline value overrides (alternatively, use `valuesFiles` to reference a file in a Git repo).
- `sync-wave: "-1"` ensures PostgreSQL is deployed before application services (which default to wave 0).
- `CreateNamespace=true` creates the `database` namespace if it does not exist.

</details>

### Exercise 4: Multi-Environment Promotion

Design a CI/CD pipeline that promotes a new version of a service through dev, staging, and production using GitOps. Include the following:
1. How the CI pipeline builds and pushes a new image
2. How the image tag is updated in the GitOps repo for each environment
3. What gates exist between environments

<details>
<summary>Show Answer</summary>

**Pipeline flow:**

```
Developer push → CI Build → Dev auto-deploy → Staging (manual gate) → Production (approval + canary)
```

**Step 1: CI Pipeline (application repo)**
```yaml
# .github/workflows/ci.yml (in the application repo)
name: CI
on:
  push:
    branches: [main]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build and push Docker image
        run: |
          docker build -t registry.example.com/webapp:${{ github.sha }} .
          docker push registry.example.com/webapp:${{ github.sha }}
      - name: Run tests
        run: make test
      - name: Update dev image tag in GitOps repo
        run: |
          git clone https://github.com/org/gitops-apps.git
          cd gitops-apps/apps/webapp/overlays/dev
          kustomize edit set image webapp=registry.example.com/webapp:${{ github.sha }}
          git add . && git commit -m "deploy webapp ${{ github.sha }} to dev"
          git push
```

**Step 2: Dev auto-deploy**
- ArgoCD watches `gitops-apps` repo and auto-syncs the `dev` overlay.
- Dev environment is updated within minutes with no human intervention.

**Step 3: Staging promotion (manual gate)**
```yaml
# .github/workflows/promote-staging.yml (in gitops-apps repo)
name: Promote to Staging
on:
  workflow_dispatch:
    inputs:
      sha:
        description: "Image SHA to promote"
        required: true
jobs:
  promote:
    runs-on: ubuntu-latest
    environment: staging    # Requires environment protection rules
    steps:
      - uses: actions/checkout@v4
      - name: Update staging image tag
        run: |
          cd apps/webapp/overlays/staging
          kustomize edit set image webapp=registry.example.com/webapp:${{ inputs.sha }}
          git add . && git commit -m "promote webapp ${{ inputs.sha }} to staging"
          git push
```

**Step 4: Production promotion (approval + canary)**
```yaml
name: Promote to Production
on:
  workflow_dispatch:
    inputs:
      sha:
        description: "Image SHA to promote"
        required: true
jobs:
  promote:
    runs-on: ubuntu-latest
    environment: production   # Requires 2 approvals + successful staging tests
    steps:
      - name: Verify staging health
        run: |
          # Check staging metrics (error rate < 0.1%, p99 < 500ms)
          curl -s http://prometheus/api/v1/query?query=... | jq ...
      - uses: actions/checkout@v4
      - name: Update production image tag
        run: |
          cd apps/webapp/overlays/production
          kustomize edit set image webapp=registry.example.com/webapp:${{ inputs.sha }}
          git add . && git commit -m "promote webapp ${{ inputs.sha }} to production"
          git push
```

**Gates between environments:**

| Transition | Gate | Description |
|-----------|------|-------------|
| Dev → Staging | Manual trigger | Engineer initiates promotion after dev testing |
| Staging → Production | 2 approvals + automated checks | Requires 2 team members to approve; CI verifies staging metrics are healthy |
| Production rollback | `git revert` | Revert the promotion commit; ArgoCD syncs previous version |

</details>

---

## References

- [ArgoCD Documentation](https://argo-cd.readthedocs.io/)
- [Flux Documentation](https://fluxcd.io/docs/)
- [OpenGitOps Principles](https://opengitops.dev/)
- [Kustomize Documentation](https://kustomize.io/)
- [Weaveworks GitOps Guide](https://www.weave.works/technologies/gitops/)

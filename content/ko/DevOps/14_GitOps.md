# GitOps

**이전**: [배포 전략](./13_Deployment_Strategies.md) | **다음**: [Secrets Management](./15_Secrets_Management.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. GitOps 원칙을 정의하고 기존 CI/CD push 기반 배포와의 차이점 설명하기
2. Pull 기반(ArgoCD, Flux)과 push 기반(Jenkins, GitHub Actions) 배포 모델 비교하기
3. ArgoCD를 Application CRD, 동기화 정책, 자동화된 롤백과 함께 배포 및 구성하기
4. GitRepository와 Kustomization 리소스로 지속적 배포를 위한 Flux 설정하기
5. 클러스터가 항상 Git의 원하는 상태와 일치하도록 드리프트 감지 및 자가 치유 구현하기
6. dev, staging, production 간 프로모션 파이프라인을 갖춘 멀티 환경 GitOps 워크플로우 설계하기

---

GitOps는 Git을 선언적 인프라 및 애플리케이션 설정의 단일 진실 소스(single source of truth)로 사용하는 운영 프레임워크입니다. CI 파이프라인이나 엔지니어의 노트북에서 `kubectl apply`를 실행하는 대신, GitOps 오퍼레이터가 Git 레포지토리를 지속적으로 감시하고 클러스터 상태를 Git에 선언된 것과 일치하도록 자동으로 조정합니다. 누군가 클러스터에 수동으로 변경을 가하면, 오퍼레이터가 드리프트를 감지하고 이를 되돌립니다. 이 모델은 버전 관리의 이점 -- 이력, 감사 추적, 동료 검토, 롤백 -- 을 인프라 운영에 가져옵니다.

> **비유 -- 온도 조절기 vs 수동 난방**: 전통적인 배포는 히터를 수동으로 켜고 끄는 것과 같습니다 -- 명령을 push하고, 온도가 맞기를 바라며, 드리프트가 발생하면 개입해야 합니다. GitOps는 온도 조절기와 같습니다: 원하는 온도(Git)를 선언하면 시스템이 이를 유지하기 위해 지속적으로 조정합니다. 누군가 창문을 열면(수동 클러스터 변경), 온도 조절기(GitOps 오퍼레이터)가 드리프트를 감지하고 자동으로 보상합니다.

## 1. GitOps 원칙

### 1.1 네 가지 원칙

| 원칙 | 설명 |
|------|------|
| **선언적** | 전체 시스템이 선언적으로 기술됨 (Kubernetes 매니페스트, Helm 차트, Kustomize) |
| **버전 관리 및 불변** | 원하는 상태가 Git에 저장되어 완전한 감사 추적과 되돌리기 가능 |
| **자동 풀** | 소프트웨어 에이전트(ArgoCD, Flux)가 Git에서 원하는 상태를 풀하여 클러스터에 적용 |
| **지속적 조정** | 에이전트가 실제 상태와 원하는 상태를 지속적으로 비교하고 드리프트를 수정 |

### 1.2 Push 기반 vs Pull 기반 배포

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

### 1.3 Pull 기반 GitOps의 이점

| 이점 | 설명 |
|------|------|
| **보안** | CI 시스템에 클러스터 자격 증명이 필요 없음. 오퍼레이터가 제한된 RBAC로 클러스터 내부에서 실행. |
| **감사 추적** | 모든 변경이 작성자, 타임스탬프, 검토 이력이 있는 Git 커밋. |
| **드리프트 감지** | 오퍼레이터가 지속적으로 조정; 수동 `kubectl` 변경은 되돌려짐. |
| **롤백** | 커밋을 `git revert`하면 오퍼레이터가 이전 상태를 복원. |
| **재해 복구** | 오퍼레이터를 Git 레포에 연결하여 클러스터를 처음부터 재구축. |
| **컴플라이언스** | 모든 변경이 필수 검토와 CI 검사가 있는 풀 리퀘스트를 통과. |

---

## 2. 레포지토리 구조

### 2.1 Monorepo vs Multi-Repo

| 패턴 | 구조 | 장점 | 단점 |
|------|------|------|------|
| **Monorepo** | 모든 환경이 하나의 레포에 | 환경 간 비교 용이, 프로모션을 위한 단일 PR | 권한 범위 설정 어려움, 대형 레포 |
| **Multi-repo** | 환경 또는 앱별 별도 레포 | 세밀한 접근 제어, 격리된 영향 범위 | 조율 어려움, 관리할 레포 증가 |
| **Hybrid** | 앱 레포 + 환경 레포 | 앱 팀이 매니페스트 소유, 플랫폼 팀이 환경 설정 소유 | 실무에서 가장 일반적 |

### 2.2 권장 디렉토리 레이아웃

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

### 2.3 Kustomize Base와 Overlay

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

### 3.1 아키텍처

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

### 3.2 설치

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

### 3.4 동기화 정책

| 정책 | 동작 |
|------|------|
| **수동 동기화** | 엔지니어가 UI에서 "Sync"를 클릭하거나 `argocd app sync`를 실행. 기본 모드. |
| **자동 동기화** | Git 변경 시 ArgoCD가 자동으로 동기화. `syncPolicy.automated` 설정. |
| **자가 치유** | 수동 클러스터 변경을 되돌림. `automated.selfHeal: true` 설정. |
| **Prune** | 클러스터에 있지만 Git에 없는 리소스 삭제. `automated.prune: true` 설정. |
| **Sync wave** | `argocd.argoproj.io/sync-wave` 어노테이션으로 동기화 순서 제어. 낮은 번호가 먼저 동기화. |

### 3.5 Sync Wave (순서 제어)

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

### 3.6 ApplicationSet (멀티 환경)

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

### 3.7 롤백

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

### 4.1 아키텍처

Flux v2는 GitOps 파이프라인의 특정 부분을 각각 담당하는 전문화된 컨트롤러 세트를 사용합니다:

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

### 4.2 Flux 부트스트랩

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

### 4.3 GitRepository와 Kustomization

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

### 4.4 이미지 자동화 (태그 자동 업데이트)

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

## 5. 드리프트 감지와 자가 치유

### 5.1 드리프트란

드리프트는 실제 클러스터 상태가 Git의 원하는 상태와 달라질 때 발생합니다. 일반적인 원인:

| 원인 | 예시 | 위험 |
|------|------|------|
| **수동 kubectl** | 인시던트 중 엔지니어가 `kubectl scale` 실행 | 문서화되지 않은 변경, 다음 동기화와 충돌 가능 |
| **직접 API 접근** | 대시보드나 오퍼레이터가 리소스 수정 | 검토 프로세스 우회 |
| **컨트롤러 변경** | HPA가 레플리카 수 변경, cert-manager가 시크릿 업데이트 | 예상되는 드리프트 (무시해야 함) |
| **동기화 실패** | ArgoCD/Flux가 오류로 인해 부분적으로만 적용 | 절반만 배포된 상태 |

### 5.2 ArgoCD 드리프트 감지

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

### 5.3 예상되는 드리프트 무시

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

## 6. 멀티 환경 관리

### 6.1 프로모션 파이프라인

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

### 6.2 Kustomize Overlay를 사용한 구현

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

### 6.3 CI를 사용한 자동화된 프로모션

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

## 7. ArgoCD vs Flux 비교

| 기능 | ArgoCD | Flux |
|------|--------|------|
| **UI** | 시각화가 포함된 풍부한 웹 UI | 내장 UI 없음 (Weave GitOps 사용) |
| **멀티 클러스터** | 내장된 멀티 클러스터 관리 | 추가 설정 필요 |
| **동기화 모델** | Application CRD, ApplicationSet | GitRepository + Kustomization CRD |
| **Helm 지원** | 네이티브 Helm 렌더링 | HelmRelease CRD |
| **이미지 자동화** | Argo CD Image Updater 필요 | 내장된 Image Automation Controller |
| **RBAC** | 프로젝트와 역할이 있는 내장 RBAC | Kubernetes RBAC |
| **학습 곡선** | 낮음 (UI 기반) | 높음 (CRD 기반, UI 없음) |
| **커뮤니티** | 대규모 커뮤니티, CNCF graduated | CNCF graduated, 강력한 Kubernetes 네이티브 초점 |

---

## 8. 다음 단계

- [15_Secrets_Management.md](./15_Secrets_Management.md) - GitOps 워크플로우에서의 시크릿 관리
- [13_Deployment_Strategies.md](./13_Deployment_Strategies.md) - GitOps를 보완하는 배포 전략

---

## 연습 문제

### 연습 문제 1: GitOps 레포지토리 설계

다음 설정을 가진 회사를 위한 GitOps 레포지토리 구조를 설계하십시오:
- 3개 환경: dev, staging, production
- 5개 마이크로서비스: auth, users, orders, payments, notifications
- production은 dev/staging과 별도의 클러스터에서 실행
- 플랫폼 팀이 인프라(ingress, monitoring, cert-manager)를 관리
- 애플리케이션 팀이 자체 서비스 매니페스트를 관리

<details>
<summary>정답 보기</summary>

**권장 구조 (하이브리드 멀티 레포):**

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

**설계 근거:**
- **두 개의 레포**: 플랫폼 팀은 인프라 레포에만 쓰기 권한; 앱 팀은 앱 레포에만 쓰기 권한. 이것은 관심사의 분리를 강제합니다.
- **Kustomize overlay**: 각 앱은 공유 매니페스트를 위한 `base/`와 환경별 패치(레플리카, 리소스 한도, 이미지 태그)를 위한 `overlays/<env>/`를 사용합니다.
- **별도 클러스터**: dev와 staging은 클러스터를 공유(다른 네임스페이스)하고, production은 격리됩니다. 각 클러스터 디렉토리의 ArgoCD ApplicationSet이 모든 앱에 대한 Application을 생성합니다.
- **ApplicationSet**: 클러스터당 하나의 ApplicationSet이 5개 서비스 모두에 대한 Application을 생성하여 YAML 중복을 줄입니다.

</details>

### 연습 문제 2: 드리프트 시나리오

엔지니어가 인시던트 중 프로덕션 Kubernetes 클러스터에 SSH로 접속하여 다음을 실행합니다:
```bash
kubectl scale deployment orders --replicas=20 -n production
```

ArgoCD(자가 치유 활성화)가 설정된 GitOps 환경에서 다음에 무슨 일이 발생하는지 설명하고, 이 상황을 처리하는 올바른 방법을 설명하십시오.

<details>
<summary>정답 보기</summary>

**자가 치유가 활성화된 상태에서 일어나는 일:**

1. 엔지니어의 `kubectl scale` 명령이 클러스터에서 즉시 배포를 20개 레플리카로 설정합니다.
2. ArgoCD의 동기화 간격(기본: 3분) 이내에 Application Controller가 실제 상태(20 레플리카)가 Git의 원하는 상태(예: 4 레플리카)와 다르다는 것을 감지합니다.
3. ArgoCD가 애플리케이션을 `OutOfSync`로 표시합니다.
4. `selfHeal: true`로 설정되어 있으므로, ArgoCD가 자동으로 애플리케이션을 Git 선언 상태로 동기화하여 배포를 4 레플리카로 다시 축소합니다.
5. 엔지니어의 변경은 몇 분 이내에 사실상 되돌려집니다.

**인시던트 중에 이것이 문제인 이유:**
엔지니어가 증가된 부하를 처리하기 위해 스케일 업했지만, GitOps가 수정 사항을 되돌렸습니다. 이것은 GitOps 관점에서 올바른 동작(Git이 진실의 원천)이지만, 인시던트 중에 마찰을 발생시킵니다.

**올바른 접근 방법:**

**옵션 1 (권장): Git 업데이트**
```bash
# In the gitops-apps repo, update the production overlay:
cd apps/orders/overlays/production
# Edit patch-replicas.yaml to set replicas: 20
git add . && git commit -m "incident: scale orders to 20 replicas"
git push origin main
# ArgoCD syncs the change within minutes
```

**옵션 2: 임시로 자가 치유 비활성화**
```bash
# Disable self-heal for the duration of the incident
argocd app set orders-production --self-heal=false
# Now kubectl changes persist
kubectl scale deployment orders --replicas=20 -n production
# After the incident, re-enable self-heal and commit the final state to Git
argocd app set orders-production --self-heal=true
```

**옵션 3: HPA를 위한 `ignoreDifferences` 사용**
orders 서비스가 HPA(오토스케일러)에 의해 관리되어야 한다면, `ignoreDifferences`에 `/spec/replicas`를 추가하여 ArgoCD가 레플리카 수 드리프트를 무시하도록 합니다. 이것이 동적 스케일링이 필요한 서비스를 위한 최선의 장기 솔루션입니다.

</details>

### 연습 문제 3: ArgoCD Application 설정

다음 요구사항을 가진 Helm 차트 배포를 위한 ArgoCD Application 매니페스트를 작성하십시오:
- 차트: `bitnami/postgresql` 버전 13.4.0
- 대상 네임스페이스: `database`
- 값 오버라이드: `primary.persistence.size: 100Gi`, `auth.database: myapp`
- 자가 치유와 prune이 있는 자동 동기화
- Sync wave: -1 (애플리케이션 서비스보다 먼저 배포)

<details>
<summary>정답 보기</summary>

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

**핵심 포인트:**
- `source.chart`와 `source.repoURL`이 Helm 차트 레포지토리를 가리킵니다 (Git 레포가 아님).
- `targetRevision`이 재현성을 위해 차트 버전을 `13.4.0`으로 고정합니다.
- `helm.values`가 인라인 값 오버라이드를 제공합니다 (대안으로 `valuesFiles`를 사용하여 Git 레포의 파일을 참조 가능).
- `sync-wave: "-1"`이 PostgreSQL이 애플리케이션 서비스(기본 wave 0)보다 먼저 배포되도록 보장합니다.
- `CreateNamespace=true`가 `database` 네임스페이스가 없으면 생성합니다.

</details>

### 연습 문제 4: 멀티 환경 프로모션

GitOps를 사용하여 서비스의 새 버전을 dev, staging, production을 통해 프로모션하는 CI/CD 파이프라인을 설계하십시오. 다음을 포함하십시오:
1. CI 파이프라인이 새 이미지를 빌드하고 푸시하는 방법
2. 각 환경의 GitOps 레포에서 이미지 태그가 업데이트되는 방법
3. 환경 간에 존재하는 게이트

<details>
<summary>정답 보기</summary>

**파이프라인 흐름:**

```
Developer push → CI Build → Dev auto-deploy → Staging (manual gate) → Production (approval + canary)
```

**Step 1: CI 파이프라인 (애플리케이션 레포)**
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

**Step 2: Dev 자동 배포**
- ArgoCD가 `gitops-apps` 레포를 감시하고 `dev` overlay를 자동 동기화합니다.
- 사람의 개입 없이 수 분 이내에 dev 환경이 업데이트됩니다.

**Step 3: Staging 프로모션 (수동 게이트)**
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

**Step 4: Production 프로모션 (승인 + canary)**
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

**환경 간 게이트:**

| 전환 | 게이트 | 설명 |
|------|--------|------|
| Dev -> Staging | 수동 트리거 | 엔지니어가 dev 테스트 후 프로모션 시작 |
| Staging -> Production | 2명 승인 + 자동화된 검사 | 2명의 팀원 승인 필요; CI가 staging 메트릭이 정상인지 확인 |
| Production 롤백 | `git revert` | 프로모션 커밋을 되돌림; ArgoCD가 이전 버전 동기화 |

</details>

---

## 참고 자료

- [ArgoCD Documentation](https://argo-cd.readthedocs.io/)
- [Flux Documentation](https://fluxcd.io/docs/)
- [OpenGitOps Principles](https://opengitops.dev/)
- [Kustomize Documentation](https://kustomize.io/)
- [Weaveworks GitOps Guide](https://www.weave.works/technologies/gitops/)

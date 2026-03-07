# Platform Engineering

**이전**: [Chaos Engineering](./16_Chaos_Engineering.md) | **다음**: [SRE Practices](./18_SRE_Practices.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Platform engineering을 정의하고 내부 개발자 플랫폼(IDP)이 개발 팀의 인지 부하를 어떻게 줄이는지 설명하기
2. Backstage 프레임워크와 핵심 컴포넌트(소프트웨어 카탈로그, 템플릿, 플러그인)를 설명하기
3. 인프라 프로비저닝, 서비스 생성, 애플리케이션 배포를 위한 개발자 셀프서비스 워크플로우 설계하기
4. 팀이 소프트웨어를 빌드하고 배포하는 방식을 표준화하면서 유연성을 유지하는 Golden Path 정의하기
5. 성숙도 모델을 사용하여 플랫폼 성숙도를 평가하고 점진적 개선 계획 수립하기
6. Platform engineering을 DevOps 및 SRE와 구별하고 서로 어떻게 보완하는지 이해하기

---

Platform engineering은 클라우드 네이티브 시대에 개발자 셀프서비스를 가능하게 하는 도구 체인과 워크플로우를 설계하고 구축하는 분야입니다. 모든 팀이 자체 CI/CD 파이프라인, Kubernetes 매니페스트, 모니터링 대시보드, 시크릿 관리를 처음부터 구축하는 대신, 플랫폼 팀이 큐레이션되고, 의견이 반영되며, 자동화된 기반 -- 내부 개발자 플랫폼(IDP) -- 을 제공하여 인지 부하를 줄이고 딜리버리를 가속화합니다.

> **비유 -- 공장 바닥**: 모든 제품 팀이 자체 조립 라인을 처음부터 구축해야 하는 공장을 상상해 보십시오: 재료를 조달하고, 컨베이어를 설계하고, 품질 검사를 연결하고, 배송을 파악합니다. 대부분의 시간이 조립 라인을 구축하는 데 들어가지 제품을 만드는 데 들어가지 않습니다. Platform engineering은 공장 바닥 자체입니다: 모든 제품 팀이 사용하는 표준화된 조립 라인, 공유 도구, 공통 인프라. 팀은 무엇을 만들 것인지(제품)에 집중하지, 공장을 어떻게 만들 것인지에 집중하지 않습니다.

## 1. Platform Engineering의 필요성

### 1.1 해결하는 문제

```
Without a Platform:                    With a Platform:
┌─────────────────────┐               ┌─────────────────────┐
│  Team A             │               │  Team A             │
│  - Own CI/CD        │               │  - Use platform     │
│  - Own K8s configs  │               │    templates        │
│  - Own monitoring   │               │  - Self-service     │
│  - Own secrets mgmt │               │    provisioning     │
│  Time: 70% infra    │               │  Time: 90% product  │
│        30% product  │               │        10% infra    │
├─────────────────────┤               ├─────────────────────┤
│  Team B             │               │  Team B             │
│  - Different CI/CD  │               │  - Same platform    │
│  - Different configs│               │  - Same standards   │
│  - Different tools  │               │  - Same guardrails  │
│  Time: 70% infra    │               │  Time: 90% product  │
│        30% product  │               │        10% infra    │
└─────────────────────┘               └─────────────────────┘

Result: Inconsistency,               Result: Consistency,
        duplication,                          velocity,
        cognitive overload                    developer joy
```

### 1.2 Platform Engineering vs DevOps vs SRE

| 측면 | DevOps | SRE | Platform Engineering |
|------|--------|-----|---------------------|
| **초점** | 문화, 협업, 자동화 | 신뢰성, SLO, 에러 예산 | 개발자 경험, 셀프서비스 |
| **누가** | 모든 사람 (문화 운동) | 전담 SRE 팀 | 전담 플랫폼 팀 |
| **산출물** | 관행과 프로세스 | 신뢰성 표준 | 내부 개발자 플랫폼 (제품) |
| **인프라 취급** | 코드 (IaC) | 신뢰할 수 있게 만들 시스템 | 개발자가 사용할 제품 |
| **핵심 메트릭** | 배포 빈도, 리드 타임 | 가용성, 에러 예산 | 개발자 만족도, 프로덕션 투입 시간 |

### 1.3 인지 부하 문제

연구에 따르면 인지 부하가 증가하면 개발자 효율성이 떨어집니다:

| 인지 부하 유형 | 예시 | 플랫폼 솔루션 |
|-------------|------|-------------|
| **Intrinsic** | Kubernetes 개념 학습 | 추상화 계층, Golden Path |
| **Extraneous** | 배포를 위해 5개 도구 탐색 | 통합 개발자 포탈 |
| **Germane** | 비즈니스 도메인 로직 이해 | Extraneous 부하를 줄여 개발자가 여기에 집중 |

---

## 2. 내부 개발자 플랫폼 (IDP)

### 2.1 IDP 컴포넌트

```
┌─────────────────────────────────────────────────────────────────┐
│                Internal Developer Platform                       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Developer Portal (Backstage)                             │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐           │   │
│  │  │  Software  │ │ Templates  │ │  TechDocs  │           │   │
│  │  │  Catalog   │ │ (Scaffolder│ │            │           │   │
│  │  │            │ │  / Create) │ │            │           │   │
│  │  └────────────┘ └────────────┘ └────────────┘           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Platform Services                                        │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐│   │
│  │  │CI/CD   │ │Infra   │ │Secrets │ │Monitor-│ │Service │ │   │
│  │  │Pipeline│ │Provisio│ │Mgmt    │ │ing     │ │Mesh    │ │   │
│  │  │(GHA/   │ │ning    │ │(Vault) │ │(Prom/  │ │(Istio) │ │   │
│  │  │ArgoCD) │ │(Tf/    │ │        │ │Grafana)│ │        │ │   │
│  │  │        │ │Crosspl)│ │        │ │        │ │        │ │   │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘│   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Infrastructure (Kubernetes, Cloud, Network)              │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 IDP 기능

| 기능 | 설명 | 예시 |
|------|------|------|
| **서비스 카탈로그** | 모든 서비스, 소유자, 의존성, API 문서의 레지스트리 | Backstage Software Catalog |
| **셀프서비스 프로비저닝** | UI/CLI를 통한 새 서비스, 데이터베이스, 환경 생성 | Backstage 템플릿, Terraform 모듈 |
| **CI/CD** | 표준화된 빌드 및 배포 파이프라인 | GitHub Actions 템플릿, ArgoCD |
| **환경 관리** | 임시 환경의 생성, 관리, 제거 | PR별 네임스페이스, 프리뷰 환경 |
| **관측성** | 통합 모니터링, 로깅, 트레이싱 | 자동 프로비저닝된 Grafana 대시보드 |
| **문서화** | 중앙화되고, 검색 가능하며, 버전 관리되는 문서 | Backstage TechDocs (docs-as-code) |
| **보안** | 자동화된 시크릿 관리, 취약점 스캐닝 | Vault 통합, Trivy 스캐닝 |
| **비용 관리** | 팀/서비스별 리소스 비용 가시성 | 클라우드 비용 대시보드 |

---

## 3. Backstage

### 3.1 Backstage란

Backstage는 Spotify가 만들고 CNCF에 기부한 개발자 포탈 구축을 위한 오픈소스 플랫폼입니다. 모든 인프라 도구를 위한 통합 프론트엔드를 제공합니다.

### 3.2 핵심 컴포넌트

```
┌─────────────────────────────────────────────────────────────────┐
│                       Backstage Components                       │
│                                                                  │
│  1. Software Catalog                                            │
│     └── Central registry of all software (services, libraries,  │
│         APIs, resources) with ownership and metadata            │
│                                                                  │
│  2. Software Templates (Scaffolder)                              │
│     └── Create new projects from templates with automated       │
│         repo creation, CI/CD setup, and infrastructure          │
│                                                                  │
│  3. TechDocs                                                     │
│     └── Docs-as-code: Markdown docs rendered in the portal,     │
│         version-controlled alongside source code                │
│                                                                  │
│  4. Plugins                                                      │
│     └── Extensible plugin architecture: Kubernetes, CI/CD,      │
│         PagerDuty, Grafana, ArgoCD, and 200+ community plugins │
│                                                                  │
│  5. Search                                                       │
│     └── Unified search across catalog, docs, and all plugins    │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Software Catalog

카탈로그는 각 서비스의 레포지토리에 있는 `catalog-info.yaml` 파일로 채워집니다:

```yaml
# catalog-info.yaml (in the service's Git repo root)
apiVersion: backstage.io/v1alpha1
kind: Component
metadata:
  name: order-service
  description: "Handles order creation, management, and fulfillment"
  annotations:
    github.com/project-slug: org/order-service
    backstage.io/techdocs-ref: dir:.
    pagerduty.com/service-id: P1234567
    grafana/dashboard-selector: "order-service"
    argocd/app-name: order-service-production
  tags:
    - python
    - grpc
    - tier-1
  links:
    - url: https://grafana.internal/d/order-service
      title: Grafana Dashboard
    - url: https://runbooks.internal/order-service
      title: Runbook
spec:
  type: service
  lifecycle: production
  owner: team-commerce
  system: e-commerce-platform
  providesApis:
    - order-api
  consumesApis:
    - inventory-api
    - payment-api
  dependsOn:
    - resource:order-db
    - resource:order-cache
```

### 3.4 Software Templates (Scaffolder)

템플릿은 모든 플랫폼 통합이 사전 구성된 새 서비스의 생성을 자동화합니다:

```yaml
# templates/python-service/template.yaml
apiVersion: scaffolder.backstage.io/v1beta3
kind: Template
metadata:
  name: python-microservice
  title: Python Microservice
  description: "Create a new Python microservice with CI/CD, monitoring, and GitOps"
  tags:
    - python
    - recommended
spec:
  owner: platform-team
  type: service

  parameters:
    - title: Service Information
      required:
        - name
        - description
        - owner
      properties:
        name:
          title: Service Name
          type: string
          pattern: "^[a-z][a-z0-9-]*$"
          description: "Lowercase, hyphen-separated (e.g., order-service)"
        description:
          title: Description
          type: string
        owner:
          title: Owner Team
          type: string
          ui:field: OwnerPicker
          ui:options:
            catalogFilter:
              kind: Group

    - title: Infrastructure
      properties:
        database:
          title: Database
          type: string
          enum: ["none", "postgresql", "redis", "both"]
          default: "none"
        port:
          title: Service Port
          type: number
          default: 8080

  steps:
    # Step 1: Generate project from skeleton
    - id: fetch
      name: Fetch Skeleton
      action: fetch:template
      input:
        url: ./skeleton
        values:
          name: ${{ parameters.name }}
          description: ${{ parameters.description }}
          owner: ${{ parameters.owner }}
          port: ${{ parameters.port }}
          database: ${{ parameters.database }}

    # Step 2: Create GitHub repository
    - id: publish
      name: Create Repository
      action: publish:github
      input:
        repoUrl: github.com?owner=org&repo=${{ parameters.name }}
        description: ${{ parameters.description }}
        defaultBranch: main
        protectDefaultBranch: true

    # Step 3: Register in Backstage catalog
    - id: register
      name: Register in Catalog
      action: catalog:register
      input:
        repoContentsUrl: ${{ steps.publish.output.repoContentsUrl }}
        catalogInfoPath: /catalog-info.yaml

    # Step 4: Create ArgoCD application
    - id: argocd
      name: Create ArgoCD Application
      action: argocd:create-resources
      input:
        appName: ${{ parameters.name }}
        argoInstance: production
        path: apps/${{ parameters.name }}/overlays/production
        repoUrl: https://github.com/org/gitops-apps.git

    # Step 5: Provision database (if selected)
    - id: provision-db
      name: Provision Database
      if: ${{ parameters.database !== 'none' }}
      action: http:backstage:request
      input:
        method: POST
        path: /api/proxy/terraform/apply
        body:
          module: ${{ parameters.database }}
          service: ${{ parameters.name }}

  output:
    links:
      - title: Repository
        url: ${{ steps.publish.output.remoteUrl }}
      - title: Open in Backstage
        icon: catalog
        entityRef: ${{ steps.register.output.entityRef }}
```

### 3.5 TechDocs

```yaml
# mkdocs.yml (in the service repo root)
site_name: Order Service
nav:
  - Home: index.md
  - Architecture: architecture.md
  - API Reference: api.md
  - Runbook: runbook.md
  - ADRs:
      - ADR-001 Database Choice: adrs/001-database-choice.md
      - ADR-002 Event Sourcing: adrs/002-event-sourcing.md

plugins:
  - techdocs-core
```

TechDocs는 Backstage 포탈에서 Markdown 문서를 직접 렌더링하여 소프트웨어 카탈로그와 함께 검색할 수 있게 합니다.

---

## 4. Golden Path

### 4.1 Golden Path란

Golden Path는 일반적인 작업을 수행하기 위한 의견이 반영되고, 지원되며, 포장된 방법입니다. 최소 저항의 경로이면서 동시에 모범 사례이기도 합니다. 개발자는 자유롭게 이탈할 수 있지만, Golden Path가 너무 편리하기 때문에 대부분이 따르기를 선택합니다.

```
Golden Path:                          Off-Road:
┌─────────────────────────────┐      ┌─────────────────────────────┐
│  "I need a new service"     │      │  "I need a new service"     │
│       ↓                     │      │       ↓                     │
│  Backstage → New Service    │      │  Manual repo creation       │
│  template (2 minutes)       │      │  Write Dockerfile           │
│       ↓                     │      │  Write K8s manifests        │
│  Auto: repo, CI/CD, K8s,   │      │  Set up CI/CD               │
│  monitoring, ArgoCD, docs   │      │  Configure monitoring       │
│       ↓                     │      │  Set up secrets             │
│  First deploy: 15 minutes   │      │  Debug networking           │
│                             │      │       ↓                     │
│  Supported by platform team │      │  First deploy: 2 days       │
│  Updates pushed centrally   │      │                             │
│                             │      │  Supported by: yourself     │
└─────────────────────────────┘      └─────────────────────────────┘
```

### 4.2 일반적인 Golden Path

| Golden Path | 제공하는 것 |
|-------------|-----------|
| **새 서비스** | 스켈레톤이 있는 레포, Dockerfile, CI/CD, K8s 매니페스트, 모니터링, 카탈로그 등록 |
| **새 API** | OpenAPI 스펙 템플릿, API 게이트웨이 설정, Rate limiting, 인증 |
| **새 데이터베이스** | Terraform 모듈, 백업 정책, 모니터링, 시크릿 로테이션 |
| **새 환경** | 네임스페이스 생성, RBAC, 리소스 쿼터, 네트워크 정책 |
| **배포** | PR -> CI -> Staging -> Canary -> Production (자동화) |
| **인시던트 대응** | PagerDuty 알림 -> Slack 채널 -> Runbook -> 포스트모템 템플릿 |

### 4.3 Golden Path 설계 원칙

| 원칙 | 설명 |
|------|------|
| **의견이 있지만 제한적이지 않음** | 기본값 제공, 오버라이드 허용. "이것이 우리의 방식; 필요하면 변경 가능." |
| **셀프서비스** | 개발자가 티켓을 제출하거나 다른 팀을 기다릴 필요 없음 |
| **자동화** | 모든 단계가 자동화되거나 템플릿 기반이어야 함 |
| **문서화** | Golden Path가 무엇을 하고 어떻게 커스터마이즈하는지에 대한 명확한 문서 |
| **유지보수** | 플랫폼 팀이 템플릿과 도구를 최신 상태로 유지 |
| **측정** | 채택률 추적: 얼마나 많은 팀이 Golden Path를 사용하는지 vs 커스텀 솔루션 |

---

## 5. 개발자 셀프서비스

### 5.1 셀프서비스 기능

```
┌─────────────────────────────────────────────────────────────────┐
│              Developer Self-Service Tiers                        │
│                                                                  │
│  Tier 1: Instant (< 5 minutes)                                  │
│  ├── Create a new microservice from template                    │
│  ├── Deploy to development environment                          │
│  ├── View service health and dashboards                         │
│  └── Access logs and traces                                     │
│                                                                  │
│  Tier 2: Automated (< 30 minutes)                               │
│  ├── Provision a PostgreSQL database                            │
│  ├── Create a Redis cache cluster                               │
│  ├── Set up a new staging environment                           │
│  └── Configure custom domain and TLS certificate                │
│                                                                  │
│  Tier 3: Guided (< 1 day)                                       │
│  ├── Request a production deployment pipeline                   │
│  ├── Set up cross-account access                                │
│  ├── Create a new Kubernetes cluster                            │
│  └── Onboard a new team to the platform                         │
│                                                                  │
│  Tier 4: Assisted (requires platform team)                      │
│  ├── Custom infrastructure not covered by templates             │
│  ├── Security exceptions                                        │
│  ├── Multi-region architecture design                           │
│  └── Platform feature requests                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Crossplane을 사용한 인프라 셀프서비스

```yaml
# Crossplane: Kubernetes-native infrastructure provisioning
# Developers create infrastructure by applying K8s resources

# Composite Resource Definition (XRD): platform team defines
apiVersion: apiextensions.crossplane.io/v1
kind: CompositeResourceDefinition
metadata:
  name: xpostgresqlinstances.database.example.org
spec:
  group: database.example.org
  names:
    kind: XPostgreSQLInstance
    plural: xpostgresqlinstances
  claimNames:
    kind: PostgreSQLInstance
    plural: postgresqlinstances
  versions:
    - name: v1alpha1
      served: true
      referenceable: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                parameters:
                  type: object
                  properties:
                    storageGB:
                      type: integer
                      default: 20
                    version:
                      type: string
                      default: "15"
                      enum: ["14", "15", "16"]

---
# Developer creates a database by applying a simple claim
apiVersion: database.example.org/v1alpha1
kind: PostgreSQLInstance
metadata:
  name: order-db
  namespace: production
spec:
  parameters:
    storageGB: 50
    version: "15"
  compositionSelector:
    matchLabels:
      provider: aws
      environment: production
```

---

## 6. 플랫폼 성숙도 모델

### 6.1 플랫폼 성숙도의 5단계

```
Level 1: Provisional                    Level 2: Operational
┌─────────────────────────┐            ┌─────────────────────────┐
│ • Shared scripts        │            │ • Standardized CI/CD    │
│ • Wiki documentation    │            │ • Container platform    │
│ • Manual provisioning   │            │ • Centralized logging   │
│ • Ad-hoc tooling        │            │ • Basic self-service    │
│ • No dedicated team     │            │ • Small platform team   │
└─────────────────────────┘            └─────────────────────────┘

Level 3: Scalable                       Level 4: Optimizing
┌─────────────────────────┐            ┌─────────────────────────┐
│ • Developer portal      │            │ • Golden paths adopted  │
│ • Service catalog       │            │ • Full self-service     │
│ • Template-based        │            │ • Platform as a product │
│   service creation      │            │ • Developer metrics     │
│ • GitOps deployment     │            │ • Cost optimization     │
│ • Dedicated team        │            │ • Platform product mgr  │
└─────────────────────────┘            └─────────────────────────┘

Level 5: Strategic
┌─────────────────────────┐
│ • Platform enables new  │
│   business capabilities │
│ • Innovation accelerator│
│ • Cross-org platform    │
│ • External ecosystem    │
│ • Platform as competitive│
│   advantage             │
└─────────────────────────┘
```

### 6.2 성숙도 평가 기준

| 차원 | Level 1 | Level 3 | Level 5 |
|------|---------|---------|---------|
| **프로비저닝** | 티켓 및 수동 | 템플릿 및 셀프서비스 | API 기반, 즉시 |
| **배포** | 수동 스크립트 | CI/CD 파이프라인 | Canary 및 자동 롤백이 있는 GitOps |
| **관측성** | SSH + grep | 중앙화된 대시보드 | 상관된 메트릭-로그-트레이스 |
| **문서화** | Wiki (오래됨) | Docs-as-code | API 문서가 있는 검색 가능한 포탈 |
| **개발자 경험** | "ops 팀에 문의" | "템플릿 사용" | "그냥 작동함" |
| **채택 측정** | 없음 | 사용 횟수 | NPS 점수, 프로덕션 투입 시간 |

---

## 7. 플랫폼 팀 구조

### 7.1 팀 구성

| 역할 | 책임 |
|------|------|
| **플랫폼 프로덕트 매니저** | 개발자 니즈에 기반한 로드맵 정의, 채택률 측정 |
| **플랫폼 엔지니어** | 플랫폼 서비스(CI/CD, K8s, IaC) 구축 및 유지 |
| **Developer Experience (DX) 엔지니어** | Backstage 개발, 템플릿, 문서화 |
| **SRE / Reliability 엔지니어** | 플랫폼 신뢰성, 플랫폼 서비스의 SLO |
| **보안 엔지니어** | 보안 가드레일, policy-as-code, 컴플라이언스 자동화 |

### 7.2 플랫폼을 제품으로

| 제품 관행 | 플랫폼 적용 |
|----------|-----------|
| **사용자 연구** | 개발자 설문조사, 쉐도잉, 페인포인트 인터뷰 |
| **로드맵** | 개발자 영향에 기반한 우선순위 기능 목록 |
| **MVP** | 하나의 Golden Path로 시작, 피드백에 기반하여 반복 |
| **릴리스 노트** | 새 플랫폼 기능, 마이그레이션, 지원 중단 공지 |
| **지원** | Slack 채널, 오피스 아워, 문서화 |
| **메트릭** | 채택률, 개발자 만족도(NPS), 프로덕션 투입 시간 |

### 7.3 핵심 메트릭

| 메트릭 | 측정하는 것 | 목표 |
|--------|-----------|------|
| **첫 배포까지의 시간** | "아이디어가 있다"에서 프로덕션 실행까지의 시간 | < 1일 |
| **배포 빈도** | 팀이 배포하는 빈도 | 하루에 여러 번 |
| **플랫폼 채택** | 플랫폼을 사용하는 서비스 비율 | > 80% |
| **개발자 NPS** | 플랫폼에 대한 개발자 만족도 | > 50 |
| **셀프서비스 비율** | 플랫폼 팀 개입 없이 처리되는 요청 비율 | > 90% |
| **프로비저닝 평균 시간** | 새 데이터베이스, 환경 등을 프로비저닝하는 시간 | < 30분 |

---

## 8. 다음 단계

- [18_SRE_Practices.md](./18_SRE_Practices.md) - 플랫폼 신뢰성을 위한 SRE 관행
- [14_GitOps.md](./14_GitOps.md) - 플랫폼 배포의 기반인 GitOps

---

## 연습 문제

### 연습 문제 1: 플랫폼 설계

회사에 15개 개발 팀과 60개 마이크로서비스가 있습니다. 현재 각 팀이 자체 CI/CD, Kubernetes 매니페스트, 모니터링, 시크릿을 관리합니다. CEO가 내부 개발자 플랫폼을 구축하라고 요청합니다. 플랫폼 로드맵의 첫 3개월을 설계하십시오.

<details>
<summary>정답 보기</summary>

**1개월차: 기반 (Level 1에서 Level 2)**

| 주 | 산출물 | 영향 |
|---|--------|------|
| 1-2 | **서비스 카탈로그**: Backstage 배포, `catalog-info.yaml`을 통해 60개 서비스 전부 임포트. 모든 서비스에 대한 소유권 확립. | 모든 엔지니어가 한 곳에서 모든 서비스, 소유자, 문서, 대시보드를 찾을 수 있음. |
| 3-4 | **표준화된 CI 파이프라인 템플릿**: 빌드, 테스트, 컨테이너 푸시를 위한 재사용 가능한 GitHub Actions 워크플로우 생성. | 팀이 CI를 처음부터 작성하지 않음. 새 서비스가 몇 분 내에 CI를 얻음. |

**2개월차: Golden Path (Level 2에서 Level 3)**

| 주 | 산출물 | 영향 |
|---|--------|------|
| 5-6 | **서비스 템플릿**: 가장 일반적인 서비스 유형(PostgreSQL이 있는 Python REST API)을 위한 Backstage scaffolder 템플릿. 레포, Dockerfile, K8s 매니페스트, CI/CD, Grafana 대시보드, ArgoCD 앱을 자동 생성. | 아이디어에서 실행까지 30분 미만의 새 서비스. |
| 7-8 | **표준화된 Kubernetes 매니페스트**: 헬스 체크, 리소스 한도, HPA, 네트워크 정책이 있는 Kustomize 베이스. 팀이 레포에 대한 PR을 통해 채택. | 모든 서비스에 걸쳐 일관된 보안, 리소스 관리, 관측성. |

**3개월차: 셀프서비스 (Level 3 강화)**

| 주 | 산출물 | 영향 |
|---|--------|------|
| 9-10 | **데이터베이스 셀프서비스**: PostgreSQL 프로비저닝을 위한 Crossplane 또는 Terraform 모듈. 개발자가 Backstage 템플릿 또는 CLI를 통해 요청. | 데이터베이스 프로비저닝이 2일 티켓에서 15분 셀프서비스로 감소. |
| 11-12 | **관측성 Golden Path**: 카탈로그의 모든 서비스에 대해 자동 프로비저닝된 Grafana 대시보드(4가지 골든 시그널). 표준화된 알림 템플릿. | 모든 서비스가 첫 날부터 모니터링을 보유. "모니터링 추가를 잊었다"가 사라짐. |

**3개월차 성공 메트릭:**
- Backstage 카탈로그에 100% 서비스 등록
- 템플릿을 사용하여 5개 이상의 새 서비스 생성
- 데이터베이스 프로비저닝 시간이 2일에서 15분으로 감소
- 개발자 NPS 기준선 수립 (설문조사)

**1-3개월에 하지 말아야 할 것:**
- 60개 서비스를 한 번에 마이그레이션하지 마십시오. 새 서비스와 의향이 있는 얼리 어답터 팀부터 시작하십시오.
- 처음부터 커스텀 개발자 포탈을 구축하지 마십시오. Backstage를 사용하십시오.
- 채택을 강제하지 마십시오. 팀이 선택하도록 플랫폼을 충분히 편리하게 만드십시오.

</details>

### 연습 문제 2: Backstage 카탈로그 설계

다음 컴포넌트를 가진 전자상거래 시스템의 `catalog-info.yaml` 구조를 설계하십시오:
- API Gateway (Node.js)
- Product Service (Go, Product API 제공)
- Order Service (Python, Order API 제공, Product API와 Payment API 소비)
- Payment Service (Java, Payment API 제공, PostgreSQL과 Redis에 의존)
- 공유 PostgreSQL 데이터베이스
- 공유 Redis 캐시
- 두 팀이 소유: `team-catalog` (Product)와 `team-commerce` (Order, Payment)

<details>
<summary>정답 보기</summary>

```yaml
# --- System Definition ---
apiVersion: backstage.io/v1alpha1
kind: System
metadata:
  name: e-commerce-platform
  description: "E-commerce platform for online retail"
  tags:
    - production
    - revenue-critical
spec:
  owner: team-commerce

---
# --- API Gateway ---
apiVersion: backstage.io/v1alpha1
kind: Component
metadata:
  name: api-gateway
  description: "API Gateway - routes and authenticates all external traffic"
  annotations:
    github.com/project-slug: org/api-gateway
    grafana/dashboard-selector: "api-gateway"
  tags:
    - nodejs
    - tier-0
spec:
  type: service
  lifecycle: production
  owner: team-commerce
  system: e-commerce-platform
  consumesApis:
    - product-api
    - order-api
    - payment-api

---
# --- Product Service ---
apiVersion: backstage.io/v1alpha1
kind: Component
metadata:
  name: product-service
  description: "Product catalog management"
  tags:
    - go
    - tier-1
spec:
  type: service
  lifecycle: production
  owner: team-catalog
  system: e-commerce-platform
  providesApis:
    - product-api
  dependsOn:
    - resource:shared-postgresql
    - resource:shared-redis

---
# --- Order Service ---
apiVersion: backstage.io/v1alpha1
kind: Component
metadata:
  name: order-service
  description: "Order creation and fulfillment"
  tags:
    - python
    - tier-1
spec:
  type: service
  lifecycle: production
  owner: team-commerce
  system: e-commerce-platform
  providesApis:
    - order-api
  consumesApis:
    - product-api
    - payment-api
  dependsOn:
    - resource:shared-postgresql

---
# --- Payment Service ---
apiVersion: backstage.io/v1alpha1
kind: Component
metadata:
  name: payment-service
  description: "Payment processing via Stripe"
  tags:
    - java
    - tier-1
    - pci-scope
spec:
  type: service
  lifecycle: production
  owner: team-commerce
  system: e-commerce-platform
  providesApis:
    - payment-api
  dependsOn:
    - resource:shared-postgresql
    - resource:shared-redis

---
# --- Shared Resources ---
apiVersion: backstage.io/v1alpha1
kind: Resource
metadata:
  name: shared-postgresql
  description: "Shared PostgreSQL 15 cluster (AWS RDS)"
spec:
  type: database
  owner: team-commerce
  system: e-commerce-platform

---
apiVersion: backstage.io/v1alpha1
kind: Resource
metadata:
  name: shared-redis
  description: "Shared Redis 7 cluster (AWS ElastiCache)"
spec:
  type: cache
  owner: team-commerce
  system: e-commerce-platform
```

**이 카탈로그 구조의 이점:**
- Backstage의 의존성 그래프가 `order-service`가 `product-api`와 `payment-api` 모두에 의존한다는 것을 보여주어 영향 분석이 쉬워집니다.
- `payment-service`의 `pci-scope` 태그가 보안 팀이 PCI 범위 컴포넌트를 식별하는 데 도움이 됩니다.
- 리소스(`shared-postgresql`, `shared-redis`)가 일급 엔티티로 팀이 어떤 서비스가 인프라를 공유하는지 볼 수 있습니다.
- 각 API가 자체 엔티티를 가지므로 API 문서 검색과 소비자 추적이 가능합니다.

</details>

### 연습 문제 3: Golden Path 평가

플랫폼 팀이 새로운 Python 마이크로서비스를 위한 Golden Path를 구축했습니다. 6개월 후 채택률이 30%에 불과합니다(15팀 중 5팀만 사용). 가능한 이유를 진단하고 해결책을 제안하십시오.

<details>
<summary>정답 보기</summary>

**진단 프레임워크 -- 채택하지 않은 팀 설문조사:**

| 가능한 이유 | 진단 질문 | 해결책 |
|-----------|---------|--------|
| **인식** | "템플릿이 존재하는지 알았습니까?" | 엔지니어링 올핸즈, Slack, 온보딩 문서에 공지 |
| **언어 불일치** | "우리 서비스는 Python이 아니라 Go/Java입니다" | Go와 Java용 템플릿 구축. 팀 수에 따라 우선순위 지정. |
| **커스터마이즈** | "템플릿이 우리 아키텍처(이벤트 드리븐, gRPC)를 지원하지 않습니다" | 템플릿 변형 추가: REST, gRPC, 이벤트 드리븐. 파라미터 선택 허용. |
| **마이그레이션 비용** | "이미 작동하는 설정이 있어서 마이그레이션 가치가 없습니다" | 기존 서비스의 강제 마이그레이션 금지. 새 서비스에 집중. 의향이 있는 팀을 위한 마이그레이션 가이드 제공. |
| **경직성** | "템플릿이 우리에게 맞지 않는 가정을 합니다(잘못된 CI 도구, 잘못된 모니터링)" | 더 많은 부분을 구성 가능하게. 모놀리식 템플릿 대신 컴포지션(모듈) 사용. |
| **신뢰** | "플랫폼 팀이 유지할 것을 신뢰하지 않습니다" | 플랫폼의 SLO 공개. 가동 시간 표시. 이슈 대응 시간 약속. 로드맵 공개. |
| **개발자 경험** | "Backstage가 느리거나/혼란스럽거나/사용하기 어렵습니다" | 사용성 연구 실행. 상위 3개 UX 문제 수정. |

**구체적인 실행 계획:**

1. **1주**: 15개 팀 전부 설문조사 (5분 설문 + 비채택자 3명 인터뷰)
2. **2주**: 결과 분석, 이유별 분류
3. **3-4주**: 상위 2개 차단 요인 해결:
   - 언어인 경우: Go 템플릿 구축 (가장 높은 수요일 가능성)
   - 커스터마이즈인 경우: 아키텍처 변형을 위한 템플릿 파라미터 추가
4. **2개월**: 개선 사항과 함께 재공지. 2개 비채택 팀과 페어링하여 온보딩 (실습 지원).
5. **3개월**: 채택률 재측정. 목표: 50% (15팀 중 8팀).

**핵심 인사이트**: 낮은 채택률은 기술 문제가 아니라 제품 문제입니다. 플랫폼을 제품으로 취급하십시오: 사용자 니즈를 조사하고, 피드백을 기반으로 반복하며, 채택률을 주요 성공 메트릭으로 측정하십시오.

</details>

### 연습 문제 4: 셀프서비스 워크플로우

개발자가 새로운 PostgreSQL 데이터베이스를 프로비저닝하는 셀프서비스 워크플로우를 설계하십시오. 워크플로우에는 다음이 포함되어야 합니다:
1. 개발자 인터페이스 (보고 하는 것)
2. 뒤에서 진행되는 자동화
3. 오용을 방지하는 가드레일

<details>
<summary>정답 보기</summary>

**1. 개발자 인터페이스 (Backstage 템플릿):**

개발자가 Backstage를 열고 "Create"를 클릭하여 "PostgreSQL Database"를 선택합니다:

| 필드 | 유형 | 옵션 | 기본값 |
|------|------|------|--------|
| 데이터베이스 이름 | 텍스트 | `^[a-z][a-z0-9-]*$` | 필수 |
| 환경 | 선택 | dev, staging, production | dev |
| 스토리지 크기 | 선택 | 10 GB, 20 GB, 50 GB, 100 GB | 20 GB |
| PostgreSQL 버전 | 선택 | 14, 15, 16 | 15 |
| 소유자 팀 | 팀 선택기 | (카탈로그에서) | 현재 사용자의 팀 |
| 고가용성 | 체크박스 | true/false | false (dev), true (production) |

개발자가 양식을 작성하고 "Create"를 클릭합니다. 결과가 15분 이내에 나타납니다.

**2. 뒤에서 진행되는 자동화:**

```
Developer clicks "Create"
        │
        ▼
┌──────────────────┐
│  Backstage       │ 1. Validate inputs
│  Scaffolder      │ 2. Apply guardrails (see below)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Crossplane      │ 3. Create AWS RDS instance (Terraform/Crossplane)
│  / Terraform     │ 4. Configure backup (daily, 7-day retention)
│  Module          │ 5. Create monitoring dashboard (Grafana)
└────────┬─────────┘ 6. Set up alerting (connection count, CPU, disk)
         │
         ▼
┌──────────────────┐
│  Vault           │ 7. Generate database credentials
│                  │ 8. Store in Vault (dynamic secret engine)
│                  │ 9. Create K8s ExternalSecret for the app
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Backstage       │ 10. Register as Resource in catalog
│  Catalog         │ 11. Link to owner team and consuming services
└────────┬─────────┘
         │
         ▼
Developer receives: connection string (via Vault),
                    Grafana dashboard URL,
                    catalog link
```

**3. 가드레일:**

| 가드레일 | 구현 | 이유 |
|---------|------|------|
| **크기 제한** | dev 최대 100 GB, production 최대 500 GB | 과도한 비용 방지 |
| **환경 제한** | Production은 HA(Multi-AZ) 필수 | 신뢰성 표준 강제 |
| **명명 규칙** | 정규식 검증: `^[a-z][a-z0-9-]*$` | 일관된 이름 지정 |
| **예산 확인** | 생성 전 예상 월 비용 표시; production 데이터베이스가 $500/월 초과하면 팀 리더 승인 필요 | 비용 가시성 |
| **네트워크 격리** | 데이터베이스를 프라이빗 서브넷에 배치, 팀의 네임스페이스에서만 접근 가능 | 보안 |
| **백업 강제** | 백업 항상 활성화; 비활성화 불가 | 데이터 보호 |
| **자격 증명 관리** | 자격 증명은 Vault에만 저장; UI나 로그에 표시하지 않음 | 시크릿 안전 |
| **dev 데이터베이스 TTL** | dev 데이터베이스는 30일간 비활성 시 자동 삭제 | 비용 정리 |
| **감사 로깅** | 모든 프로비저닝 작업에 사용자, 타임스탬프, 파라미터 기록 | 컴플라이언스 |

</details>

---

## 참고 자료

- [Backstage Documentation](https://backstage.io/docs/)
- [CNCF Platforms White Paper](https://tag-app-delivery.cncf.io/whitepapers/platforms/)
- [Team Topologies (Book)](https://teamtopologies.com/)
- [Crossplane Documentation](https://docs.crossplane.io/)
- [Platform Engineering on Kubernetes (Book)](https://www.manning.com/books/platform-engineering-on-kubernetes)
- [Internal Developer Platform](https://internaldeveloperplatform.org/)

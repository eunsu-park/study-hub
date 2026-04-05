# Platform Engineering

**Previous**: [Chaos Engineering](./16_Chaos_Engineering.md) | **Next**: [SRE Practices](./18_SRE_Practices.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define platform engineering and explain how internal developer platforms (IDPs) reduce cognitive load on development teams
2. Describe the Backstage framework and its core components: software catalog, templates, and plugins
3. Design developer self-service workflows for provisioning infrastructure, creating services, and deploying applications
4. Define golden paths that standardize how teams build and deploy software while preserving flexibility
5. Assess platform maturity using a maturity model and plan incremental improvements
6. Distinguish platform engineering from DevOps and SRE, understanding how they complement each other

---

Platform engineering is the discipline of designing and building toolchains and workflows that enable developer self-service in a cloud-native era. Instead of every team building their own CI/CD pipelines, Kubernetes manifests, monitoring dashboards, and secret management from scratch, a platform team provides a curated, opinionated, and automated foundation -- an Internal Developer Platform (IDP) -- that reduces cognitive load and accelerates delivery.

> **Analogy -- The Factory Floor**: Imagine a factory where every product team must build their own assembly line from scratch: source materials, design conveyors, wire up quality checks, and figure out shipping. Most of their time goes to building the assembly line rather than building the product. Platform engineering is the factory floor itself: standardized assembly lines, shared tooling, and common infrastructure that every product team uses. Teams focus on what to build (their product), not how to build the factory.

## 1. Why Platform Engineering

### 1.1 The Problem It Solves

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

| Aspect | DevOps | SRE | Platform Engineering |
|--------|--------|-----|---------------------|
| **Focus** | Culture, collaboration, automation | Reliability, SLOs, error budgets | Developer experience, self-service |
| **Who** | Everyone (cultural movement) | Dedicated SRE team | Dedicated platform team |
| **Output** | Practices and processes | Reliability standards | Internal Developer Platform (product) |
| **Treats infra as** | Code (IaC) | A system to be made reliable | A product to be used by developers |
| **Key metric** | Deployment frequency, lead time | Availability, error budget | Developer satisfaction, time-to-production |

### 1.3 The Cognitive Load Problem

Research shows developer effectiveness drops as cognitive load increases:

| Cognitive Load Type | Example | Platform Solution |
|-------------------|---------|------------------|
| **Intrinsic** | Learning Kubernetes concepts | Abstraction layers, golden paths |
| **Extraneous** | Navigating 5 different tools to deploy | Unified developer portal |
| **Germane** | Understanding business domain logic | Reduce extraneous load so developers focus here |

---

## 2. Internal Developer Platform (IDP)

### 2.1 IDP Components

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

### 2.2 IDP Capabilities

| Capability | Description | Example |
|-----------|-------------|---------|
| **Service Catalog** | Registry of all services, owners, dependencies, API docs | Backstage Software Catalog |
| **Self-Service Provisioning** | Create new services, databases, environments via UI/CLI | Backstage Templates, Terraform modules |
| **CI/CD** | Standardized build and deploy pipelines | GitHub Actions templates, ArgoCD |
| **Environment Management** | Create, manage, and tear down ephemeral environments | Namespace-per-PR, preview environments |
| **Observability** | Integrated monitoring, logging, tracing | Grafana dashboards auto-provisioned |
| **Documentation** | Centralized, searchable, version-controlled docs | Backstage TechDocs (docs-as-code) |
| **Security** | Automated secret management, vulnerability scanning | Vault integration, Trivy scanning |
| **Cost Management** | Resource cost visibility per team/service | Cloud cost dashboards |

---

## 3. Backstage

### 3.1 What is Backstage

Backstage is an open-source platform for building developer portals, created by Spotify and donated to the CNCF. It provides a unified frontend for all infrastructure tooling.

### 3.2 Core Components

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

The catalog is populated by `catalog-info.yaml` files in each service's repository:

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

---
apiVersion: backstage.io/v1alpha1
kind: API
metadata:
  name: order-api
  description: "Order management REST API"
spec:
  type: openapi
  lifecycle: production
  owner: team-commerce
  definition:
    $text: ./openapi.yaml

---
apiVersion: backstage.io/v1alpha1
kind: Resource
metadata:
  name: order-db
  description: "PostgreSQL database for order data"
spec:
  type: database
  owner: team-commerce
  system: e-commerce-platform
```

### 3.4 Software Templates (Scaffolder)

Templates automate the creation of new services with all platform integrations pre-configured:

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

TechDocs renders Markdown documentation directly in the Backstage portal, making it searchable alongside the software catalog.

---

## 4. Golden Paths

### 4.1 What is a Golden Path

A golden path is the opinionated, supported, and paved way to accomplish a common task. It is the path of least resistance that also happens to be the best practice. Developers are free to deviate, but the golden path is so convenient that most choose to follow it.

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

### 4.2 Common Golden Paths

| Golden Path | What It Provides |
|-------------|-----------------|
| **New Service** | Repo with skeleton, Dockerfile, CI/CD, K8s manifests, monitoring, catalog registration |
| **New API** | OpenAPI spec template, API gateway configuration, rate limiting, authentication |
| **New Database** | Terraform module, backup policy, monitoring, secret rotation |
| **New Environment** | Namespace creation, RBAC, resource quotas, network policies |
| **Deployment** | PR → CI → Staging → Canary → Production (automated) |
| **Incident Response** | PagerDuty alert → Slack channel → Runbook → Postmortem template |

### 4.3 Golden Path Design Principles

| Principle | Description |
|-----------|-------------|
| **Opinionated but not restrictive** | Provide defaults, allow overrides. "This is how we do it; you can change it if needed." |
| **Self-service** | Developers should not need to file tickets or wait for another team |
| **Automated** | Every step should be automated or template-driven |
| **Documented** | Clear docs on what the golden path does and how to customize it |
| **Maintained** | Platform team keeps templates and tools up to date |
| **Measured** | Track adoption: how many teams use the golden path vs. custom solutions |

---

## 5. Developer Self-Service

### 5.1 Self-Service Capabilities

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

### 5.2 Infrastructure Self-Service with Crossplane

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

## 6. Platform Maturity Model

### 6.1 Five Levels of Platform Maturity

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

### 6.2 Maturity Assessment Criteria

| Dimension | Level 1 | Level 3 | Level 5 |
|-----------|---------|---------|---------|
| **Provisioning** | Tickets & manual | Templates & self-service | API-driven, instant |
| **Deployment** | Manual scripts | CI/CD pipelines | GitOps with canary, auto-rollback |
| **Observability** | SSH + grep | Centralized dashboards | Correlated metrics-logs-traces |
| **Documentation** | Wiki (stale) | Docs-as-code | Searchable portal with API docs |
| **Developer experience** | "Ask ops team" | "Use the template" | "It just works" |
| **Adoption measurement** | None | Usage counts | NPS scores, time-to-production |

---

## 7. Platform Team Structure

### 7.1 Team Composition

| Role | Responsibility |
|------|----------------|
| **Platform Product Manager** | Defines roadmap based on developer needs, measures adoption |
| **Platform Engineers** | Build and maintain platform services (CI/CD, K8s, IaC) |
| **Developer Experience (DX) Engineer** | Backstage development, templates, documentation |
| **SRE / Reliability Engineer** | Platform reliability, SLOs for platform services |
| **Security Engineer** | Security guardrails, policy-as-code, compliance automation |

### 7.2 Platform as a Product

| Product Practice | Platform Application |
|-----------------|---------------------|
| **User research** | Developer surveys, shadowing, pain-point interviews |
| **Roadmap** | Prioritized feature list based on developer impact |
| **MVP** | Start with one golden path, iterate based on feedback |
| **Release notes** | Announce new platform features, migrations, deprecations |
| **Support** | Slack channel, office hours, documentation |
| **Metrics** | Adoption rate, developer satisfaction (NPS), time-to-production |

### 7.3 Key Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Time to first deploy** | How long from "I have an idea" to running in production | < 1 day |
| **Deployment frequency** | How often teams deploy | Multiple times per day |
| **Platform adoption** | % of services using the platform | > 80% |
| **Developer NPS** | Developer satisfaction with the platform | > 50 |
| **Self-service ratio** | % of requests handled without platform team involvement | > 90% |
| **Mean time to provision** | Time to provision a new database, environment, etc. | < 30 minutes |

---

## 8. Next Steps

- [18_SRE_Practices.md](./18_SRE_Practices.md) - SRE practices for platform reliability
- [14_GitOps.md](./14_GitOps.md) - GitOps as a foundation for platform deployment

---

## Exercises

### Exercise 1: Platform Design

Your company has 15 development teams and 60 microservices. Currently, each team manages their own CI/CD, Kubernetes manifests, monitoring, and secrets. The CEO asks you to build an internal developer platform. Design the first 3 months of the platform roadmap.

<details>
<summary>Show Answer</summary>

**Month 1: Foundation (Level 1 to Level 2)**

| Week | Deliverable | Impact |
|------|------------|--------|
| 1-2 | **Service Catalog**: Deploy Backstage, import all 60 services via `catalog-info.yaml`. Establish ownership for every service. | Every engineer can find any service, its owner, docs, and dashboards from one place. |
| 3-4 | **Standardized CI pipeline template**: Create a reusable GitHub Actions workflow for build, test, container push. | Teams stop writing CI from scratch. New services get CI in minutes. |

**Month 2: Golden Path (Level 2 to Level 3)**

| Week | Deliverable | Impact |
|------|------------|--------|
| 5-6 | **Service template**: Backstage scaffolder template for the most common service type (Python REST API with PostgreSQL). Auto-creates repo, Dockerfile, K8s manifests, CI/CD, Grafana dashboard, ArgoCD app. | New service from idea to running in < 30 minutes. |
| 7-8 | **Standardized Kubernetes manifests**: Kustomize base with health checks, resource limits, HPA, network policies. Teams adopt via PR to their repos. | Consistent security, resource management, and observability across all services. |

**Month 3: Self-Service (Strengthening Level 3)**

| Week | Deliverable | Impact |
|------|------------|--------|
| 9-10 | **Database self-service**: Crossplane or Terraform module for PostgreSQL provisioning. Developers request via Backstage template or CLI. | Database provisioning drops from 2-day ticket to 15-minute self-service. |
| 11-12 | **Observability golden path**: Auto-provisioned Grafana dashboards (4 golden signals) for every service in the catalog. Standardized alerting templates. | Every service has monitoring from day one. No more "we forgot to add monitoring." |

**Success metrics at Month 3:**
- 100% of services registered in Backstage catalog
- 5+ new services created using the template
- Database provisioning time reduced from 2 days to 15 minutes
- Developer NPS baseline established (survey)

**What NOT to do in Month 1-3:**
- Do not try to migrate all 60 services at once. Start with new services and willing early-adopter teams.
- Do not build a custom developer portal from scratch. Use Backstage.
- Do not mandate adoption. Make the platform so convenient that teams choose it.

</details>

### Exercise 2: Backstage Catalog Design

Design the `catalog-info.yaml` structure for an e-commerce system with the following components:
- API Gateway (Node.js)
- Product Service (Go, provides Product API)
- Order Service (Python, provides Order API, consumes Product API and Payment API)
- Payment Service (Java, provides Payment API, depends on PostgreSQL and Redis)
- Shared PostgreSQL database
- Shared Redis cache
- The system is owned by two teams: `team-catalog` (Product) and `team-commerce` (Order, Payment)

<details>
<summary>Show Answer</summary>

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
  annotations:
    github.com/project-slug: org/product-service
    grafana/dashboard-selector: "product-service"
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
apiVersion: backstage.io/v1alpha1
kind: API
metadata:
  name: product-api
  description: "Product catalog REST API"
spec:
  type: openapi
  lifecycle: production
  owner: team-catalog
  system: e-commerce-platform
  definition:
    $text: ./openapi/product-api.yaml

---
# --- Order Service ---
apiVersion: backstage.io/v1alpha1
kind: Component
metadata:
  name: order-service
  description: "Order creation and fulfillment"
  annotations:
    github.com/project-slug: org/order-service
    grafana/dashboard-selector: "order-service"
    pagerduty.com/service-id: P1234567
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
apiVersion: backstage.io/v1alpha1
kind: API
metadata:
  name: order-api
  description: "Order management REST API"
spec:
  type: openapi
  lifecycle: production
  owner: team-commerce
  system: e-commerce-platform
  definition:
    $text: ./openapi/order-api.yaml

---
# --- Payment Service ---
apiVersion: backstage.io/v1alpha1
kind: Component
metadata:
  name: payment-service
  description: "Payment processing via Stripe"
  annotations:
    github.com/project-slug: org/payment-service
    grafana/dashboard-selector: "payment-service"
    pagerduty.com/service-id: P7654321
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
apiVersion: backstage.io/v1alpha1
kind: API
metadata:
  name: payment-api
  description: "Payment processing API"
spec:
  type: openapi
  lifecycle: production
  owner: team-commerce
  system: e-commerce-platform
  definition:
    $text: ./openapi/payment-api.yaml

---
# --- Shared Resources ---
apiVersion: backstage.io/v1alpha1
kind: Resource
metadata:
  name: shared-postgresql
  description: "Shared PostgreSQL 15 cluster (AWS RDS)"
  tags:
    - database
    - aws
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
  tags:
    - cache
    - aws
spec:
  type: cache
  owner: team-commerce
  system: e-commerce-platform

---
# --- Team Definitions ---
apiVersion: backstage.io/v1alpha1
kind: Group
metadata:
  name: team-catalog
  description: "Product catalog team"
spec:
  type: team
  children: []

---
apiVersion: backstage.io/v1alpha1
kind: Group
metadata:
  name: team-commerce
  description: "Commerce platform team (orders, payments)"
spec:
  type: team
  children: []
```

**Benefits of this catalog structure:**
- The dependency graph in Backstage shows that `order-service` depends on both `product-api` and `payment-api`, making impact analysis easy.
- The `pci-scope` tag on `payment-service` helps security teams identify PCI-scoped components.
- Resources (`shared-postgresql`, `shared-redis`) are first-class entities, so teams can see which services share infrastructure.
- Each API has its own entity, enabling API documentation discovery and consumer tracking.

</details>

### Exercise 3: Golden Path Evaluation

Your platform team has built a golden path for creating new Python microservices. After 6 months, adoption is only 30% (5 of 15 teams use it). Diagnose possible reasons and propose solutions.

<details>
<summary>Show Answer</summary>

**Diagnosis framework -- survey teams that did NOT adopt:**

| Possible Reason | Diagnostic Question | Solution |
|----------------|-------------------|----------|
| **Awareness** | "Did you know the template exists?" | Announce in engineering all-hands, Slack, onboarding docs |
| **Language mismatch** | "Our service is in Go/Java, not Python" | Build templates for Go and Java. Prioritize by team count. |
| **Customization** | "The template does not support our architecture (event-driven, gRPC)" | Add template variants: REST, gRPC, event-driven. Allow parameter selection. |
| **Migration cost** | "We already have a working setup; migration is not worth it" | Do not force migration of existing services. Focus on new services. Provide a migration guide for willing teams. |
| **Rigidity** | "The template makes assumptions that do not fit us (wrong CI tool, wrong monitoring)" | Make more parts configurable. Use composition (modules) instead of monolithic templates. |
| **Trust** | "We do not trust the platform team to maintain it" | Publish SLOs for the platform. Show uptime. Commit to response times for issues. Publish a roadmap. |
| **Developer experience** | "Backstage is slow / confusing / hard to use" | Run a usability study. Fix the top 3 UX issues. |

**Concrete action plan:**

1. **Week 1**: Survey all 15 teams (5-minute survey + 3 interviews with non-adopters)
2. **Week 2**: Analyze results, categorize by reason
3. **Week 3-4**: Address the top 2 blockers:
   - If language: Build Go template (likely highest demand)
   - If customization: Add template parameters for architecture variants
4. **Month 2**: Re-announce with improvements. Pair with 2 non-adopter teams to onboard them (hands-on support).
5. **Month 3**: Measure adoption again. Target: 50% (8 of 15 teams).

**Key insight**: Low adoption is a product problem, not a technology problem. Treat the platform as a product: research user needs, iterate on feedback, and measure adoption as the primary success metric.

</details>

### Exercise 4: Self-Service Workflow

Design a self-service workflow for a developer to provision a new PostgreSQL database. The workflow should include:
1. The developer interface (what they see/do)
2. The automation behind the scenes
3. The guardrails that prevent misuse

<details>
<summary>Show Answer</summary>

**1. Developer interface (Backstage template):**

The developer opens Backstage, clicks "Create", selects "PostgreSQL Database":

| Field | Type | Options | Default |
|-------|------|---------|---------|
| Database name | Text | `^[a-z][a-z0-9-]*$` | Required |
| Environment | Select | dev, staging, production | dev |
| Storage size | Select | 10 GB, 20 GB, 50 GB, 100 GB | 20 GB |
| PostgreSQL version | Select | 14, 15, 16 | 15 |
| Owner team | Team picker | (from catalog) | Current user's team |
| High availability | Checkbox | true/false | false (dev), true (production) |

The developer fills in the form and clicks "Create". Result appears in < 15 minutes.

**2. Automation behind the scenes:**

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

**3. Guardrails:**

| Guardrail | Implementation | Reason |
|-----------|---------------|--------|
| **Size limits** | Max 100 GB for dev, max 500 GB for production | Prevent runaway costs |
| **Environment restrictions** | Production requires HA (Multi-AZ) | Enforce reliability standards |
| **Naming convention** | Regex validation: `^[a-z][a-z0-9-]*$` | Consistent naming |
| **Budget check** | Estimated monthly cost shown before creation; production databases require team lead approval if > $500/month | Cost visibility |
| **Network isolation** | Database placed in private subnet, accessible only from the team's namespace | Security |
| **Backup enforcement** | Backups always enabled; cannot be disabled | Data protection |
| **Credential management** | Credentials stored in Vault only; never shown in UI or logs | Secret safety |
| **TTL for dev databases** | Dev databases auto-deleted after 30 days of inactivity | Cost cleanup |
| **Audit logging** | All provisioning actions logged with user, timestamp, and parameters | Compliance |

</details>

---

## References

- [Backstage Documentation](https://backstage.io/docs/)
- [CNCF Platforms White Paper](https://tag-app-delivery.cncf.io/whitepapers/platforms/)
- [Team Topologies (Book)](https://teamtopologies.com/)
- [Crossplane Documentation](https://docs.crossplane.io/)
- [Platform Engineering on Kubernetes (Book)](https://www.manning.com/books/platform-engineering-on-kubernetes)
- [Internal Developer Platform](https://internaldeveloperplatform.org/)

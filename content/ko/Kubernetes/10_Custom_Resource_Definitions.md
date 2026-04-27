# 10. 커스텀 리소스 정의(Custom Resource Definitions)

**이전**: [Helm과 Kustomize](./09_Helm_and_Kustomize.md) | **다음**: [오퍼레이터](./11_Operators.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 커스텀 리소스 정의(Custom Resource Definitions)가 Kubernetes API를 어떻게 확장하는지 설명할 수 있다
2. 구조적 스키마(structural schema), 유효성 검증 규칙(validation rule), 기본값(default value)을 갖춘 CRD를 생성할 수 있다
3. 스토리지 버전(storage version)과 변환 웹훅(conversion webhook)을 사용한 CRD 버전 관리를 구현할 수 있다
4. 커스텀 프린터 컬럼(printer column)과 함께 status 및 scale 하위 리소스(subresource)를 사용할 수 있다
5. CRD와 집계 API 서버(aggregated API server) 중 어떤 것을 사용할지 평가할 수 있다

---

Kubernetes의 가장 강력한 기능 중 하나는 확장성(extensibility)입니다. 코어 API가 Pod, Service, Deployment와 같은 리소스를 제공하는 반면, 커스텀 리소스 정의(CRD)를 사용하면 Kubernetes에 완전히 새로운 리소스 유형 -- 데이터베이스, 인증서, 네트워크 정책, 워크플로우 단계, 또는 조직에 필요한 모든 도메인별 개념 -- 을 가르칠 수 있습니다. 등록되면 커스텀 리소스(custom resource)는 내장 리소스와 정확히 동일하게 동작합니다: etcd에 저장되고, RESTful API 엔드포인트를 가지며, RBAC를 지원하고, kubectl과 함께 작동합니다. 이 레슨에서는 기본 정의부터 버전 관리, 변환 웹훅(conversion webhook), 하위 리소스(subresource)와 같은 고급 기능까지 CRD 설계를 다룹니다.

> **확장 스펙트럼(Extension Spectrum):** CRD는 Kubernetes API를 확장하는 가장 간단한 방법입니다. 정의하는 데 Go 코드가 필요 없이 YAML 매니페스트만 있으면 됩니다. 더 복잡한 요구사항(커스텀 스토리지 백엔드, 인증, 또는 API 집계)의 경우 집계 API 서버(aggregated API server)를 구축할 수 있습니다. 대부분의 사용 사례는 CRD와 컨트롤러(controller)의 조합으로 잘 처리됩니다(오퍼레이터 레슨에서 다룹니다).

## 목차

- [이론과 원리](#이론과-원리)
- [1. Kubernetes API 확장](#1-extending-the-kubernetes-api)
  - [1.1 왜 Kubernetes를 확장하는가?](#11-why-extend-kubernetes)
  - [1.2 확장 메커니즘](#12-extension-mechanisms)
  - [1.3 CRD 작동 방식](#13-how-crds-work)
- [2. CRD 명세](#2-crd-specification)
  - [2.1 기본 CRD](#21-basic-crd)
  - [2.2 커스텀 리소스 생성 및 사용](#22-creating-and-using-custom-resources)
- [3. 구조적 스키마와 유효성 검증](#3-structural-schemas-and-validation)
  - [3.1 OpenAPI v3 스키마](#31-openapi-v3-schema)
  - [3.2 유효성 검증 규칙 (CEL)](#32-validation-rules-cel)
  - [3.3 기본값](#33-default-values)
  - [3.4 열거형과 패턴 제약 조건](#34-enum-and-pattern-constraints)
- [4. CRD 버전 관리](#4-crd-versioning)
  - [4.1 다중 버전](#41-multiple-versions)
  - [4.2 스토리지 버전](#42-storage-version)
  - [4.3 변환 웹훅](#43-conversion-webhooks)
- [5. 하위 리소스](#5-subresources)
  - [5.1 Status 하위 리소스](#51-status-subresource)
  - [5.2 Scale 하위 리소스](#52-scale-subresource)
- [6. 프린터 컬럼](#6-printer-columns)
- [7. 카테고리와 짧은 이름](#7-categories-and-short-names)
- [8. CRD 모범 사례](#8-crd-best-practices)
  - [8.1 설계 가이드라인](#81-design-guidelines)
  - [8.2 스키마 진화](#82-schema-evolution)
  - [8.3 성능 고려사항](#83-performance-considerations)
- [9. 집계 API 서버 vs CRD](#9-aggregated-api-servers-vs-crds)
  - [9.1 각각의 사용 시기](#91-when-to-use-each)
  - [9.2 집계 API 서버 예제](#92-aggregated-api-server-example)
- [연습문제](#exercises)

---

## 1. Kubernetes API 확장

### 이론: CRD가 존재하는 이유 — 확장성 세금

장기 채택을 원하는 플랫폼은 근본적 선택에 직면합니다. 둘 중 하나:

- **코어를 패치하여 기능 추가** — 모든 벤더와 operator 작성자가 코드를 upstream에 보내고, 리뷰를 받고, 쿠버네티스 릴리스를 기다려야 함을 의미. 플랫폼이 병목이자 전쟁터가 됩니다.
- **확장 메커니즘 추가** — 응집력의 일부 손실을 수용하는 대가로 생태계가 자체 속도로 움직이게 합니다.

쿠버네티스는 두 번째를 선택했습니다. 두 확장 메커니즘은:

- **CRD** (이 레슨) — 선언적 — 새 kind를 기술하는 CRD 객체를 POST하면, API 서버의 CRD 컨트롤러가 동적으로 등록합니다. 새 프로세스 없음, API 표면에 Go 코드 불필요.
- **Aggregated API server** (§D) — 쿠버네티스 API 관습(list, watch, create 등)을 구현하는 자체 HTTPS 서버를 실행하고, `APIService` 객체로 등록. kube-apiserver가 당신의 group/version에 대한 요청을 당신의 서버로 프록시합니다.

CRD는 실제 사용 사례의 약 95%를 다룹니다. cert-manager의 `Certificate`, ArgoCD의 `Application`, Knative의 `Service`, Istio의 `VirtualService`, 모든 CNCF operator의 리소스 — 모두 CRD입니다. Aggregated API server는 주로 커스텀 스토리지(etcd가 아닌), 커스텀 인증, 또는 CRD 이전의 역사적 이유가 필요한 케이스에 존재합니다(metrics API server가 유명한 예).

CRD의 "세금"은 엄격한 요청/응답 스키마를 넘는 것(커스텀 스토리지, 여러 객체 간 트랜잭션 시맨틱)이 컨트롤러(11강) 형태의 코드를 필요로 한다는 것입니다. CRD가 리소스의 *형태*를 주고, 컨트롤러가 *의미*를 줍니다.

### 1.1 왜 Kubernetes를 확장하는가?

커스텀 리소스(custom resource)를 사용하면 Kubernetes API에서 도메인별 개념을 표현할 수 있습니다:

| 도메인 | 커스텀 리소스 | 표현하는 것 |
|--------|----------------|-------------------|
| 데이터베이스 | `PostgresCluster` | 관리형 PostgreSQL 클러스터 |
| 인증서 | `Certificate` | TLS 인증서 요청 (cert-manager) |
| CI/CD | `Pipeline` | CI/CD 파이프라인 정의 (Tekton) |
| 네트워킹 | `Gateway` | L4/L7 로드 밸런서 (Gateway API) |
| ML | `TFJob` | TensorFlow 학습 잡 (Kubeflow) |
| GitOps | `Application` | ArgoCD 애플리케이션 |

### 1.2 확장 메커니즘

```
┌──────────────────────────────────────────────────────────────┐
│  Kubernetes API Extension Mechanisms                         │
│                                                              │
│  Simple ─────────────────────────────────────────── Complex  │
│                                                              │
│  ConfigMaps    CRDs         CRDs +          Aggregated       │
│  (data only,   (declarative  Controller     API Server       │
│   no schema)    schema,      (reconciliation (custom storage, │
│                 validation)   loop)          custom auth)     │
│                                                              │
│  Use case:     Use case:     Use case:      Use case:        │
│  App config    API objects   Operators       metrics-server   │
│                               (cert-manager,  (custom backend)│
│                               ArgoCD)                        │
└──────────────────────────────────────────────────────────────┘
```

### 1.3 CRD 작동 방식

CRD를 생성하면 API 서버가 새로운 REST 엔드포인트를 동적으로 등록합니다:

```
1. Apply CRD manifest
   ┌──────────┐    POST /apis/apiextensions.k8s.io/v1/customresourcedefinitions
   │  kubectl  │──────────────────────────────────────────────────────────────▶
   └──────────┘

2. API server creates endpoints
   /apis/<group>/<version>/namespaces/<ns>/<plural>
   /apis/<group>/<version>/namespaces/<ns>/<plural>/<name>
   /apis/<group>/<version>/namespaces/<ns>/<plural>/<name>/status
   /apis/<group>/<version>/namespaces/<ns>/<plural>/<name>/scale

3. Custom resources are stored in etcd
   key: /registry/<group>/<plural>/<namespace>/<name>

4. Standard tooling works immediately
   kubectl get <plural>
   kubectl describe <plural> <name>
   kubectl delete <plural> <name>
```

---

## 2. CRD 명세

### 2.1 기본 CRD

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: databases.example.com       # Must be <plural>.<group>
spec:
  group: example.com                 # API group
  names:
    kind: Database                   # CamelCase singular
    listKind: DatabaseList
    singular: database               # lowercase singular (kubectl)
    plural: databases                # lowercase plural (API path)
    shortNames:                      # Short names for kubectl
      - db
    categories:                      # Grouping for kubectl get all
      - all
      - example
  scope: Namespaced                  # or Cluster
  versions:
    - name: v1alpha1
      served: true                   # Accept requests for this version
      storage: true                  # Store in etcd in this version
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              required:
                - engine
                - version
              properties:
                engine:
                  type: string
                  enum: ["postgres", "mysql", "mongodb"]
                version:
                  type: string
                replicas:
                  type: integer
                  minimum: 1
                  maximum: 10
                  default: 1
                storage:
                  type: object
                  properties:
                    size:
                      type: string
                      pattern: "^[0-9]+(Gi|Ti)$"
                      default: "10Gi"
                    storageClassName:
                      type: string
            status:
              type: object
              properties:
                phase:
                  type: string
                readyReplicas:
                  type: integer
                conditions:
                  type: array
                  items:
                    type: object
                    properties:
                      type:
                        type: string
                      status:
                        type: string
                        enum: ["True", "False", "Unknown"]
                      lastTransitionTime:
                        type: string
                        format: date-time
                      reason:
                        type: string
                      message:
                        type: string
```

```bash
# Apply the CRD
kubectl apply -f database-crd.yaml

# Verify the CRD is registered
kubectl get crd databases.example.com
# NAME                     CREATED AT
# databases.example.com    2024-01-15T10:00:00Z

# Check the new API endpoints
kubectl api-resources | grep database
# databases   db    example.com/v1alpha1   true   Database
```

### 2.2 커스텀 리소스 생성 및 사용

```yaml
# my-database.yaml
apiVersion: example.com/v1alpha1
kind: Database
metadata:
  name: orders-db
  namespace: production
spec:
  engine: postgres
  version: "16.2"
  replicas: 3
  storage:
    size: 100Gi
    storageClassName: fast-ssd
```

```bash
# Create a custom resource
kubectl apply -f my-database.yaml

# List databases
kubectl get databases -n production
# or use the short name
kubectl get db -n production
# NAME        AGE
# orders-db   5s

# Describe the resource
kubectl describe db orders-db -n production

# Get as YAML
kubectl get db orders-db -n production -o yaml

# Delete
kubectl delete db orders-db -n production

# Watch for changes
kubectl get db -n production -w
```

---

## 3. 구조적 스키마와 유효성 검증

### 이론: 스키마 파이프라인 — OpenAPI v3 + CEL

검증 없는 CRD는 자유 형식 JSON 블롭에 대한 타입화된 이름입니다. 그것은 거의 원하는 바가 아닙니다. 현대 CRD는 API 서버가 어드미션 시점에 모든 create/update를 검증하는 데 사용하는 **OpenAPI v3 스키마**를 임베드합니다:

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: databases.example.com
spec:
  group: example.com
  names: { kind: Database, listKind: DatabaseList, plural: databases, singular: database }
  scope: Namespaced
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
              required: [engine, version, replicas]
              properties:
                engine:
                  type: string
                  enum: [postgres, mysql]
                version:
                  type: string
                  pattern: '^[0-9]+\.[0-9]+$'
                replicas:
                  type: integer
                  minimum: 1
                  maximum: 100
              x-kubernetes-validations:
                - rule: "self.engine == 'postgres' || self.replicas <= 10"
                  message: "MySQL clusters limited to 10 replicas"
```

스키마는 다음을 강제합니다:

- **타입 구조** (`type`, `properties`, `required`, `additionalProperties`) — 표준 JSON Schema. 필수 속성, 타입 검사, 중첩 객체 형태.
- **패턴 및 경계 제약** (`pattern`, `minimum`, `maximum`, `enum`, `minLength`) — 단순한 데이터 품질 오류를 컨트롤러가 크래시하는 런타임이 아니라 어드미션 시점에 잡습니다.
- **CEL 검증 규칙** (`x-kubernetes-validations`) — **Common Expression Language**가 위 예제의 `engine == 'postgres' || replicas <= 10`처럼 여러 필드를 가로지르는 제약을 표현하게 해줍니다. CEL은 또한 transition 규칙("version 필드는 증가만 가능")을 위해 *옛* 객체(`oldSelf`)에 접근할 수 있습니다.
- **기본값(defaulting)** (`default: ...`) — API 서버가 누락된 필드를 채우므로, 컨트롤러는 그것들이 설정되었다고 가정할 수 있습니다.

이는 도메인 객체를 모델링하기 위해 generic ConfigMap 대신 CRD를 사용하는 주요 이유입니다 — 검증, defaulting, IDE 자동 완성(스키마가 클러스터에 게시되므로)을 무료로 얻습니다.

**Subresource**(`/status`, `/scale`)는 단일 객체를 다른 RBAC와 업데이트 시맨틱을 가진 여러 엔드포인트로 분할합니다. `/status` subresource는 컨트롤러가 spec을 변경할 권한 없이 status를 업데이트하게 해주어, 컨트롤러 버그가 사용자 의도를 실수로 덮어쓰는 것을 방지합니다. `/scale` subresource는 HPA가 내부 형태를 모르고도 커스텀 리소스를 스케일하게 해줍니다 — `scale.spec.replicas`가 표준 위치입니다.

### 3.1 OpenAPI v3 스키마

모든 CRD는 구조적 스키마(structural schema)를 가져야 합니다. 이 스키마는 생성 및 업데이트 시 모든 커스텀 리소스의 유효성을 검증합니다.

```yaml
schema:
  openAPIV3Schema:
    type: object
    description: "Database represents a managed database instance"
    properties:
      spec:
        type: object
        description: "Desired state of the database"
        required:
          - engine
          - version
        properties:
          engine:
            type: string
            description: "Database engine type"
            enum: ["postgres", "mysql", "mongodb"]
          version:
            type: string
            description: "Engine version"
          replicas:
            type: integer
            description: "Number of database replicas"
            minimum: 1
            maximum: 10
            default: 1
          storage:
            type: object
            description: "Storage configuration"
            properties:
              size:
                type: string
                description: "Storage size (e.g., 10Gi, 1Ti)"
                pattern: "^[0-9]+(Mi|Gi|Ti)$"
                default: "10Gi"
              storageClassName:
                type: string
                description: "StorageClass to use"
          backup:
            type: object
            description: "Backup configuration"
            properties:
              enabled:
                type: boolean
                default: true
              schedule:
                type: string
                description: "Cron schedule for backups"
                default: "0 2 * * *"
              retention:
                type: integer
                description: "Number of backups to retain"
                minimum: 1
                default: 7
          connection:
            type: object
            description: "Connection settings"
            properties:
              maxConnections:
                type: integer
                minimum: 10
                maximum: 10000
                default: 100
              sslMode:
                type: string
                enum: ["disable", "require", "verify-ca", "verify-full"]
                default: "require"
    # x-kubernetes-preserve-unknown-fields: true  # Uncomment to allow extra fields
```

### 3.2 유효성 검증 규칙 (CEL)

Common Expression Language(CEL) 규칙은 교차 필드 유효성 검증(cross-field validation)을 제공합니다 (Kubernetes 1.25+).

```yaml
schema:
  openAPIV3Schema:
    type: object
    properties:
      spec:
        type: object
        x-kubernetes-validations:
          # Cross-field validation: replicas must be odd for postgres
          - rule: "self.engine != 'postgres' || self.replicas % 2 == 1"
            message: "PostgreSQL replicas must be an odd number for quorum"

          # Conditional requirement: if backup enabled, schedule must be set
          - rule: "!has(self.backup) || !self.backup.enabled || has(self.backup.schedule)"
            message: "Backup schedule is required when backups are enabled"

          # Immutable field: engine cannot be changed after creation
          - rule: "self.engine == oldSelf.engine"
            message: "Engine type is immutable and cannot be changed"

          # String format validation
          - rule: "self.version.matches('^[0-9]+\\\\.[0-9]+$')"
            message: "Version must be in format MAJOR.MINOR (e.g., 16.2)"
        properties:
          engine:
            type: string
            enum: ["postgres", "mysql", "mongodb"]
          version:
            type: string
          replicas:
            type: integer
            minimum: 1
            maximum: 10
            x-kubernetes-validations:
              - rule: "self >= oldSelf"
                message: "Replicas can only be scaled up, not down"
          storage:
            type: object
            properties:
              size:
                type: string
                x-kubernetes-validations:
                  - rule: "self == oldSelf"
                    message: "Storage size is immutable (resize not supported)"
```

### 3.3 기본값

```yaml
# Defaults are applied server-side when fields are omitted
properties:
  spec:
    type: object
    properties:
      replicas:
        type: integer
        default: 1
      storage:
        type: object
        default: {}      # Ensures the object exists so nested defaults apply
        properties:
          size:
            type: string
            default: "10Gi"
          storageClassName:
            type: string
            default: "standard"
      monitoring:
        type: object
        default: {}
        properties:
          enabled:
            type: boolean
            default: true
          interval:
            type: string
            default: "30s"
```

```bash
# Creating a minimal resource -- defaults fill in the rest
cat <<EOF | kubectl apply -f -
apiVersion: example.com/v1alpha1
kind: Database
metadata:
  name: test-db
  namespace: default
spec:
  engine: postgres
  version: "16.2"
EOF

# Verify defaults were applied
kubectl get db test-db -o yaml
# spec:
#   engine: postgres
#   version: "16.2"
#   replicas: 1              <-- default
#   storage:
#     size: 10Gi             <-- default
#     storageClassName: standard  <-- default
#   monitoring:
#     enabled: true           <-- default
#     interval: 30s           <-- default
```

### 3.4 열거형과 패턴 제약 조건

```yaml
properties:
  spec:
    type: object
    properties:
      # Enum: fixed set of allowed values
      tier:
        type: string
        enum: ["development", "staging", "production"]
        description: "Service tier determines resource allocation"

      # Pattern: regex validation
      name:
        type: string
        pattern: "^[a-z][a-z0-9-]{0,61}[a-z0-9]$"
        description: "DNS-compatible name"
        minLength: 2
        maxLength: 63

      # Numeric constraints
      port:
        type: integer
        minimum: 1024
        maximum: 65535
        exclusiveMinimum: true   # port > 1024

      # Array constraints
      allowedCIDRs:
        type: array
        items:
          type: string
          pattern: "^[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}/[0-9]{1,2}$"
        minItems: 1
        maxItems: 50

      # Map/object with additional properties
      labels:
        type: object
        additionalProperties:
          type: string
        x-kubernetes-map-type: granular   # Allow individual field updates
```

---

## 4. CRD 버전 관리

### 이론: 버전 진화 — Storage Version, Served Versions, Conversion Webhook

CRD는 진화합니다. v1alpha1, 그다음 v1beta1, 그다음 v1을 릴리스합니다. 각 버전은 필드를 추가하거나, 제거하거나, 하위 객체를 재구성할 수 있습니다. 세 개념이 이를 안전하게 만듭니다:

**Served versions vs storage version.** CRD는 여러 버전을 나열합니다 — 각각 `served: true/false`(API가 그것을 제공하는지)이고 정확히 하나가 `storage: true`(실제로 etcd에 있는 정규 버전). 클라이언트가 v1beta1로 리소스를 GET하지만 storage가 v1이면, API 서버는 etcd 객체를 v1beta1로 즉석 변환하여 반환합니다. 반대로, v1beta1의 POST는 저장 전 v1으로 변환됩니다.

**No-op 변환**(기본)은 모든 served 버전이 구조적으로 호환되어야 합니다 — 선택 필드 추가는 OK, 그러나 이름 변경이나 재구성은 안 됩니다. 더 큰 진화에는 **conversion webhook**을 제공합니다:

```yaml
spec:
  conversion:
    strategy: Webhook
    webhook:
      conversionReviewVersions: [v1]
      clientConfig:
        service: { name: my-converter, namespace: example, path: /convert }
        caBundle: <base64-cert>
```

웹훅은 어떤 served 버전이든 객체를 받아 요청된 버전으로 반환합니다. 이는 옛 클라이언트가 계속 동작하면서도 진짜 리팩토링(필드를 둘로 분할, 열거형의 표현 변경)을 할 수 있게 합니다.

**Storage version 마이그레이션.** `storage: true`를 새 버전으로 변경하면, API 서버는 새 버전으로 쓰기 시작하지만, 기존 etcd 객체는 옛 형식 그대로 남습니다. 별도의 "storage version migrator"(또는 일회성 `kubectl get ... | kubectl apply -f -` 왕복)가 그것들을 새 형식으로 다시 저장합니다. 모든 것이 마이그레이션되면, `served`에서 옛 버전을 제거할 수 있습니다.

이 세 부분 춤(served, storage, conversion)이 CRD가 기존 배포를 깨뜨리지 않고 수년간 진화할 수 있게 하는 것입니다 — 쿠버네티스 자체가 내장 리소스에 사용하는 동일한 기법입니다.

### 4.1 다중 버전

CRD는 여러 버전을 동시에 제공하여 점진적인 마이그레이션(migration)을 가능하게 합니다.

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: databases.example.com
spec:
  group: example.com
  names:
    kind: Database
    plural: databases
    singular: database
    shortNames: ["db"]
  scope: Namespaced
  versions:
    - name: v1alpha1
      served: true              # Still accepts requests
      storage: false            # NOT the storage version
      deprecated: true          # Mark as deprecated
      deprecationWarning: "example.com/v1alpha1 Database is deprecated; use v1beta1"
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                engine:
                  type: string
                version:
                  type: string
                replicas:
                  type: integer
    - name: v1beta1
      served: true
      storage: true             # This is the storage version
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              required:
                - engine
                - version
              properties:
                engine:
                  type: string
                  enum: ["postgres", "mysql", "mongodb"]
                version:
                  type: string
                replicas:
                  type: integer
                  minimum: 1
                  maximum: 10
                  default: 1
                # New fields in v1beta1
                highAvailability:
                  type: object
                  properties:
                    enabled:
                      type: boolean
                      default: false
                    syncMode:
                      type: string
                      enum: ["sync", "async"]
                      default: "async"
```

### 4.2 스토리지 버전

스토리지 버전(storage version)은 etcd에 객체를 저장하는 데 사용되는 버전입니다. 한 번에 하나의 버전만 스토리지 버전이 될 수 있습니다.

```bash
# Check which version is the storage version
kubectl get crd databases.example.com -o jsonpath='{.status.storedVersions}'
# ["v1beta1"]

# When migrating storage versions:
# 1. Add new version with storage: true
# 2. Set old version to storage: false
# 3. Migrate existing objects: read and re-write each object
# 4. Remove old version from storedVersions
kubectl get db --all-namespaces -o json | kubectl replace -f -

# Verify migration
kubectl get crd databases.example.com -o jsonpath='{.status.storedVersions}'
# ["v1beta1"]
```

### 4.3 변환 웹훅

변환 웹훅(conversion webhook)은 클라이언트가 스토리지 버전과 다른 버전을 요청할 때 CRD 버전 간 변환을 수행합니다.

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: databases.example.com
spec:
  group: example.com
  conversion:
    strategy: Webhook
    webhook:
      clientConfig:
        service:
          name: database-conversion-webhook
          namespace: system
          path: /convert
          port: 443
        caBundle: LS0tLS1CRUdJTi...    # Base64-encoded CA certificate
      conversionReviewVersions:
        - v1
  # ... versions, names, scope
```

Go로 작성된 변환 웹훅 구현:

```go
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func handleConvert(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	var review apiextensionsv1.ConversionReview
	if err := json.Unmarshal(body, &review); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	desiredVersion := review.Request.DesiredAPIVersion
	convertedObjects := make([]unstructured.Unstructured, 0, len(review.Request.Objects))

	for _, raw := range review.Request.Objects {
		var obj unstructured.Unstructured
		if err := json.Unmarshal(raw.Raw, &obj); err != nil {
			sendError(w, &review, fmt.Sprintf("failed to unmarshal: %v", err))
			return
		}

		// Convert based on source and target versions
		converted, err := convert(&obj, desiredVersion)
		if err != nil {
			sendError(w, &review, fmt.Sprintf("conversion failed: %v", err))
			return
		}
		convertedObjects = append(convertedObjects, *converted)
	}

	review.Response = &apiextensionsv1.ConversionResponse{
		UID:              review.Request.UID,
		Result:           successStatus(),
		ConvertedObjects: toRawExtensions(convertedObjects),
	}
	review.Request = nil

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(review)
}

func convert(obj *unstructured.Unstructured, targetVersion string) (*unstructured.Unstructured, error) {
	sourceVersion := obj.GetAPIVersion()

	// v1alpha1 -> v1beta1: add highAvailability defaults
	if sourceVersion == "example.com/v1alpha1" && targetVersion == "example.com/v1beta1" {
		spec, _, _ := unstructured.NestedMap(obj.Object, "spec")
		if _, exists := spec["highAvailability"]; !exists {
			spec["highAvailability"] = map[string]interface{}{
				"enabled":  false,
				"syncMode": "async",
			}
		}
		unstructured.SetNestedMap(obj.Object, spec, "spec")
		obj.SetAPIVersion(targetVersion)
		return obj, nil
	}

	// v1beta1 -> v1alpha1: strip highAvailability
	if sourceVersion == "example.com/v1beta1" && targetVersion == "example.com/v1alpha1" {
		spec, _, _ := unstructured.NestedMap(obj.Object, "spec")
		delete(spec, "highAvailability")
		unstructured.SetNestedMap(obj.Object, spec, "spec")
		obj.SetAPIVersion(targetVersion)
		return obj, nil
	}

	return obj, nil
}

func main() {
	http.HandleFunc("/convert", handleConvert)
	http.ListenAndServeTLS(":8443", "/certs/tls.crt", "/certs/tls.key", nil)
}
```

---

## 5. 하위 리소스

### 5.1 Status 하위 리소스

status 하위 리소스(subresource)는 spec(원하는 상태, 사용자 쓰기 가능)과 status(관찰된 상태, 컨트롤러 쓰기 가능)를 분리합니다. status 하위 리소스가 활성화되면 `kubectl apply`로 `.status`를 수정할 수 없고, status 업데이트로 `.spec`을 수정할 수 없습니다.

```yaml
# In the CRD version definition:
versions:
  - name: v1beta1
    served: true
    storage: true
    subresources:
      status: {}              # Enable the status subresource
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              engine:
                type: string
              replicas:
                type: integer
          status:
            type: object
            properties:
              phase:
                type: string
                enum: ["Creating", "Running", "Failed", "Deleting"]
              readyReplicas:
                type: integer
              observedGeneration:
                type: integer
                format: int64
              conditions:
                type: array
                items:
                  type: object
                  required: ["type", "status"]
                  properties:
                    type:
                      type: string
                    status:
                      type: string
                      enum: ["True", "False", "Unknown"]
                    lastTransitionTime:
                      type: string
                      format: date-time
                    reason:
                      type: string
                    message:
                      type: string
```

```bash
# Update status from a controller (uses the /status subresource endpoint)
kubectl proxy &

curl -X PUT http://localhost:8001/apis/example.com/v1beta1/namespaces/default/databases/orders-db/status \
  -H "Content-Type: application/json" \
  -d '{
    "apiVersion": "example.com/v1beta1",
    "kind": "Database",
    "metadata": {
      "name": "orders-db",
      "namespace": "default"
    },
    "status": {
      "phase": "Running",
      "readyReplicas": 3,
      "observedGeneration": 1,
      "conditions": [
        {
          "type": "Ready",
          "status": "True",
          "lastTransitionTime": "2024-01-15T10:05:00Z",
          "reason": "AllReplicasReady",
          "message": "All 3 replicas are ready"
        }
      ]
    }
  }'
```

Go에서 status를 업데이트하는 방법(일반적인 컨트롤러 패턴):

```go
package main

import (
	"context"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
)

func updateDatabaseStatus(client dynamic.Interface, name, namespace string) error {
	gvr := schema.GroupVersionResource{
		Group:    "example.com",
		Version:  "v1beta1",
		Resource: "databases",
	}

	// Get the current resource
	db, err := client.Resource(gvr).Namespace(namespace).Get(
		context.TODO(), name, metav1.GetOptions{},
	)
	if err != nil {
		return err
	}

	// Update status fields
	status := map[string]interface{}{
		"phase":              "Running",
		"readyReplicas":      int64(3),
		"observedGeneration": db.GetGeneration(),
		"conditions": []interface{}{
			map[string]interface{}{
				"type":               "Ready",
				"status":             "True",
				"lastTransitionTime": time.Now().UTC().Format(time.RFC3339),
				"reason":             "AllReplicasReady",
				"message":            "All replicas are ready",
			},
		},
	}
	unstructured.SetNestedMap(db.Object, status, "status")

	// Update via the /status subresource
	_, err = client.Resource(gvr).Namespace(namespace).UpdateStatus(
		context.TODO(), db, metav1.UpdateOptions{},
	)
	return err
}

func main() {
	config, _ := rest.InClusterConfig()
	client, _ := dynamic.NewForConfig(config)
	updateDatabaseStatus(client, "orders-db", "default")
}
```

### 5.2 Scale 하위 리소스

scale 하위 리소스(subresource)는 커스텀 리소스에 대해 `kubectl scale`과 HPA 통합을 가능하게 합니다.

```yaml
versions:
  - name: v1beta1
    served: true
    storage: true
    subresources:
      status: {}
      scale:
        specReplicasPath: .spec.replicas
        statusReplicasPath: .status.readyReplicas
        labelSelectorPath: .status.selector    # Optional: for HPA
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              replicas:
                type: integer
                minimum: 1
                maximum: 10
          status:
            type: object
            properties:
              readyReplicas:
                type: integer
              selector:
                type: string
```

```bash
# Now kubectl scale works with your custom resource
kubectl scale db orders-db --replicas=5

# HPA can also target your custom resource
kubectl autoscale db orders-db --min=3 --max=10 --cpu-percent=80
```

```yaml
# HPA targeting a custom resource
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: orders-db-hpa
spec:
  scaleTargetRef:
    apiVersion: example.com/v1beta1
    kind: Database
    name: orders-db
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

---

## 6. 프린터 컬럼

프린터 컬럼(printer column)은 `kubectl get`의 테이블 출력에 표시되는 내용을 정의합니다.

```yaml
versions:
  - name: v1beta1
    served: true
    storage: true
    additionalPrinterColumns:
      - name: Engine
        type: string
        jsonPath: .spec.engine
        description: "Database engine type"
      - name: Version
        type: string
        jsonPath: .spec.version
        description: "Engine version"
      - name: Replicas
        type: integer
        jsonPath: .spec.replicas
        description: "Desired number of replicas"
      - name: Ready
        type: integer
        jsonPath: .status.readyReplicas
        description: "Number of ready replicas"
      - name: Phase
        type: string
        jsonPath: .status.phase
        description: "Current phase"
      - name: Storage
        type: string
        jsonPath: .spec.storage.size
        priority: 1              # Only shown with -o wide
        description: "Storage size"
      - name: Age
        type: date
        jsonPath: .metadata.creationTimestamp
```

```bash
# Default view
kubectl get db -n production
# NAME        ENGINE     VERSION   REPLICAS   READY   PHASE     AGE
# orders-db   postgres   16.2      3          3       Running   2d
# users-db    postgres   16.2      1          1       Running   5d

# Wide view (includes priority 1 columns)
kubectl get db -n production -o wide
# NAME        ENGINE     VERSION   REPLICAS   READY   PHASE     STORAGE   AGE
# orders-db   postgres   16.2      3          3       Running   100Gi     2d
# users-db    postgres   16.2      1          1       Running   10Gi      5d
```

---

## 7. 카테고리와 짧은 이름

```yaml
names:
  kind: Database
  listKind: DatabaseList
  singular: database
  plural: databases
  shortNames:
    - db                # kubectl get db
    - dbs               # kubectl get dbs
  categories:
    - all               # Included in kubectl get all
    - example           # kubectl get example (shows all resources in this category)
    - databases         # kubectl get databases (custom category)
```

```bash
# Short names
kubectl get db                  # same as kubectl get databases
kubectl get dbs                 # same as kubectl get databases

# Categories
kubectl get all -n production   # Now includes Database resources
kubectl get example             # Shows all resources with category "example"

# Check available short names
kubectl api-resources | grep database
# databases   db,dbs   example.com/v1beta1   true   Database
```

---

## 8. CRD 모범 사례

### 8.1 설계 가이드라인

| 가이드라인 | 근거 |
|-----------|-----------|
| API 그룹에 소유하고 있는 도메인 사용 | 충돌 방지 (예: `example.com`, `k8s.io` 아님) |
| `v1alpha1`로 시작 | 불안정성을 전달; API가 성숙해지면 승격 |
| spec과 status 분리 | status 하위 리소스 사용; 컨트롤러가 status 업데이트 |
| spec 필드를 선언적으로 만들기 | 명령적 동작이 아닌 원하는 상태 기술 |
| status에 condition 사용 | 표준 `type`/`status`/`reason`/`message` 패턴 따르기 |
| status에 `observedGeneration` 설정 | 컨트롤러가 처리한 마지막 spec 세대(generation) 보고 |
| 프린터 컬럼 포함 | 더 나은 `kubectl get` 경험 |
| 유효성 검증 규칙 추가 | 조기 오류 포착; 교차 필드 검증에 CEL 사용 |
| 모든 필드 문서화 | 스키마에서 `description` 사용 |
| 정리를 위한 파이널라이저(finalizer) 사용 | 외부 리소스가 정리될 때까지 삭제 방지 |

### 8.2 스키마 진화

```
v1alpha1 ──▶ v1alpha2 ──▶ v1beta1 ──▶ v1
   │              │            │         │
   │  Breaking    │  Breaking  │  No     │  Stable
   │  changes     │  changes   │  breaking│  API
   │  allowed     │  allowed   │  changes │
   │              │            │         │
   │  Short       │  Longer    │  Long    │  Indefinite
   │  support     │  support   │  support │  support
```

각 안정성 수준에 대한 규칙:

- **v1alpha1/v1alpha2**: 호환성이 깨지는 변경(breaking change) 허용. 마이그레이션 경로 없이 삭제될 수 있음.
- **v1beta1**: 호환성이 깨지는 변경 권장하지 않음. 마이그레이션 경로를 제공해야 함.
- **v1**: 호환성이 깨지는 변경 불가. 필드를 추가할 수 있지만 제거하거나 이름을 변경할 수 없음.

### 8.3 성능 고려사항

```yaml
# Large CRDs: set x-kubernetes-list-type for efficient updates
properties:
  spec:
    type: object
    properties:
      endpoints:
        type: array
        x-kubernetes-list-type: map       # Merge by key, not replace entire list
        x-kubernetes-list-map-keys:
          - name
        items:
          type: object
          required: ["name"]
          properties:
            name:
              type: string
            address:
              type: string
            port:
              type: integer
```

```bash
# Monitor CRD storage usage
kubectl get --raw /metrics | grep apiserver_storage_objects
# apiserver_storage_objects{resource="databases.example.com"} 150

# Watch for excessive object size
kubectl get db -A -o json | wc -c
# Keep individual objects under 1.5MB (etcd limit)
```

---

## 9. 집계 API 서버 vs CRD

### 이론: CRD vs Aggregated API Server

언제 CRD로는 부족해질까요? 세 가지 케이스가 aggregated API server로 밀어붙입니다:

- **etcd가 아닌 스토리지.** 모든 CRD는 클러스터의 etcd에 영속화됩니다. 리소스를 관계형 데이터베이스, 외부 서비스, 또는 read 시 생성되는 데이터(metrics API server처럼 — 어디에도 저장되지 않은 노드/파드 메트릭을 노출하며, kubelet에서 라이브로 계산)로 백업해야 한다면 CRD는 잘못된 선택입니다. API 표면을 직접 구현해야 합니다.
- **JSON CRUD를 넘는 커스텀 프로토콜.** 스트리밍 subresource(`kubectl exec`을 생각하세요), 임의의 HTTP 시맨틱, 또는 쿠버네티스에 부합하지 않는 요청 본문은 aggregated server를 필요로 합니다.
- **역사적 또는 조직적 이유.** CRD 이전의 일부 쿠버네티스 코어 API(apiregistration, certificates)는 레거시 호환성을 위해 aggregated 됩니다.

Aggregated API server의 비용은 상당합니다 — 실제 HTTPS 서버를 실행(인증, 감사, watch 구현 포함)하고, 자체 RBAC를 유지하며, 또 다른 컨트롤 플레인 컴포넌트의 운영 부담을 처리. 따라서 위 케이스 중 하나에 부딪히지 않는 한, CRD가 단순함에서 이깁니다.

흔한 패턴 — 사용자 대면 객체용 CRD를 출시하고, 실제 일을 하는 **컨트롤러**(11강)를 함께. CRD는 객체를 `kubectl`에서 일급 시민으로 만들고, 컨트롤러는 사용자 의도를 현실로 바꿉니다.

### 9.1 각각의 사용 시기

| 요소 | CRD | 집계 API 서버(Aggregated API Server) |
|--------|------|----------------------|
| **구현 노력** | YAML 매니페스트만 | 완전한 Go API 서버 |
| **스토리지** | etcd (API 서버 경유) | 커스텀 (모든 백엔드) |
| **유효성 검증** | OpenAPI 스키마 + CEL | 임의의 Go 코드 |
| **인증** | Kubernetes 네이티브 | 커스텀 또는 Kubernetes |
| **하위 리소스** | status, scale만 | 모든 하위 리소스 |
| **API 디스커버리** | 자동 | APIService 등록을 통해 |
| **장시간 요청** | 지원하지 않음 | 지원 (watch, exec) |
| **Protobuf 지원** | 아니오 (JSON만) | 예 |
| **사용 사례** | 대부분의 커스텀 리소스 | metrics-server, custom-metrics |

**CRD를 선택하는 경우:**
- 데이터가 Kubernetes 리소스 모델(metadata, spec, status)에 적합한 경우
- 표준 CRUD 작업이 필요한 경우
- OpenAPI 유효성 검증 + CEL이 충분한 경우
- 운영 오버헤드를 최소화하고 싶은 경우

**집계 API 서버를 선택하는 경우:**
- 커스텀 스토리지 백엔드(etcd가 아닌)가 필요한 경우
- status와 scale 이외의 하위 리소스가 필요한 경우
- 성능을 위해 protobuf이 필요한 경우
- 스트리밍이나 장시간 요청이 필요한 경우

### 9.2 집계 API 서버 예제

```yaml
# Register an aggregated API server with Kubernetes
apiVersion: apiregistration.k8s.io/v1
kind: APIService
metadata:
  name: v1beta1.custom.metrics.k8s.io
spec:
  service:
    name: custom-metrics-apiserver
    namespace: monitoring
    port: 443
  group: custom.metrics.k8s.io
  version: v1beta1
  insecureSkipTLSVerify: false
  caBundle: LS0tLS1CRUdJTi...
  groupPriorityMinimum: 100
  versionPriority: 100
```

```bash
# The metrics-server is a real-world example of an aggregated API server
kubectl get apiservices | grep metrics
# v1beta1.metrics.k8s.io   kube-system/metrics-server   True   30d

# It serves custom API endpoints
kubectl get --raw /apis/metrics.k8s.io/v1beta1/nodes
kubectl top nodes   # Uses the aggregated API behind the scenes
```

---

## 연습문제

### 연습문제 1: 캐시 리소스용 CRD 생성

`infrastructure.example.com` 그룹에 다음 필드를 가진 `Cache` 리소스용 CRD를 생성하세요:
- `engine` (필수, enum: redis, memcached)
- `version` (필수, string)
- `replicas` (기본값: 1, 최소: 1, 최대: 5)
- `memory` (string, "256Mi", "1Gi"와 같은 크기 패턴)
- `evictionPolicy` (enum: noeviction, allkeys-lru, volatile-lru)

engine, version, replicas, age에 대한 프린터 컬럼을 포함하세요.

<details><summary>정답 보기</summary>

```yaml
# cache-crd.yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: caches.infrastructure.example.com
spec:
  group: infrastructure.example.com
  names:
    kind: Cache
    listKind: CacheList
    singular: cache
    plural: caches
    shortNames:
      - ca
    categories:
      - all
      - infrastructure
  scope: Namespaced
  versions:
    - name: v1alpha1
      served: true
      storage: true
      subresources:
        status: {}
      additionalPrinterColumns:
        - name: Engine
          type: string
          jsonPath: .spec.engine
        - name: Version
          type: string
          jsonPath: .spec.version
        - name: Replicas
          type: integer
          jsonPath: .spec.replicas
        - name: Memory
          type: string
          jsonPath: .spec.memory
          priority: 1
        - name: Age
          type: date
          jsonPath: .metadata.creationTimestamp
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              required:
                - engine
                - version
              properties:
                engine:
                  type: string
                  enum: ["redis", "memcached"]
                version:
                  type: string
                replicas:
                  type: integer
                  minimum: 1
                  maximum: 5
                  default: 1
                memory:
                  type: string
                  pattern: "^[0-9]+(Mi|Gi)$"
                  default: "256Mi"
                evictionPolicy:
                  type: string
                  enum: ["noeviction", "allkeys-lru", "volatile-lru"]
                  default: "allkeys-lru"
            status:
              type: object
              properties:
                phase:
                  type: string
                readyReplicas:
                  type: integer
```

```bash
kubectl apply -f cache-crd.yaml

# Create a cache instance
cat <<EOF | kubectl apply -f -
apiVersion: infrastructure.example.com/v1alpha1
kind: Cache
metadata:
  name: session-cache
  namespace: default
spec:
  engine: redis
  version: "7.2"
  replicas: 3
  memory: "1Gi"
  evictionPolicy: volatile-lru
EOF

# Verify
kubectl get caches
# NAME            ENGINE   VERSION   REPLICAS   AGE
# session-cache   redis    7.2       3          5s

kubectl get ca -o wide
# NAME            ENGINE   VERSION   REPLICAS   MEMORY   AGE
# session-cache   redis    7.2       3          1Gi      10s

# Test validation
cat <<EOF | kubectl apply -f -
apiVersion: infrastructure.example.com/v1alpha1
kind: Cache
metadata:
  name: bad-cache
spec:
  engine: couchbase
  version: "1.0"
EOF
# Error: spec.engine: Unsupported value: "couchbase": supported values: "redis", "memcached"
```

</details>

### 연습문제 2: CEL 유효성 검증 규칙 추가

연습문제 1의 Cache CRD에 CEL 유효성 검증 규칙을 추가하세요:
- Engine은 생성 후 변경 불가(immutable)
- engine이 "memcached"인 경우 replicas는 정확히 1이어야 함 (memcached는 기본적으로 클러스터링을 지원하지 않음)
- memory는 redis의 경우 최소 "128Mi", memcached의 경우 최소 "64Mi"이어야 함

<details><summary>정답 보기</summary>

```yaml
# cache-crd-validated.yaml (relevant spec.versions section)
versions:
  - name: v1alpha1
    served: true
    storage: true
    subresources:
      status: {}
    additionalPrinterColumns:
      - name: Engine
        type: string
        jsonPath: .spec.engine
      - name: Version
        type: string
        jsonPath: .spec.version
      - name: Replicas
        type: integer
        jsonPath: .spec.replicas
      - name: Age
        type: date
        jsonPath: .metadata.creationTimestamp
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            required:
              - engine
              - version
            x-kubernetes-validations:
              # Engine is immutable
              - rule: "self.engine == oldSelf.engine"
                message: "Engine type cannot be changed after creation"
              # Memcached: replicas must be 1
              - rule: "self.engine != 'memcached' || self.replicas == 1"
                message: "Memcached does not support clustering; replicas must be 1"
              # Memory minimum based on engine
              - rule: >-
                  self.engine != 'redis' ||
                  (self.memory.endsWith('Gi') ||
                   (self.memory.endsWith('Mi') &&
                    int(self.memory.replace('Mi','')) >= 128))
                message: "Redis requires at least 128Mi of memory"
              - rule: >-
                  self.engine != 'memcached' ||
                  (self.memory.endsWith('Gi') ||
                   (self.memory.endsWith('Mi') &&
                    int(self.memory.replace('Mi','')) >= 64))
                message: "Memcached requires at least 64Mi of memory"
            properties:
              engine:
                type: string
                enum: ["redis", "memcached"]
              version:
                type: string
              replicas:
                type: integer
                minimum: 1
                maximum: 5
                default: 1
              memory:
                type: string
                pattern: "^[0-9]+(Mi|Gi)$"
                default: "256Mi"
              evictionPolicy:
                type: string
                enum: ["noeviction", "allkeys-lru", "volatile-lru"]
                default: "allkeys-lru"
          status:
            type: object
            properties:
              phase:
                type: string
              readyReplicas:
                type: integer
```

```bash
kubectl apply -f cache-crd-validated.yaml

# Test: memcached with replicas > 1 should fail
cat <<EOF | kubectl apply -f -
apiVersion: infrastructure.example.com/v1alpha1
kind: Cache
metadata:
  name: bad-memcached
spec:
  engine: memcached
  version: "1.6"
  replicas: 3
EOF
# Error: Memcached does not support clustering; replicas must be 1

# Test: redis with insufficient memory should fail
cat <<EOF | kubectl apply -f -
apiVersion: infrastructure.example.com/v1alpha1
kind: Cache
metadata:
  name: small-redis
spec:
  engine: redis
  version: "7.2"
  memory: "64Mi"
EOF
# Error: Redis requires at least 128Mi of memory

# Test: changing engine should fail
kubectl patch cache session-cache --type=merge -p '{"spec":{"engine":"memcached"}}'
# Error: Engine type cannot be changed after creation
```

</details>

### 연습문제 3: Status 하위 리소스를 가진 CRD

condition을 포함하는 status 하위 리소스가 있는 `Backup` 리소스용 CRD를 생성하세요. API를 통해 status를 업데이트하는 컨트롤러를 시뮬레이션하는 셸 스크립트를 작성하세요.

<details><summary>정답 보기</summary>

```yaml
# backup-crd.yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: backups.storage.example.com
spec:
  group: storage.example.com
  names:
    kind: Backup
    plural: backups
    singular: backup
    shortNames: ["bk"]
  scope: Namespaced
  versions:
    - name: v1alpha1
      served: true
      storage: true
      subresources:
        status: {}
      additionalPrinterColumns:
        - name: Source
          type: string
          jsonPath: .spec.source
        - name: Phase
          type: string
          jsonPath: .status.phase
        - name: Size
          type: string
          jsonPath: .status.backupSize
        - name: Age
          type: date
          jsonPath: .metadata.creationTimestamp
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              required: ["source", "destination"]
              properties:
                source:
                  type: string
                destination:
                  type: string
                schedule:
                  type: string
            status:
              type: object
              properties:
                phase:
                  type: string
                  enum: ["Pending", "InProgress", "Completed", "Failed"]
                backupSize:
                  type: string
                startedAt:
                  type: string
                  format: date-time
                completedAt:
                  type: string
                  format: date-time
                conditions:
                  type: array
                  items:
                    type: object
                    required: ["type", "status"]
                    properties:
                      type:
                        type: string
                      status:
                        type: string
                      lastTransitionTime:
                        type: string
                        format: date-time
                      reason:
                        type: string
                      message:
                        type: string
```

```bash
kubectl apply -f backup-crd.yaml

# Create a backup resource
cat <<EOF | kubectl apply -f -
apiVersion: storage.example.com/v1alpha1
kind: Backup
metadata:
  name: daily-backup
  namespace: default
spec:
  source: orders-db
  destination: s3://my-bucket/backups
  schedule: "0 2 * * *"
EOF

# Simulate controller updating status
# Start kubectl proxy in the background
kubectl proxy --port=8001 &
PROXY_PID=$!

# Get the current resource for the resourceVersion
RESOURCE=$(curl -s http://localhost:8001/apis/storage.example.com/v1alpha1/namespaces/default/backups/daily-backup)
RV=$(echo "$RESOURCE" | jq -r '.metadata.resourceVersion')

# Update status to InProgress
curl -s -X PUT \
  http://localhost:8001/apis/storage.example.com/v1alpha1/namespaces/default/backups/daily-backup/status \
  -H "Content-Type: application/json" \
  -d "{
    \"apiVersion\": \"storage.example.com/v1alpha1\",
    \"kind\": \"Backup\",
    \"metadata\": {
      \"name\": \"daily-backup\",
      \"namespace\": \"default\",
      \"resourceVersion\": \"$RV\"
    },
    \"status\": {
      \"phase\": \"InProgress\",
      \"startedAt\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
      \"conditions\": [
        {
          \"type\": \"Started\",
          \"status\": \"True\",
          \"lastTransitionTime\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
          \"reason\": \"BackupStarted\",
          \"message\": \"Backup process has started\"
        }
      ]
    }
  }" | jq .

# Verify status
kubectl get bk daily-backup
# NAME           SOURCE      PHASE        SIZE   AGE
# daily-backup   orders-db   InProgress          30s

# Clean up proxy
kill $PROXY_PID
```

</details>

### 연습문제 4: 다중 버전 CRD

두 개의 버전(`v1alpha1`과 `v1beta1`)을 가진 CRD를 생성하세요. `v1beta1` 버전은 `v1alpha1`에 없는 `monitoring` 필드를 추가합니다. 두 버전을 사용하여 리소스를 생성하고 어느 버전에서든 접근 가능한지 확인하세요.

<details><summary>정답 보기</summary>

```yaml
# multi-version-crd.yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: appconfigs.config.example.com
spec:
  group: config.example.com
  names:
    kind: AppConfig
    plural: appconfigs
    singular: appconfig
    shortNames: ["ac"]
  scope: Namespaced
  versions:
    - name: v1alpha1
      served: true
      storage: false
      deprecated: true
      deprecationWarning: "config.example.com/v1alpha1 AppConfig is deprecated; migrate to v1beta1"
      additionalPrinterColumns:
        - name: Port
          type: integer
          jsonPath: .spec.port
        - name: Age
          type: date
          jsonPath: .metadata.creationTimestamp
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              required: ["appName", "port"]
              properties:
                appName:
                  type: string
                port:
                  type: integer
                replicas:
                  type: integer
                  default: 1
            status:
              type: object
              properties:
                phase:
                  type: string
    - name: v1beta1
      served: true
      storage: true
      additionalPrinterColumns:
        - name: Port
          type: integer
          jsonPath: .spec.port
        - name: Monitoring
          type: boolean
          jsonPath: .spec.monitoring.enabled
        - name: Age
          type: date
          jsonPath: .metadata.creationTimestamp
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              required: ["appName", "port"]
              properties:
                appName:
                  type: string
                port:
                  type: integer
                replicas:
                  type: integer
                  default: 1
                monitoring:
                  type: object
                  default: {}
                  properties:
                    enabled:
                      type: boolean
                      default: false
                    metricsPath:
                      type: string
                      default: "/metrics"
                    interval:
                      type: string
                      default: "30s"
            status:
              type: object
              properties:
                phase:
                  type: string
```

```bash
kubectl apply -f multi-version-crd.yaml

# Create a resource using v1alpha1 (deprecated)
cat <<EOF | kubectl apply -f -
apiVersion: config.example.com/v1alpha1
kind: AppConfig
metadata:
  name: legacy-app
spec:
  appName: legacy-service
  port: 8080
  replicas: 2
EOF
# Warning: config.example.com/v1alpha1 AppConfig is deprecated; migrate to v1beta1

# Create a resource using v1beta1
cat <<EOF | kubectl apply -f -
apiVersion: config.example.com/v1beta1
kind: AppConfig
metadata:
  name: modern-app
spec:
  appName: modern-service
  port: 9090
  replicas: 3
  monitoring:
    enabled: true
    metricsPath: "/api/metrics"
    interval: "15s"
EOF

# Access v1alpha1-created resource via v1beta1 API
kubectl get appconfigs.v1beta1.config.example.com legacy-app -o yaml
# The monitoring field will have defaults applied

# Access v1beta1-created resource via v1alpha1 API
kubectl get appconfigs.v1alpha1.config.example.com modern-app -o yaml
# The monitoring field is present but not in v1alpha1 schema

# List all AppConfigs (shows both)
kubectl get ac
# NAME          PORT   MONITORING   AGE
# legacy-app    8080   false        30s
# modern-app    9090   true         15s
```

</details>

### 연습문제 5: Scale 하위 리소스를 가진 CRD

scale 하위 리소스를 지원하는 `WorkerPool` 리소스용 CRD를 생성하세요. `kubectl scale`을 사용할 수 있고 HPA가 이를 대상으로 할 수 있는지 확인하세요.

<details><summary>정답 보기</summary>

```yaml
# workerpool-crd.yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: workerpools.compute.example.com
spec:
  group: compute.example.com
  names:
    kind: WorkerPool
    plural: workerpools
    singular: workerpool
    shortNames: ["wp"]
    categories:
      - all
  scope: Namespaced
  versions:
    - name: v1alpha1
      served: true
      storage: true
      subresources:
        status: {}
        scale:
          specReplicasPath: .spec.workers
          statusReplicasPath: .status.readyWorkers
          labelSelectorPath: .status.labelSelector
      additionalPrinterColumns:
        - name: Workers
          type: integer
          jsonPath: .spec.workers
        - name: Ready
          type: integer
          jsonPath: .status.readyWorkers
        - name: Task
          type: string
          jsonPath: .spec.taskType
        - name: Age
          type: date
          jsonPath: .metadata.creationTimestamp
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              required: ["taskType", "workers"]
              properties:
                taskType:
                  type: string
                  enum: ["batch", "stream", "ml-training"]
                workers:
                  type: integer
                  minimum: 0
                  maximum: 100
                workerTemplate:
                  type: object
                  properties:
                    image:
                      type: string
                    resources:
                      type: object
                      properties:
                        cpu:
                          type: string
                        memory:
                          type: string
            status:
              type: object
              properties:
                readyWorkers:
                  type: integer
                labelSelector:
                  type: string
                conditions:
                  type: array
                  items:
                    type: object
                    required: ["type", "status"]
                    properties:
                      type:
                        type: string
                      status:
                        type: string
```

```bash
kubectl apply -f workerpool-crd.yaml

# Create a WorkerPool
cat <<EOF | kubectl apply -f -
apiVersion: compute.example.com/v1alpha1
kind: WorkerPool
metadata:
  name: data-processors
  namespace: default
spec:
  taskType: batch
  workers: 5
  workerTemplate:
    image: my-worker:v1
    resources:
      cpu: "500m"
      memory: "1Gi"
EOF

# Scale using kubectl
kubectl scale wp data-processors --replicas=10
kubectl get wp data-processors
# NAME              WORKERS   READY   TASK    AGE
# data-processors   10                batch   30s

# Verify the scale subresource works
kubectl get --raw /apis/compute.example.com/v1alpha1/namespaces/default/workerpools/data-processors/scale | jq .
# {
#   "kind": "Scale",
#   "apiVersion": "autoscaling/v1",
#   "metadata": { "name": "data-processors", ... },
#   "spec": { "replicas": 10 },
#   "status": { "replicas": 0 }
# }

# Create an HPA targeting the WorkerPool
cat <<EOF | kubectl apply -f -
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: data-processors-hpa
spec:
  scaleTargetRef:
    apiVersion: compute.example.com/v1alpha1
    kind: WorkerPool
    name: data-processors
  minReplicas: 3
  maxReplicas: 50
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 75
EOF

kubectl get hpa data-processors-hpa
# NAME                    REFERENCE                    TARGETS   MINPODS   MAXPODS
# data-processors-hpa     WorkerPool/data-processors   <unknown> 3         50
```

</details>

---

**이전**: [Helm과 Kustomize](./09_Helm_and_Kustomize.md) | **다음**: [오퍼레이터](./11_Operators.md)

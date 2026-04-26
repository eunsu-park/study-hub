# 16. Kubernetes API 프로그래밍

**이전**: [15. 멀티 클러스터](./15_Multi_Cluster.md) | **다음**: [17. 프로덕션 운영](./17_Production_Operations.md)

## 학습 목표

- 그룹(Group), 버전(Version), 리소스(Resource)를 포함한 Kubernetes API 구조 이해
- client-go를 사용하여 Go에서 Kubernetes API와 프로그래밍 방식으로 상호작용
- 인포머(Informer), 워크 큐(Work Queue), 조정 루프(Reconciliation Loop)를 사용한 커스텀 컨트롤러 구축
- 프로덕션 수준의 컨트롤러 개발을 위한 controller-runtime 라이브러리 활용
- envtest과 통합 테스트 패턴을 사용한 컨트롤러 테스트

---

Kubernetes API 서버는 모든 클러스터의 중앙 허브입니다. 모든 `kubectl` 명령, 모든 컨트롤러, 모든 오퍼레이터(Operator)가 이 단일 RESTful 인터페이스를 통해 통신합니다. Kubernetes API에 대한 프로그래밍 방법을 이해하면 커스텀 자동화를 구축하고, 새로운 동작으로 플랫폼을 확장하며, Kubernetes를 더 큰 시스템에 통합할 수 있습니다. 이 레슨에서는 API 서버와 상호작용하는 Go 프로그램을 작성하는 방법을 배웁니다 -- 간단한 CRUD 작업부터 리소스를 감시하고 지속적으로 상태를 조정하는 본격적인 컨트롤러까지.

Go 코드에 들어가기 전에, [**이론과 원리**](#이론과-원리) 섹션을 먼저 읽어보세요. 모든 쿠버네티스 리소스를 카탈로그화하는 GVR/GVK 시스템, client-go의 typed clientset과 dynamic client가 컴파일 시간 안전성과 일반성을 트레이드하는 이유, 모든 컨트롤러를 구동하는 informer + work queue 패턴, 그리고 실제 클러스터 없이 reconciliation 로직을 검증하게 해주는 테스팅 전략(envtest, fake client)을 다룹니다.

## 목차

- [이론과 원리](#이론과-원리)
- [1. Kubernetes API 구조](#1-kubernetes-api-구조)
- [2. client-go 라이브러리](#2-client-go-라이브러리)
- [3. REST 클라이언트와 Clientsets](#3-rest-클라이언트와-clientsets)
- [4. 다이나믹 클라이언트와 비구조화 객체](#4-다이나믹-클라이언트와-비구조화-객체)
- [5. 인포머와 캐싱](#5-인포머와-캐싱)
- [6. 워크 큐](#6-워크-큐)
- [7. 커스텀 컨트롤러 구축](#7-커스텀-컨트롤러-구축)
- [8. Controller-Runtime 라이브러리](#8-controller-runtime-라이브러리)
- [9. 리소스 감시와 이벤트 처리](#9-리소스-감시와-이벤트-처리)
- [10. 컨트롤러 테스트](#10-컨트롤러-테스트)
- [연습문제](#연습문제)

---

## 이론과 원리

Go에서 쿠버네티스 API를 프로그래밍하는 것은 당신이 사용한 모든 컨트롤러, operator, 플랫폼 도구(Argo CD, cert-manager, Prometheus Operator, ...)가 내부에서 하는 일입니다. API 서버 자체는 RESTful HTTP+JSON 서비스 — `curl`로 호출할 수 있습니다 — 그러나 프로덕션 코드는 typed 접근, 캐싱(informer), 효율적 변경 알림(watch), 그리고 재시작과 동시성 하에서 reconciliation을 정확하게 만드는 work-queue 패턴을 제공하므로 **client-go** 라이브러리를 사용합니다. 이 섹션은 리소스 분류(GVR/GVK), 클라이언트 선택, informer 아키텍처(11강의 operator-runtime의 기반이기도 함), 그리고 취미 코드와 프로덕션 컨트롤러를 구분하는 테스팅 접근을 설명합니다.

### A. 리소스 분류 — GVR과 GVK

모든 쿠버네티스 리소스는 두 평행 식별을 가집니다:

**GVK (Group, Version, Kind)**는 *Go 타입*을 식별합니다 — 예 — `apps/v1.Deployment`. 이는 코드가 구성하고 검사하는 것입니다(`appsv1.Deployment{...}`). Kind는 PascalCase이고 단수.

**GVR (Group, Version, Resource)**는 *REST URL 경로*를 식별합니다 — 예 — `apps/v1/deployments`. 이는 URL(`/apis/apps/v1/namespaces/default/deployments`)에 나타나고 RBAC 규칙이 참조하는 것(`apiGroups: ["apps"], resources: ["deployments"]`)입니다. Resource는 소문자이고 복수.

둘 사이의 매핑은 API 서버의 **discovery** 엔드포인트를 통하며, 이는 등록된 모든 (group, version)과 그 안의 kind와 resource를 나열합니다. 라이브러리 `RESTMapper`가 이 조회를 대신 해주므로 `meta.RESTMapper.RESTMapping(GroupKind, version)`을 작성하고 올바른 URL fragment를 돌려받을 수 있습니다.

왜 둘? 와이어 형식과 인메모리 표현이 독립적으로 진화하기 때문입니다. `Deployment` Kind는 항상 "동일한 개념적 객체"를 의미하지만, REST resource 경로는 (원칙적으로) API 버전 간에 변경될 수 있습니다. 대부분의 코드는 Go에서 Kind(`*appsv1.Deployment`)를 사용하고 RBAC와 dynamic-client 계층에서만 Resource를 다룹니다.

### B. 세 가지 클라이언트 스타일 — Typed, Discovery, Dynamic

client-go는 API 서버와 통신하는 세 가지 방법을 제공합니다:

**1. Typed clientset (`kubernetes.Clientset`)** — 내장 리소스를 위한 Go 타입 인터페이스. `clientset.AppsV1().Deployments("default").Get(ctx, "my-app", metav1.GetOptions{})`를 작성하고 `*appsv1.Deployment`를 돌려받습니다. 컴파일 시간 안전성, IDE 자동 완성, 리팩토링 쉬움. **한계** — clientset 컴파일 시점에 타입이 알려진 리소스에만 동작 — 내장과 typed 클라이언트를 생성한 CRD.

**2. Dynamic client (`dynamic.Interface`)** — `unstructured.Unstructured`(`map[string]interface{}`)에 동작합니다. GVR을 구성하고, `ResourceInterface`를 가져오고, `*unstructured.Unstructured` 객체에 작업합니다. 빌드 시점에 알려지지 않은 임의의 CRD를 코드 생성 없이 다룰 능력을 위해 컴파일 시간 안전성을 트레이드. **사용 시기** — 빌드 시점에 알려지지 않은 사용자 제공 CRD를 다루는 generic operator(Argo CD 같은) 작성.

**3. controller-runtime client (`client.Client`)** — 11강에서 도입; clientset 위에 빌드되었지만 런타임 등록을 통해 내장과 커스텀 타입에 통합됨. Reconciler 패턴과 깔끔히 통합되므로 새 컨트롤러의 표준입니다.

각각 뒤에는 HTTP, 인증, 콘텐츠 협상(JSON vs protobuf), rate limiting을 처리하는 `rest.RESTClient`가 있습니다. 이 계층과 직접 상호작용하는 일은 거의 없습니다 — 더 높은 수준의 클라이언트가 그것을 감쌉니다.

**Discovery client** (`discovery.DiscoveryInterface`)는 네 번째, 특수 목적 클라이언트입니다 — 사용 가능한 group/version/resource의 API 서버 목록을 반환합니다. "이 클러스터에서 무엇과 작업할 수 있는가?"를 열거해야 하는 도구에 유용합니다.

### C. Informer 아키텍처 — List-Watch + 캐시 + 인덱싱된 읽기

모든 컨트롤러는 변경에 대해 리소스를 watch해야 합니다. 순진하게 하면(컨트롤러당 리소스당 watch HTTP 연결) 스케일하지 않습니다. **informer** 패턴이 이를 공유 캐시로 해결합니다:

```
API Server ←─watch─ Informer ─→ Indexer (캐시) ─→ Lister
                       │
                       └─→ Event Handler ─→ Work Queue ─→ Reconciler
```

**SharedInformerFactory**가 (resource, namespace)당 하나의 informer를 만들고 모든 consumer 간에 공유합니다. 따라서 operator가 Deployment를 watch하고 CRD 컨트롤러도 Deployment를 watch하면, 단 하나의 watch HTTP 연결만 열립니다. 팩토리는 참조 카운트를 추적하고 consumer가 남지 않으면 정리합니다.

**Indexer**가 로컬 캐시입니다. 초기 list 결과와 모든 후속 watch 델타를 보관합니다. 읽기(Get, List)는 indexer를 히트 — API 서버가 아닙니다 — 즉, 컨트롤러는 멀티 메가바이트 API 서버 왕복 대신 마이크로초 안에 로컬에서 10,000개 파드를 list할 수 있습니다. 빠른 조회를 위해 레이블, 필드, 또는 임의 함수에 커스텀 인덱스를 빌드할 수 있습니다("스캔 없이 ReplicaSet X가 소유한 모든 파드 줘").

**Event handler**는 `ADDED`/`MODIFIED`/`DELETED`에서 호출되는 사용자 제공 콜백입니다. 표준 패턴 — 핸들러는 일을 *하지 않습니다* — 키(`namespace/name`)를 추출하고 work queue에 `Add()`. 이는 이벤트 속도와 작업 속도를 분리합니다 — 폭발 이벤트는 큐에 흡수되고, 작업은 reconciler 페이스로 진행됩니다.

**Work queue** (`workqueue.RateLimitingInterface`)는 정확한 컨트롤러에 중요한 세 속성을 제공합니다:
- **중복 제거** — 같은 키에 대한 100개 이벤트는 하나의 reconcile이 됨.
- **키별 직렬화** — 주어진 키에 대해 한 번에 하나의 worker만 reconcile.
- **Rate limiting** — 실패한 reconcile은 exponentially backoff.

**Reconciler**는 당신의 코드입니다. 큐에서 키를 가져오고, indexer에서 현재 객체를 가져오고, desired 상태를 계산하고, 행동합니다. 오류 시 키를 큐로 반환하여 재시도; 성공 시 잊습니다. 이는 11강과 동일한 패턴 — 여기서는 더 낮은 수준의 client-go 관점에서 봅니다.

### D. 컨트롤러 테스팅 — envtest, Fake Client, 그리고 둘 다 존재하는 이유

컨트롤러는 API 서버의 동작 — 어드미션, defaulting, status 업데이트, watch 시맨틱 — 에 의존하기에 악명 높게 테스트하기 어렵습니다. 두 보완적 접근:

**Fake client** (`fake.NewSimpleClientset`) — 작업을 기록하고 미리 정해진 응답을 반환하는 clientset 인터페이스의 인메모리 구현. 장점 — 매우 빠름(작업당 마이크로초), 외부 의존성 없음, "컨트롤러가 X로 Update 호출했음"을 주장하기 쉬움. 단점 — 어드미션을 실행하지 않음, 스키마를 강제하지 않음, goroutine을 가로질러 watch 이벤트를 적절히 생성하지 않음. 순수 reconciler 로직의 단위 테스트에 가장 적합.

**envtest** (controller-runtime) — 테스트 프로세스에서 실제 `etcd`와 `kube-apiserver` 바이너리를 부팅. 장점 — 어드미션, 검증, defaulting, watch를 포함한 실제 API 동작 행사. Reconciler가 실제 API 서버에 대해 실행됩니다. 단점 — 더 느림(약 5초 시작, 작업당 약 100ms); kubebuilder envtest 바이너리 설치 필요. 컨트롤러 동작의 종단 간 통합 테스트에 가장 적합.

흔한 테스트 레이아웃 — fake client로 reconciler 로직에 대한 빠른 단위 테스트(CI에서 fail-fast), 더해 전체 reconciler-API 상호작용을 행사하는 envtest 기반 통합 테스트의 더 작은 모음(느리지만 높은 신뢰).

미묘한 점 — fake-client 테스트는 통과하지만 envtest 테스트는 실패하는 컨트롤러는 보통 fake client가 시뮬레이트하지 않는 무언가(어드미션 웹훅, server-side apply 시맨틱, watch 이벤트 순서)에 의존하는 것입니다. "테스트에서는 동작, 클러스터에서는 실패" 디버깅 시, envtest가 진실에 더 가깝습니다.

### 이론에서 아래의 코드로

이제 레슨은 이 추상을 적용합니다:

- **섹션 1 (Kubernetes API 구조)**는 §A입니다 — GVR/GVK, discovery, API 서버의 리소스 그래프.
- **섹션 2 (client-go 라이브러리)**는 §B의 개요입니다 — 패키지 레이아웃과 상위 수준 설계.
- **섹션 3 (REST 클라이언트와 Clientsets)**은 구체적 코드의 §B의 typed client.
- **섹션 4 (다이나믹 클라이언트와 비구조화 객체)**는 generic 도구를 위한 §B의 dynamic client.
- **섹션 5 (인포머와 캐싱)**은 코드의 `SharedInformerFactory`와 함께한 §C의 informer 아키텍처.
- **섹션 6 (워크 큐)**는 rate limiting과 키별 직렬화를 가진 §C의 큐.
- **섹션 7 (커스텀 컨트롤러 구축)**은 §C를 함께 꿰맵니다 — 실행 가능한 프로그램의 informer + 큐 + reconciler.
- **섹션 8 (Controller-Runtime 라이브러리)**는 동일한 프리미티브 위의 더 높은 수준 추상(11강)입니다.
- **섹션 9 (리소스 감시와 이벤트 처리)**는 event handler 패턴(필터, requeue, owner-reference watch)입니다.
- **섹션 10 (컨트롤러 테스트)**는 §D입니다 — 실무의 fake client와 envtest.

GVR/GVK를 리소스 분류로, 세 클라이언트 스타일을 일반성-vs-안전성 트레이드오프로, informer + 큐 + reconciler를 보편적 컨트롤러 패턴으로 보고 나면, 모든 쿠버네티스 인식 Go 프로그램은 동일한 빌딩 블록으로 분해됩니다.

---

## 1. Kubernetes API 구조

### 1.1 API 그룹과 버전

Kubernetes API는 독립적인 버전 관리와 진화를 허용하는 **API 그룹(API Groups)**으로 구성됩니다. 각 리소스는 그룹에 속하고, 버전을 가지며, 표준 REST 작업을 노출합니다.

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

Kubernetes의 모든 리소스는 **GroupVersionResource (GVR)** 트리플로 식별됩니다:

```go
// GroupVersionResource는 리소스 타입을 고유하게 식별합니다
import "k8s.io/apimachinery/pkg/runtime/schema"

// apps/v1 그룹의 Deployments
deploymentsGVR := schema.GroupVersionResource{
    Group:    "apps",
    Version:  "v1",
    Resource: "deployments",
}

// core 그룹(빈 문자열)의 Pods
podsGVR := schema.GroupVersionResource{
    Group:    "",
    Version:  "v1",
    Resource: "pods",
}
```

### 1.2 리소스와 Kind

**리소스(Resource)**는 URL 경로 컴포넌트입니다 (항상 소문자 복수형: `pods`, `deployments`). **Kind**는 직렬화에 사용되는 Go 타입 이름입니다 (항상 CamelCase 단수형: `Pod`, `Deployment`). 매핑은 **스킴(Scheme)**에 의해 추적됩니다.

```go
// GroupVersionKind는 Go 타입을 식별합니다
deploymentsGVK := schema.GroupVersionKind{
    Group:   "apps",
    Version: "v1",
    Kind:    "Deployment",
}
```

### 1.3 API 디스커버리

API 서버는 사용 가능한 리소스를 열거하는 디스커버리 엔드포인트를 노출합니다:

```bash
# 모든 API 그룹 목록
kubectl api-versions

# 그룹, 버전, Kind, 동사와 함께 모든 리소스 목록
kubectl api-resources -o wide

# 특정 리소스 탐색
kubectl explain deployment.spec.strategy --api-version=apps/v1

# 디스커버리 엔드포인트에 대한 Raw API 호출
kubectl get --raw /apis | jq '.groups[].name'

# 특정 그룹/버전의 리소스 가져오기
kubectl get --raw /apis/apps/v1 | jq '.resources[].name'
```

### 1.4 API 요청 해부

모든 API 요청은 예측 가능한 URL 패턴을 따릅니다:

```
네임스페이스 리소스:
  GET /apis/<group>/<version>/namespaces/<namespace>/<resource>/<name>

클러스터 범위 리소스:
  GET /apis/<group>/<version>/<resource>/<name>

Core 그룹 (그룹 접두사 없음):
  GET /api/v1/namespaces/<namespace>/<resource>/<name>

예시:
  GET /apis/apps/v1/namespaces/default/deployments/nginx
  GET /api/v1/namespaces/kube-system/pods
  GET /apis/rbac.authorization.k8s.io/v1/clusterroles
```

응답에는 낙관적 동시성(Optimistic Concurrency)을 가능하게 하는 메타데이터가 포함됩니다:

```yaml
# 모든 Kubernetes 객체에 포함되는 메타데이터
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
  namespace: default
  uid: "a1b2c3d4-..."
  resourceVersion: "12345"       # 낙관적 동시성 토큰
  generation: 3                   # Spec 변경 카운터
  creationTimestamp: "2025-01-15T10:00:00Z"
```

`resourceVersion` 필드는 감시(Watch) 프로토콜에 중요합니다 -- API 서버에 변경 사항 스트리밍을 어디서부터 시작할지 알려줍니다.

---

## 2. client-go 라이브러리

### 2.1 개요

`client-go`는 Kubernetes의 공식 Go 클라이언트 라이브러리입니다. 대부분의 Go 기반 Kubernetes 도구가 의존하는 타입화된 클라이언트, 인포머, 캐싱, 유틸리티를 제공합니다.

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

### 2.2 프로젝트 설정

```bash
# Go 모듈 초기화
mkdir k8s-controller && cd k8s-controller
go mod init github.com/example/k8s-controller

# client-go 추가 (버전은 클러스터와 일치해야 함)
go get k8s.io/client-go@v0.29.0
go get k8s.io/apimachinery@v0.29.0
go get k8s.io/api@v0.29.0
```

### 2.3 클러스터에 연결

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

    // default 네임스페이스의 파드 목록
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

// getConfig는 인클러스터(in-cluster) 구성을 반환하거나 kubeconfig으로 폴백
func getConfig() (*rest.Config, error) {
    // 인클러스터 구성을 먼저 시도 (파드 내부에서 실행 시)
    config, err := rest.InClusterConfig()
    if err == nil {
        return config, nil
    }

    // kubeconfig로 폴백
    kubeconfig := filepath.Join(os.Getenv("HOME"), ".kube", "config")
    if envKC := os.Getenv("KUBECONFIG"); envKC != "" {
        kubeconfig = envKC
    }
    return clientcmd.BuildConfigFromFlags("", kubeconfig)
}
```

### 2.4 인증 방법

```go
// 방법 1: Bearer 토큰 (ServiceAccount)
config := &rest.Config{
    Host:        "https://api-server:6443",
    BearerToken: "eyJhbGciOiJSU...",
    TLSClientConfig: rest.TLSClientConfig{
        CAFile: "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
    },
}

// 방법 2: 클라이언트 인증서
config := &rest.Config{
    Host: "https://api-server:6443",
    TLSClientConfig: rest.TLSClientConfig{
        CertFile: "/path/to/client.crt",
        KeyFile:  "/path/to/client.key",
        CAFile:   "/path/to/ca.crt",
    },
}

// 방법 3: 레이트 리미팅 구성
config.QPS = 50       // 초당 쿼리 수 (기본값: 5)
config.Burst = 100     // 버스트 용량 (기본값: 10)
```

---

## 3. REST 클라이언트와 Clientsets

### 3.1 타입화된 Clientsets

**Clientset**은 모든 내장 Kubernetes 리소스에 대한 강타입 메서드를 제공합니다:

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

    // Deployment 생성 (CREATE)
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

    // Deployment 가져오기 (GET)
    fetched, err := clientset.AppsV1().Deployments("default").Get(
        ctx, "web-server", metav1.GetOptions{},
    )
    if err != nil {
        return fmt.Errorf("get deployment: %w", err)
    }

    // 업데이트 (UPDATE) - 5개 레플리카로 스케일
    fetched.Spec.Replicas = ptr.To(int32(5))
    updated, err := clientset.AppsV1().Deployments("default").Update(
        ctx, fetched, metav1.UpdateOptions{},
    )
    if err != nil {
        return fmt.Errorf("update deployment: %w", err)
    }
    fmt.Printf("Updated replicas to %d (rv=%s)\n",
        *updated.Spec.Replicas, updated.ResourceVersion)

    // 레이블 셀렉터로 목록 조회 (LIST)
    deploys, err := clientset.AppsV1().Deployments("default").List(
        ctx, metav1.ListOptions{
            LabelSelector: "app=web-server",
        },
    )
    if err != nil {
        return fmt.Errorf("list deployments: %w", err)
    }
    fmt.Printf("Found %d deployments\n", len(deploys.Items))

    // 삭제 (DELETE)
    err = clientset.AppsV1().Deployments("default").Delete(
        ctx, "web-server", metav1.DeleteOptions{},
    )
    return err
}
```

### 3.2 Status 서브리소스

많은 리소스에는 spec과 별도로 업데이트되는 `/status` 서브리소스가 있습니다:

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

    // status 필드 수정
    deploy.Status.Conditions = append(deploy.Status.Conditions,
        appsv1.DeploymentCondition{
            Type:               appsv1.DeploymentProgressing,
            Status:             corev1.ConditionTrue,
            LastTransitionTime: metav1.Now(),
            Reason:             "NewReplicaSetAvailable",
            Message:            "Deployment has minimum availability",
        },
    )

    // UpdateStatus 사용 - status 서브리소스만 기록
    _, err = clientset.AppsV1().Deployments(namespace).UpdateStatus(
        ctx, deploy, metav1.UpdateOptions{},
    )
    return err
}
```

### 3.3 서버 사이드 적용(Server-Side Apply)

서버 사이드 적용(SSA)은 여러 관리자가 동일한 객체의 다른 필드를 안전하게 소유할 수 있게 합니다:

```go
import (
    "encoding/json"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/types"
)

func serverSideApply(clientset kubernetes.Interface) error {
    ctx := context.TODO()

    // 적용할 부분 객체 정의
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

    // 고유한 필드 관리자 이름으로 적용
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

## 4. 다이나믹 클라이언트와 비구조화 객체

### 4.1 다이나믹 클라이언트를 사용해야 하는 경우

다이나믹 클라이언트(Dynamic Client)는 생성된 Go 타입 없이도 커스텀 리소스(Custom Resource)를 포함한 모든 리소스 타입과 작동합니다. 본질적으로 `map[string]interface{}` 래퍼인 `unstructured.Unstructured` 객체를 사용합니다.

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

    // 커스텀 리소스의 GVR 정의
    certificateGVR := schema.GroupVersionResource{
        Group:    "cert-manager.io",
        Version:  "v1",
        Resource: "certificates",
    }

    // 비구조화 커스텀 리소스 생성
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

    // 리소스 생성
    created, err := dynClient.Resource(certificateGVR).Namespace("default").Create(
        ctx, cert, metav1.CreateOptions{},
    )
    if err != nil {
        return err
    }

    // 중첩 필드 안전하게 읽기
    secretName, found, err := unstructured.NestedString(
        created.Object, "spec", "secretName",
    )
    if err != nil || !found {
        return fmt.Errorf("secretName not found")
    }
    fmt.Printf("Certificate created, secretName: %s\n", secretName)

    // 모든 인증서 목록
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

### 4.2 비구조화 헬퍼(Unstructured Helpers)

```go
import "k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"

// 중첩 필드 읽기
val, found, err := unstructured.NestedString(obj.Object, "spec", "field")
num, found, err := unstructured.NestedInt64(obj.Object, "spec", "replicas")
slice, found, err := unstructured.NestedSlice(obj.Object, "spec", "containers")
nestedMap, found, err := unstructured.NestedMap(obj.Object, "metadata", "labels")

// 중첩 필드 설정
err := unstructured.SetNestedField(obj.Object, "value", "spec", "field")
err := unstructured.SetNestedSlice(obj.Object, items, "spec", "containers")
```

---

## 5. 인포머와 캐싱

### 5.1 인포머가 필요한 이유

API 서버를 폴링하면 확장할 수 없습니다. 인포머(Informer)는 API 서버와의 감시(Watch) 연결을 유지하고 로컬 캐시를 최신 상태로 유지합니다.

```
인포머 아키텍처:
┌────────────────────────────────────────────────────┐
│                  SharedInformer                      │
│                                                      │
│  ┌──────────┐    ┌──────────┐    ┌───────────────┐  │
│  │ Reflector │───▶│  Store   │───▶│  Event Handler │  │
│  │           │    │ (cache)  │    │  (your code)   │  │
│  │ List+Watch│    │          │    │                │  │
│  │ from API  │    │ Thread-  │    │ OnAdd()        │  │
│  │ server    │    │ safe     │    │ OnUpdate()     │  │
│  │           │    │          │    │ OnDelete()     │  │
│  └──────────┘    └──────────┘    └───────────────┘  │
└────────────────────────────────────────────────────┘
```

### 5.2 SharedInformerFactory 사용

```go
import (
    "context"
    "fmt"
    "time"

    "k8s.io/client-go/informers"
    "k8s.io/client-go/kubernetes"
    "k8s.io/client-go/tools/cache"
)

func informerExample(clientset kubernetes.Interface) {
    // 모든 인포머가 공유하는 팩토리 생성 (30초 재동기화)
    factory := informers.NewSharedInformerFactory(clientset, 30*time.Second)

    // 파드 인포머 가져오기
    podInformer := factory.Core().V1().Pods().Informer()

    // 이벤트 핸들러 등록
    podInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
        AddFunc: func(obj interface{}) {
            pod := obj.(*corev1.Pod)
            fmt.Printf("Pod added: %s/%s\n", pod.Namespace, pod.Name)
        },
        UpdateFunc: func(oldObj, newObj interface{}) {
            newPod := newObj.(*corev1.Pod)
            fmt.Printf("Pod updated: %s/%s\n", newPod.Namespace, newPod.Name)
        },
        DeleteFunc: func(obj interface{}) {
            pod := obj.(*corev1.Pod)
            fmt.Printf("Pod deleted: %s/%s\n", pod.Namespace, pod.Name)
        },
    })

    // 모든 인포머 시작
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    factory.Start(ctx.Done())

    // 캐시 동기화 대기
    factory.WaitForCacheSync(ctx.Done())

    // 캐시에서 읽기 (API 서버에 접근하지 않음)
    lister := factory.Core().V1().Pods().Lister()
    pods, _ := lister.Pods("default").List(labels.Everything())
    fmt.Printf("Cached pods in default: %d\n", len(pods))
}
```

### 5.3 SharedInformerFactory

팩토리는 여러 컴포넌트가 동일한 리소스를 관찰해야 하더라도 리소스 타입당 하나의 감시(watch)만 생성되도록 보장합니다:

```go
// 네임스페이스 범위 팩토리 ("production"의 리소스만 감시)
factory := informers.NewSharedInformerFactoryWithOptions(
    clientset,
    30*time.Second,
    informers.WithNamespace("production"),
)

// 동일한 팩토리에서 여러 인포머 사용
deployInformer := factory.Apps().V1().Deployments().Informer()
svcInformer := factory.Core().V1().Services().Informer()
nodeInformer := factory.Core().V1().Nodes().Informer()

// 모두 동일한 stop 채널을 공유
stopCh := make(chan struct{})
factory.Start(stopCh)
factory.WaitForCacheSync(stopCh)
```

### 5.4 인덱서(Indexers)

인포머 캐시는 빠른 조회를 위한 커스텀 인덱스를 지원합니다:

```go
const byNodeIndex = "byNode"

podInformer := factory.Core().V1().Pods().Informer()

// 노드 이름으로 파드를 인덱싱하는 커스텀 인덱스 추가
podInformer.AddIndexers(cache.Indexers{
    byNodeIndex: func(obj interface{}) ([]string, error) {
        pod := obj.(*corev1.Pod)
        if pod.Spec.NodeName == "" {
            return nil, nil
        }
        return []string{pod.Spec.NodeName}, nil
    },
})

// 캐시 동기화 후, 노드로 파드 조회
indexer := podInformer.GetIndexer()
items, err := indexer.ByIndex(byNodeIndex, "worker-node-1")
if err == nil {
    fmt.Printf("worker-node-1의 파드: %d\n", len(items))
}
```

---

## 6. 워크 큐

### 6.1 워크 큐가 필요한 이유

이벤트 핸들러에서 직접 무거운 작업을 하면 인포머의 이벤트 루프가 차단됩니다. 워크 큐(Work Queue)는 이벤트 핸들러를 작업 처리에서 분리합니다.

```go
import (
    "k8s.io/client-go/util/workqueue"
)

// 레이트 리미팅 큐 생성
queue := workqueue.NewRateLimitingQueue(
    workqueue.DefaultControllerRateLimiter(),
)

// 이벤트 핸들러에서 큐에 항목 추가
podInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
    AddFunc: func(obj interface{}) {
        key, err := cache.MetaNamespaceKeyFunc(obj)
        if err == nil {
            queue.Add(key)  // "namespace/name" 형식의 키
        }
    },
    UpdateFunc: func(oldObj, newObj interface{}) {
        key, err := cache.MetaNamespaceKeyFunc(newObj)
        if err == nil {
            queue.Add(key)
        }
    },
    DeleteFunc: func(obj interface{}) {
        key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
        if err == nil {
            queue.Add(key)
        }
    },
})
```

### 6.2 레이트 리미팅 워크 큐

```go
import (
    "time"

    "k8s.io/client-go/util/workqueue"
)

// 지수 백오프가 있는 레이트 리미팅 큐 생성
queue := workqueue.NewRateLimitingQueueWithConfig(
    workqueue.NewItemExponentialFailureRateLimiter(
        200*time.Millisecond, // 기본 지연
        5*time.Minute,        // 최대 지연
    ),
    workqueue.RateLimitingQueueConfig{
        Name: "my-controller",
    },
)
defer queue.ShutDown()

// 항목 큐에 추가 (일반적으로 namespace/name 키)
key := "default/my-pod"
queue.Add(key)

// 지연 후 큐에 추가
queue.AddAfter("default/retry-me", 30*time.Second)

// 워커 루프
for {
    item, shutdown := queue.Get()
    if shutdown {
        break
    }

    // 항목 처리
    key := item.(string)
    err := processItem(key)

    if err != nil {
        // 실패 시 레이트 리미팅으로 재큐잉
        if queue.NumRequeues(item) < 5 {
            queue.AddRateLimited(item)
        } else {
            queue.Forget(item)
        }
    } else {
        queue.Forget(item)
    }
    queue.Done(item)
}
```

### 6.3 워커(Worker) 처리 루프

```go
func processNextItem(queue workqueue.RateLimitingInterface) bool {
    // 큐에서 다음 항목 가져오기 (비어 있으면 차단)
    key, quit := queue.Get()
    if quit {
        return false
    }
    defer queue.Done(key)

    // 조정 로직 수행
    err := reconcile(key.(string))
    if err != nil {
        // 실패 시 큐에 다시 추가 (지수 백오프)
        queue.AddRateLimited(key)
        return true
    }

    // 성공: 레이트 리미터 리셋
    queue.Forget(key)
    return true
}
```

---

## 7. 커스텀 컨트롤러 구축

### 7.1 전체 컨트롤러 예시

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
    "k8s.io/client-go/informers"
    "k8s.io/client-go/kubernetes"
    "k8s.io/client-go/tools/cache"
    "k8s.io/client-go/tools/clientcmd"
    "k8s.io/client-go/util/workqueue"
    "k8s.io/klog/v2"
)

type ConfigMapController struct {
    clientset kubernetes.Interface
    informer  cache.SharedIndexInformer
    queue     workqueue.RateLimitingInterface
}

func NewConfigMapController(
    clientset kubernetes.Interface,
    factory informers.SharedInformerFactory,
) *ConfigMapController {
    informer := factory.Core().V1().ConfigMaps().Informer()
    queue := workqueue.NewRateLimitingQueue(
        workqueue.DefaultControllerRateLimiter(),
    )

    ctrl := &ConfigMapController{
        clientset: clientset,
        informer:  informer,
        queue:     queue,
    }

    informer.AddEventHandler(cache.ResourceEventHandlerFuncs{
        AddFunc: func(obj interface{}) {
            key, _ := cache.MetaNamespaceKeyFunc(obj)
            queue.Add(key)
        },
        UpdateFunc: func(oldObj, newObj interface{}) {
            key, _ := cache.MetaNamespaceKeyFunc(newObj)
            queue.Add(key)
        },
    })

    return ctrl
}

func (c *ConfigMapController) Run(ctx context.Context, workers int) error {
    defer c.queue.ShutDown()

    // 캐시 동기화 대기
    if !cache.WaitForCacheSync(ctx.Done(), c.informer.HasSynced) {
        return fmt.Errorf("cache sync failed")
    }
    klog.Info("Cache synced, starting workers")

    // 워커 시작
    for i := 0; i < workers; i++ {
        go func() {
            for c.processNextItem() {
            }
        }()
    }

    <-ctx.Done()
    return nil
}

func (c *ConfigMapController) processNextItem() bool {
    key, quit := c.queue.Get()
    if quit {
        return false
    }
    defer c.queue.Done(key)

    err := c.reconcile(key.(string))
    if err != nil {
        c.queue.AddRateLimited(key)
        klog.Errorf("Error reconciling %s: %v", key, err)
        return true
    }

    c.queue.Forget(key)
    return true
}

func (c *ConfigMapController) reconcile(key string) error {
    namespace, name, err := cache.SplitMetaNamespaceKey(key)
    if err != nil {
        return err
    }

    cm, err := c.clientset.CoreV1().ConfigMaps(namespace).Get(
        context.TODO(), name, metav1.GetOptions{},
    )
    if errors.IsNotFound(err) {
        klog.Infof("ConfigMap %s deleted", key)
        return nil
    }
    if err != nil {
        return err
    }

    // 조정 로직: "validated" 어노테이션이 없으면 추가
    if cm.Annotations == nil {
        cm.Annotations = make(map[string]string)
    }
    if _, ok := cm.Annotations["validated"]; !ok {
        cm.Annotations["validated"] = "true"
        _, err = c.clientset.CoreV1().ConfigMaps(namespace).Update(
            context.TODO(), cm, metav1.UpdateOptions{},
        )
        if err != nil {
            return err
        }
        klog.Infof("Validated ConfigMap %s", key)
    }

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

### 7.2 소유자 참조(Owner References)

컨트롤러는 가비지 컬렉션을 가능하게 하기 위해 소유자 참조를 설정해야 합니다:

```go
import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// 소유자가 삭제되면 자식 객체도 삭제됨
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

## 8. Controller-Runtime 라이브러리

### 8.1 개요

`controller-runtime` (Kubebuilder와 Operator SDK에서 사용)은 client-go 위에 상위 수준 추상화를 제공합니다:

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

### 8.2 간단한 리컨실러(Reconciler)

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

// DeploymentReconciler는 모든 Deployment에 대응하는 Service가 있는지 확인
type DeploymentReconciler struct {
    client.Client
}

func (r *DeploymentReconciler) Reconcile(
    ctx context.Context,
    req ctrl.Request,
) (ctrl.Result, error) {
    logger := log.FromContext(ctx)

    // Deployment 가져오기
    var deploy appsv1.Deployment
    if err := r.Get(ctx, req.NamespacedName, &deploy); err != nil {
        if errors.IsNotFound(err) {
            return ctrl.Result{}, nil
        }
        return ctrl.Result{}, err
    }

    // 우리의 어노테이션이 없으면 건너뛰기
    if deploy.Annotations["auto-service"] != "true" {
        return ctrl.Result{}, nil
    }

    // Service가 이미 존재하는지 확인
    var svc corev1.Service
    svcName := client.ObjectKey{
        Namespace: deploy.Namespace,
        Name:      deploy.Name + "-auto",
    }

    err := r.Get(ctx, svcName, &svc)
    if errors.IsNotFound(err) {
        // Service 생성
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

        // 가비지 컬렉션을 위한 소유자 참조 설정
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

// SetupWithManager는 이 리컨실러를 매니저에 등록
func (r *DeploymentReconciler) SetupWithManager(mgr ctrl.Manager) error {
    return ctrl.NewControllerManagedBy(mgr).
        For(&appsv1.Deployment{}).        // Deployment 감시
        Owns(&corev1.Service{}).           // 소유한 Service 감시
        Complete(r)
}
```

### 8.3 매니저 설정

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

    // 헬스 체크 등록
    mgr.AddHealthzCheck("healthz", healthz.Ping)
    mgr.AddReadyzCheck("readyz", healthz.Ping)

    // 리컨실러 설정
    reconciler := &controller.DeploymentReconciler{
        Client: mgr.GetClient(),
    }
    if err := reconciler.SetupWithManager(mgr); err != nil {
        os.Exit(1)
    }

    // 매니저 시작 (차단)
    if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
        os.Exit(1)
    }
}
```

### 8.4 프레디케이트(Predicate)와 필터링

```go
import (
    "sigs.k8s.io/controller-runtime/pkg/event"
    "sigs.k8s.io/controller-runtime/pkg/predicate"
)

// 제너레이션 변경(spec 변경, status가 아닌)에만 조정
generationChangedPredicate := predicate.GenerationChangedPredicate{}

// 커스텀 프레디케이트: 레이블이 있는 리소스에만 조정
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

## 9. 리소스 감시와 이벤트 처리

### 9.1 크로스 리소스 감시

컨트롤러는 종종 관련 리소스를 감시해야 합니다. 예를 들어, Deployment와 그것이 참조하는 ConfigMap을 모두 감시:

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
        // ConfigMap이 변경되면 해당 ConfigMap을 참조하는
        // 모든 Deployment를 찾아서 조정
        Watches(
            &corev1.ConfigMap{},
            handler.EnqueueRequestsFromMapFunc(
                func(ctx context.Context, obj client.Object) []reconcile.Request {
                    // 이 ConfigMap을 참조하는 deployment 찾기
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

### 9.2 파이널라이저(Finalizer)

파이널라이저를 사용하면 리소스가 삭제되기 전에 컨트롤러가 정리 작업을 수행할 수 있습니다:

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

    // 객체가 삭제 중인지 확인
    if !obj.DeletionTimestamp.IsZero() {
        if controllerutil.ContainsFinalizer(&obj, finalizerName) {
            // 정리 로직 수행
            if err := r.cleanupExternalResources(ctx, &obj); err != nil {
                return ctrl.Result{}, err
            }

            // 삭제를 허용하기 위해 파이널라이저 제거
            controllerutil.RemoveFinalizer(&obj, finalizerName)
            if err := r.Update(ctx, &obj); err != nil {
                return ctrl.Result{}, err
            }
        }
        return ctrl.Result{}, nil
    }

    // 파이널라이저가 없으면 추가
    if !controllerutil.ContainsFinalizer(&obj, finalizerName) {
        controllerutil.AddFinalizer(&obj, finalizerName)
        if err := r.Update(ctx, &obj); err != nil {
            return ctrl.Result{}, err
        }
    }

    // 일반 조정 로직...
    return ctrl.Result{}, nil
}
```

### 9.3 재큐잉(Requeueing) 전략

```go
// 즉시 재큐 (가능한 빨리 다시 처리)
return ctrl.Result{Requeue: true}, nil

// 지연 후 재큐 (나중에 다시 확인)
return ctrl.Result{RequeueAfter: 30 * time.Second}, nil

// 재큐 없음 (처리 완료)
return ctrl.Result{}, nil

// 에러는 백오프와 함께 자동 재큐를 트리거
return ctrl.Result{}, fmt.Errorf("external API unavailable")
```

---

## 10. 컨트롤러 테스트

### 10.1 가짜 클라이언트(Fake Client)를 사용한 유닛 테스트

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
    // 설정: auto-service 어노테이션이 있는 Deployment
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

    // 실행
    result, err := reconciler.Reconcile(context.TODO(), ctrl.Request{
        NamespacedName: types.NamespacedName{
            Name:      "web",
            Namespace: "default",
        },
    })

    // 검증
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }
    if result.Requeue {
        t.Error("expected no requeue")
    }

    // Service가 생성되었는지 확인
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

### 10.2 envtest을 사용한 통합 테스트

`envtest`은 통합 테스트를 위해 로컬에서 실제 API 서버와 etcd를 실행합니다:

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

    // 컨트롤러 매니저 시작
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

        // 컨트롤러가 조정할 때까지 대기
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

### 10.3 컨트롤러 배포

컨트롤러를 컨테이너 이미지로 패키징하고 클러스터에 배포합니다:

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
  replicas: 2       # 리더 일렉션을 통한 HA
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

## 연습문제

### 연습문제 1: 노드별 파드 목록

client-go를 사용하여 클러스터의 모든 파드를 실행 중인 노드별로 그룹화하여 나열하는 Go 프로그램을 작성하세요. 각 파드의 이름, 네임스페이스, 페이즈(Phase)를 표시하세요.

<details><summary>정답 보기</summary>

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

    // 노드별 그룹화
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

### 연습문제 2: 커스텀 리소스용 다이나믹 클라이언트

다이나믹 클라이언트를 사용하여 클러스터의 모든 CustomResourceDefinition을 나열하고 그룹, 버전, Kind, 스코프를 출력하는 Go 프로그램을 작성하세요.

<details><summary>정답 보기</summary>

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

### 연습문제 3: 커스텀 인덱서를 가진 인포머

SharedInformerFactory와 커스텀 인덱서를 사용하여 Service를 타입(ClusterIP, NodePort, LoadBalancer)별로 빠르게 조회하는 프로그램을 만드세요. 캐시 동기화 후 인덱스를 쿼리하고 서비스 타입별 개수를 출력하세요.

<details><summary>정답 보기</summary>

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

    // 커스텀 인덱서 추가
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

    // 변경 사항 감시를 위해 계속 실행
    fmt.Println("\nWatching for changes (Ctrl+C to stop)...")
    sig := make(chan os.Signal, 1)
    signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
    <-sig
}
```

</details>

### 연습문제 4: 재큐를 사용하는 컨트롤러

각 네임스페이스에 `default-quota`라는 `ResourceQuota`가 있는지 확인하는 controller-runtime 리컨실러를 작성하세요. 없으면 10개 파드와 4Gi 메모리 제한으로 생성하세요. 생성에 실패하면 30초 후에 재큐하세요.

<details><summary>정답 보기</summary>

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

    // 시스템 네임스페이스 건너뛰기
    systemNS := map[string]bool{
        "kube-system": true, "kube-public": true,
        "kube-node-lease": true, "default": true,
    }
    if systemNS[req.Name] {
        return ctrl.Result{}, nil
    }

    // 네임스페이스가 아직 존재하는지 확인
    var ns corev1.Namespace
    if err := r.Get(ctx, req.NamespacedName, &ns); err != nil {
        return ctrl.Result{}, client.IgnoreNotFound(err)
    }

    // ResourceQuota 존재 확인
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

### 연습문제 5: 가짜 클라이언트를 사용한 테스트

연습문제 4의 QuotaReconciler가 ResourceQuota가 없을 때 생성하는지, 그리고 이미 있을 때 중복 생성하지 않는지 검증하는 유닛 테스트를 작성하세요.

<details><summary>정답 보기</summary>

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

    // quota가 수정되지 않았는지 확인
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

    // quota가 생성되지 않았는지 확인
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

**이전**: [15. 멀티 클러스터](./15_Multi_Cluster.md) | **다음**: [17. 프로덕션 운영](./17_Production_Operations.md)

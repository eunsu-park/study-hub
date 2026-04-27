# 11. 오퍼레이터(Operators)

**이전**: [커스텀 리소스 정의](./10_Custom_Resource_Definitions.md) | **다음**: [어드미션 컨트롤러](./12_Admission_Controllers.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 오퍼레이터 패턴(Operator pattern)을 설명하고 왜 존재하는지 이해할 수 있다
2. Kubebuilder와 operator-sdk를 사용하여 오퍼레이터를 스캐폴딩, 빌드, 배포할 수 있다
3. 적절한 오류 처리 및 재큐(requeue) 로직을 갖춘 조정 루프(reconciliation loop)를 구현할 수 있다
4. 파이널라이저(finalizer), 소유자 참조(owner reference), 리더 선출(leader election)을 올바르게 사용할 수 있다
5. Operator Lifecycle Manager(OLM)를 통해 오퍼레이터를 배포할 수 있다

---

Kubernetes에서 무상태(stateless) 워크로드를 실행하는 것은 간단합니다 -- Deployment, Service, Ingress가 대부분의 작업을 처리합니다. 하지만 상태 유지(stateful), 도메인별 애플리케이션(데이터베이스, 메시지 큐, ML 파이프라인)은 설치, 구성, 스케일링, 업그레이드, 복구에 사람의 전문 지식이 필요합니다. 오퍼레이터 패턴(Operator pattern)은 이러한 인간의 지식을 클러스터 내부에서 실행되는 소프트웨어로 인코딩하여 시스템을 지속적으로 원하는 상태(desired state)로 이끕니다. 이 레슨에서는 Kubernetes 오퍼레이터의 구축, 배포, 유지보수의 전체 라이프사이클을 다룹니다.

## 목차

- [이론과 원리](#이론과-원리)
- [1. 오퍼레이터 패턴](#1-the-operator-pattern)
- [2. Operator Framework와 operator-sdk](#2-operator-framework-and-operator-sdk)
- [3. Kubebuilder](#3-kubebuilder)
- [4. Controller-Runtime 라이브러리](#4-controller-runtime-library)
- [5. 조정 루프 구현](#5-implementing-a-reconciliation-loop)
- [6. 리더 선출](#6-leader-election)
- [7. 파이널라이저](#7-finalizers)
- [8. 소유자 참조](#8-owner-references)
- [9. Operator Lifecycle Manager (OLM)](#9-operator-lifecycle-manager-olm)
- [10. 모범 사례와 안티패턴](#10-best-practices-and-anti-patterns)
- [연습문제](#exercises)

---

## 1. 오퍼레이터 패턴

### 이론: Operator = 커스텀 리소스 + 도메인 인식 컨트롤러

Operator는 이미 알고 있는 두 가지의 합성입니다:

1. **CRD**(10강)가 사용자 의도의 *형태*를 정의합니다 — `PostgresCluster`는 `spec.replicas`, `spec.version`, `spec.storage` 등을 가집니다.
2. **컨트롤러**가 그 CRD를 watch하고 매치되도록 세상을 조정합니다 — 실제로 StatefulSet, Service, Secret, PVC를 생성, 스트리밍 복제 구성, 헬스 모니터링, 롤링 업그레이드 수행.

CRD 자체는 etcd에 저장된 타입화된 형태일 뿐입니다. 컨트롤러 없이 `PostgresCluster` 객체 적용은 아무것도 하지 않습니다. 컨트롤러가 있으면, 같은 적용이 완전히 동작하는 클러스터를 만듭니다 — 컨트롤러가 모든 운영 지식을 내재하고 있기 때문입니다.

패턴의 우아함은 내장 쿠버네티스(Deployment, ReplicaSet, ...)와 동일한 *모델*을 따르되 쿠버네티스 자체가 모르는 도메인 객체에 대해 그렇게 한다는 것입니다. 이 패턴을 채택하면 플랫폼 어휘가 확장됩니다 — "StatefulSet 만들고, 그다음 복제 구성, 그다음 ..." 대신 `kubectl apply -f my-database.yaml`이라 말합니다.

Operator 패턴은 마법이 *아닙니다* — 10강의 CRD에 **controller-runtime**을 사용하는 Go(또는 쿠버네티스 API를 직접 사용하는 어떤 언어든, 그러나 Go가 최고의 생태계를 가짐)로 작성된 컨트롤러를 더한 것일 뿐입니다. operator를 강력하게 만드는 것은 새 프레임워크 기능이 아니라 *전문성의 인코딩*입니다.

### 1.1 오퍼레이터란?

오퍼레이터(Operator)는 하나 이상의 커스텀 리소스(CR)를 감시하고 애플리케이션별 자동화를 수행하는 Kubernetes 컨트롤러입니다. 내장 컨트롤러(Deployment, ReplicaSet)가 사용하는 동일한 Kubernetes 제어 루프(control loop) 패턴을 확장하여 복잡한 상태 유지 워크로드를 관리합니다.

```
                  ┌──────────────────────────────────────────────┐
                  │            Kubernetes API Server              │
                  └────────────┬────────────────┬────────────────┘
                               │                │
                       Watch CRs/Events   Update Status
                               │                │
                  ┌────────────▼────────────────▼────────────────┐
                  │              Operator Controller              │
                  │  ┌─────────────────────────────────────────┐  │
                  │  │         Reconciliation Loop              │  │
                  │  │  1. Observe current state                │  │
                  │  │  2. Compare with desired state (spec)    │  │
                  │  │  3. Act to converge                      │  │
                  │  │  4. Report status                        │  │
                  │  └─────────────────────────────────────────┘  │
                  └──────────────────────────────────────────────┘
                               │
                  ┌────────────▼────────────────┐
                  │   Managed Resources          │
                  │   (Pods, Services, PVCs...)   │
                  └─────────────────────────────┘
```

### 1.2 동기: 왜 Helm만으로는 안 되는가?

Helm은 리소스를 배포하고 떠납니다. 오퍼레이터는 계속 실행되면서 애플리케이션을 지속적으로 관리합니다.

| 기능 | Helm Chart | 오퍼레이터(Operator) |
|---|---|---|
| 초기 배포 | 예 | 예 |
| Day-2 운영 (백업, 장애 조치) | 아니오 | 예 |
| Pod 재시작 이상의 자가 치유 | 아니오 | 예 |
| 업그레이드 시 스키마 마이그레이션 | 수동 | 자동화 |
| 애플리케이션 인식 스케일링 | 아니오 | 예 |
| 라이프사이클 관리 | 제한적 | 전체 |

### 1.3 성숙도 모델

Operator SDK는 다섯 가지 기능 수준을 정의합니다:

| 레벨 | 기능 | 예시 |
|---|---|---|
| 1 | 기본 설치 | 자동화된 프로비저닝 |
| 2 | 원활한 업그레이드 | 패치 및 마이너 버전 업그레이드 |
| 3 | 전체 라이프사이클 | 백업, 복원, 장애 복구 |
| 4 | 심층 인사이트 | 메트릭, 알림, 로그 처리 |
| 5 | 자동 파일럿 | 자동 스케일링, 자동 튜닝, 이상 감지 |

---

## 2. Operator Framework와 operator-sdk

### 2.1 아키텍처 개요

Operator Framework는 세 가지 구성 요소로 이루어져 있습니다:

1. **Operator SDK** -- 스캐폴딩 및 빌드 도구
2. **Operator Lifecycle Manager (OLM)** -- 설치 및 업그레이드 관리
3. **OperatorHub** -- 발견 및 공유

### 2.2 설치

```bash
# Install operator-sdk CLI
# macOS
brew install operator-sdk

# Linux (amd64)
export ARCH=$(case $(uname -m) in x86_64) echo -n amd64 ;; aarch64) echo -n arm64 ;; esac)
export OS=$(uname | awk '{print tolower($0)}')
export OPERATOR_SDK_DL_URL=https://github.com/operator-framework/operator-sdk/releases/download/v1.34.1
curl -LO ${OPERATOR_SDK_DL_URL}/operator-sdk_${OS}_${ARCH}
chmod +x operator-sdk_${OS}_${ARCH}
sudo mv operator-sdk_${OS}_${ARCH} /usr/local/bin/operator-sdk

# Verify
operator-sdk version
```

### 2.3 SDK 프로젝트 유형

operator-sdk는 세 가지 프로젝트 유형을 지원합니다:

| 유형 | 언어 | 사용 사례 |
|---|---|---|
| Go | Go | 최대 제어가 가능한 완전한 기능의 오퍼레이터 |
| Ansible | YAML/Ansible | Ansible에 익숙한 팀을 위한 오퍼레이터 |
| Helm | Go template | 기존 Helm 차트를 오퍼레이터로 래핑 |

### 2.4 Go 오퍼레이터 스캐폴딩

```bash
# Create project directory
mkdir memcached-operator && cd memcached-operator

# Initialize the project
operator-sdk init \
  --domain example.com \
  --repo github.com/example/memcached-operator

# Create an API (CRD + controller)
operator-sdk create api \
  --group cache \
  --version v1alpha1 \
  --kind Memcached \
  --resource --controller

# Project structure after scaffolding
# .
# ├── Dockerfile
# ├── Makefile
# ├── PROJECT
# ├── api/
# │   └── v1alpha1/
# │       ├── memcached_types.go    # CRD spec/status types
# │       └── zz_generated.deepcopy.go
# ├── cmd/
# │   └── main.go                   # Manager entrypoint
# ├── config/
# │   ├── crd/                      # Generated CRD manifests
# │   ├── manager/                  # Controller manager deployment
# │   ├── rbac/                     # RBAC for the operator
# │   └── samples/                  # Example CR
# └── internal/
#     └── controller/
#         └── memcached_controller.go  # Reconciliation logic
```

---

## 3. Kubebuilder

### 3.1 Kubebuilder vs operator-sdk

Kubebuilder는 operator-sdk가 기반으로 하는 업스트림 프로젝트입니다. v1.28+ 이후 operator-sdk는 Kubebuilder의 프로젝트 레이아웃을 직접 사용합니다.

| 기능 | Kubebuilder | operator-sdk |
|---|---|---|
| Go 스캐폴딩 | 예 | 예 (Kubebuilder 래핑) |
| Ansible/Helm 지원 | 아니오 | 예 |
| OLM 통합 | 아니오 | 예 |
| 스코어카드 테스팅 | 아니오 | 예 |
| 번들/카탈로그 도구 | 아니오 | 예 |

### 3.2 Kubebuilder 설치

```bash
# macOS
brew install kubebuilder

# Linux
curl -L -o kubebuilder "https://go.kubebuilder.io/dl/latest/$(go env GOOS)/$(go env GOARCH)"
chmod +x kubebuilder
sudo mv kubebuilder /usr/local/bin/

# Verify
kubebuilder version
```

### 3.3 API 타입 정의

```go
// api/v1alpha1/memcached_types.go
package v1alpha1

import (
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// MemcachedSpec defines the desired state of Memcached
type MemcachedSpec struct {
    // Size is the number of Memcached pods
    // +kubebuilder:validation:Minimum=1
    // +kubebuilder:validation:Maximum=10
    Size int32 `json:"size"`

    // ContainerPort is the port for the Memcached container
    // +kubebuilder:validation:Minimum=1024
    // +kubebuilder:validation:Maximum=65535
    // +kubebuilder:default:=11211
    ContainerPort int32 `json:"containerPort,omitempty"`

    // Image is the Memcached container image
    // +kubebuilder:default:="memcached:1.6-alpine"
    Image string `json:"image,omitempty"`
}

// MemcachedStatus defines the observed state of Memcached
type MemcachedStatus struct {
    // Conditions store the status conditions of the Memcached instances
    // +operator-sdk:csv:customresourcedefinitions:type=status
    Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`

    // ReadyReplicas is the number of ready pods
    ReadyReplicas int32 `json:"readyReplicas,omitempty"`

    // Nodes are the names of the Memcached pods
    Nodes []string `json:"nodes,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Size",type="integer",JSONPath=".spec.size"
// +kubebuilder:printcolumn:name="Ready",type="integer",JSONPath=".status.readyReplicas"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// Memcached is the Schema for the memcacheds API
type Memcached struct {
    metav1.TypeMeta   `json:",inline"`
    metav1.ObjectMeta `json:"metadata,omitempty"`

    Spec   MemcachedSpec   `json:"spec,omitempty"`
    Status MemcachedStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// MemcachedList contains a list of Memcached
type MemcachedList struct {
    metav1.TypeMeta `json:",inline"`
    metav1.ListMeta `json:"metadata,omitempty"`
    Items           []Memcached `json:"items"`
}

func init() {
    SchemeBuilder.Register(&Memcached{}, &MemcachedList{})
}
```

### 3.4 Kubebuilder 마커 참조

타입 정의에서 사용되는 일반적인 마커:

```go
// Validation markers
// +kubebuilder:validation:Minimum=0
// +kubebuilder:validation:Maximum=100
// +kubebuilder:validation:Enum=Active;Standby;Failed
// +kubebuilder:validation:Pattern=`^[a-z]+$`
// +kubebuilder:validation:Required

// Resource markers
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:subresource:scale:specpath=.spec.size,statuspath=.status.readyReplicas

// Print column markers
// +kubebuilder:printcolumn:name="Status",type="string",JSONPath=".status.phase"

// RBAC markers (on controller methods)
// +kubebuilder:rbac:groups=cache.example.com,resources=memcacheds,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=cache.example.com,resources=memcacheds/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
```

---

## 4. Controller-Runtime 라이브러리

### 4.1 핵심 구성 요소

Controller-runtime은 Kubebuilder와 operator-sdk 컨트롤러를 구동하는 라이브러리입니다. 주요 구성 요소는 다음과 같습니다:

```
┌─────────────────────────────────────────────────────┐
│                    Manager                           │
│  ┌────────────┐  ┌────────────┐  ┌───────────────┐  │
│  │   Cache     │  │   Client   │  │ Leader Election│  │
│  │ (Informers) │  │ (API calls)│  │               │  │
│  └──────┬─────┘  └──────┬─────┘  └───────────────┘  │
│         │               │                            │
│  ┌──────▼───────────────▼──────┐                     │
│  │       Controller            │                     │
│  │  ┌──────────────────────┐   │                     │
│  │  │    Work Queue        │   │                     │
│  │  └──────────┬───────────┘   │                     │
│  │             │               │                     │
│  │  ┌──────────▼───────────┐   │                     │
│  │  │    Reconciler        │   │                     │
│  │  │  (Your logic here)   │   │                     │
│  │  └──────────────────────┘   │                     │
│  └─────────────────────────────┘                     │
└─────────────────────────────────────────────────────┘
```

### 4.2 Manager 설정

```go
// cmd/main.go
package main

import (
    "flag"
    "os"

    "k8s.io/apimachinery/pkg/runtime"
    utilruntime "k8s.io/apimachinery/pkg/util/runtime"
    clientgoscheme "k8s.io/client-go/kubernetes/scheme"
    ctrl "sigs.k8s.io/controller-runtime"
    "sigs.k8s.io/controller-runtime/pkg/healthz"
    "sigs.k8s.io/controller-runtime/pkg/log/zap"
    metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

    cachev1alpha1 "github.com/example/memcached-operator/api/v1alpha1"
    "github.com/example/memcached-operator/internal/controller"
)

var (
    scheme   = runtime.NewScheme()
    setupLog = ctrl.Log.WithName("setup")
)

func init() {
    utilruntime.Must(clientgoscheme.AddToScheme(scheme))
    utilruntime.Must(cachev1alpha1.AddToScheme(scheme))
}

func main() {
    var metricsAddr string
    var probeAddr string
    var enableLeaderElection bool

    flag.StringVar(&metricsAddr, "metrics-bind-address", ":8080", "The address the metric endpoint binds to.")
    flag.StringVar(&probeAddr, "health-probe-bind-address", ":8081", "The address the probe endpoint binds to.")
    flag.BoolVar(&enableLeaderElection, "leader-elect", false, "Enable leader election.")
    flag.Parse()

    ctrl.SetLogger(zap.New(zap.UseDevMode(true)))

    mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
        Scheme: scheme,
        Metrics: metricsserver.Options{
            BindAddress: metricsAddr,
        },
        HealthProbeBindAddress: probeAddr,
        LeaderElection:         enableLeaderElection,
        LeaderElectionID:       "memcached-operator-lock",
    })
    if err != nil {
        setupLog.Error(err, "unable to start manager")
        os.Exit(1)
    }

    if err = (&controller.MemcachedReconciler{
        Client: mgr.GetClient(),
        Scheme: mgr.GetScheme(),
    }).SetupWithManager(mgr); err != nil {
        setupLog.Error(err, "unable to create controller", "controller", "Memcached")
        os.Exit(1)
    }

    if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
        setupLog.Error(err, "unable to set up health check")
        os.Exit(1)
    }
    if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
        setupLog.Error(err, "unable to set up ready check")
        os.Exit(1)
    }

    setupLog.Info("starting manager")
    if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
        setupLog.Error(err, "problem running manager")
        os.Exit(1)
    }
}
```

### 4.3 Client 인터페이스

Controller-runtime은 캐시에서 읽고 API 서버에 쓰는 통합 클라이언트를 제공합니다:

```go
// Reading (from cache by default)
instance := &cachev1alpha1.Memcached{}
err := r.Get(ctx, req.NamespacedName, instance)

// Listing with label selectors
podList := &corev1.PodList{}
listOpts := []client.ListOption{
    client.InNamespace(req.Namespace),
    client.MatchingLabels{"app": "memcached", "memcached_cr": req.Name},
}
err := r.List(ctx, podList, listOpts...)

// Writing (always goes to API server)
err := r.Create(ctx, deployment)
err := r.Update(ctx, instance)
err := r.Status().Update(ctx, instance)  // status subresource
err := r.Delete(ctx, pod)

// Patch for conflict-free updates
patch := client.MergeFrom(instance.DeepCopy())
instance.Status.ReadyReplicas = readyCount
err := r.Status().Patch(ctx, instance, patch)
```

---

## 5. 조정 루프 구현

### 이론: 컨트롤러의 심장 박동 — Informer + Work Queue + Reconcile

모든 operator(그리고 모든 내장 컨트롤러)는 **controller-runtime**이 제공하는 동일한 아키텍처를 실행합니다:

```
Watch → Informer (캐시) → Event Handler → Work Queue → Reconciler
```

**Informer**는 로컬 캐시를 유지하는 리소스 유형에 대한 장기 watch입니다. 왜 캐시? 대안 — 모든 reconcile이 API 서버에서 읽는 것 — 은 감당할 수 없기 때문입니다. Informer는 한 번의 초기 list를 하고 그다음 델타를 스트리밍합니다(1강 §A) — 읽기는 로컬이고 빠릅니다.

**Event Handler**는 informer로부터 `ADDED`/`MODIFIED`/`DELETED` 이벤트를 보고 무엇을 할지 결정합니다. 보통 — 객체의 namespace/name을 추출하고 그것에 대한 *reconcile request*를 enqueue. 주의 — 핸들러는 작업을 하지 않고 enqueue합니다.

**Work Queue**는 이벤트 생산과 조정 사이의 버퍼입니다. 중복 제거(같은 객체에 대한 100개 이벤트가 도착하면 reconcile은 한 번만), rate limiting 지원(오류에 대한 exponential backoff), 키별 순차 처리 강제(`default/my-cluster`에 대해 한 번에 하나의 reconcile, 경쟁 없음). controller-runtime이 합리적 기본값을 제공합니다 — 오류에 대한 exponential-backoff를 가진 rate-limited 큐.

**Reconciler**는 당신이 작성하는 함수입니다. 시그니처는:

```go
func (r *PostgresReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error)
```

내부에서 다음을 합니다:
1. 원하는 상태 가져오기 — 캐시에서 CR 읽기.
2. 실제 상태 가져오기 — 자식 리소스(StatefulSet, Service)와 그 status 읽기.
3. diff 계산 후 행동 — 필요에 따라 생성/업데이트/삭제.
4. 반환 — 즉시 돌아오려면 `Result{Requeue: true}`, 주기적 검사를 위해 `Result{RequeueAfter: 30s}`, 또는 backoff 재시도를 트리거하려면 `error`.

이 패턴과 함께 무료로 오는 두 속성:

- **기본 멱등(idempotence by default).** Reconciler는 같은 객체에 대해 여러 번 호출됩니다 — 초기 생성, 모든 변경, 재동기화 간격, 재시도 시. 코드는 매번 같은 결과를 만들어야 합니다. 패턴은 "create"가 아니라 "create-or-update" — 보통 `controllerutil.CreateOrUpdate`를 사용.
- **레벨 트리거, 엣지 트리거 아님.** *현재 상태*에 반응합니다 — 이벤트가 아닙니다. reconcile 도중 컨트롤러가 크래시해도, 다음 시작은 같은 상태를 보고 계속합니다 — 복구할 놓친 이벤트 없음.

이 루프가 쿠버네티스 확장을 위한 *바로 그* 프로그래밍 모델입니다. 다른 모든 것(finalizer, owner reference, status 업데이트)은 그 위의 정제입니다.

### 5.1 Reconciler 인터페이스

모든 컨트롤러는 `Reconciler` 인터페이스를 구현해야 합니다:

```go
type Reconciler interface {
    Reconcile(ctx context.Context, req Request) (Result, error)
}
```

반환값은 재큐(requeue) 동작을 제어합니다:

| 반환값 | 동작 |
|---|---|
| `Result{}, nil` | 성공, 조정 중지 |
| `Result{Requeue: true}, nil` | 즉시 재큐 |
| `Result{RequeueAfter: 30 * time.Second}, nil` | 지정 시간 후 재큐 |
| `Result{}, err` | 지수 백오프(exponential backoff)로 재큐 |

### 5.2 전체 Reconciler 구현

```go
// internal/controller/memcached_controller.go
package controller

import (
    "context"
    "fmt"
    "time"

    appsv1 "k8s.io/api/apps/v1"
    corev1 "k8s.io/api/core/v1"
    apierrors "k8s.io/apimachinery/pkg/api/errors"
    "k8s.io/apimachinery/pkg/api/meta"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/runtime"
    "k8s.io/apimachinery/pkg/types"
    "k8s.io/client-go/tools/record"
    ctrl "sigs.k8s.io/controller-runtime"
    "sigs.k8s.io/controller-runtime/pkg/client"
    "sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
    "sigs.k8s.io/controller-runtime/pkg/log"

    cachev1alpha1 "github.com/example/memcached-operator/api/v1alpha1"
)

const memcachedFinalizer = "cache.example.com/finalizer"

// MemcachedReconciler reconciles a Memcached object
type MemcachedReconciler struct {
    client.Client
    Scheme   *runtime.Scheme
    Recorder record.EventRecorder
}

// +kubebuilder:rbac:groups=cache.example.com,resources=memcacheds,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=cache.example.com,resources=memcacheds/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=cache.example.com,resources=memcacheds/finalizers,verbs=update
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=events,verbs=create;patch

func (r *MemcachedReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    log := log.FromContext(ctx)

    // Step 1: Fetch the Memcached instance
    memcached := &cachev1alpha1.Memcached{}
    err := r.Get(ctx, req.NamespacedName, memcached)
    if err != nil {
        if apierrors.IsNotFound(err) {
            log.Info("Memcached resource not found. Ignoring since object must be deleted.")
            return ctrl.Result{}, nil
        }
        log.Error(err, "Failed to get Memcached")
        return ctrl.Result{}, err
    }

    // Step 2: Handle finalizer for cleanup
    if memcached.ObjectMeta.DeletionTimestamp.IsZero() {
        // Object is not being deleted -- add finalizer if not present
        if !controllerutil.ContainsFinalizer(memcached, memcachedFinalizer) {
            controllerutil.AddFinalizer(memcached, memcachedFinalizer)
            if err := r.Update(ctx, memcached); err != nil {
                return ctrl.Result{}, err
            }
        }
    } else {
        // Object is being deleted -- run cleanup logic
        if controllerutil.ContainsFinalizer(memcached, memcachedFinalizer) {
            if err := r.cleanupResources(ctx, memcached); err != nil {
                return ctrl.Result{}, err
            }
            controllerutil.RemoveFinalizer(memcached, memcachedFinalizer)
            if err := r.Update(ctx, memcached); err != nil {
                return ctrl.Result{}, err
            }
        }
        return ctrl.Result{}, nil
    }

    // Step 3: Check if the Deployment already exists, create if not
    found := &appsv1.Deployment{}
    err = r.Get(ctx, types.NamespacedName{Name: memcached.Name, Namespace: memcached.Namespace}, found)
    if err != nil && apierrors.IsNotFound(err) {
        dep, err := r.deploymentForMemcached(memcached)
        if err != nil {
            log.Error(err, "Failed to define new Deployment for Memcached")
            meta.SetStatusCondition(&memcached.Status.Conditions, metav1.Condition{
                Type:    "Available",
                Status:  metav1.ConditionFalse,
                Reason:  "Reconciling",
                Message: fmt.Sprintf("Failed to create Deployment: %s", err),
            })
            if statusErr := r.Status().Update(ctx, memcached); statusErr != nil {
                log.Error(statusErr, "Failed to update Memcached status")
                return ctrl.Result{}, statusErr
            }
            return ctrl.Result{}, err
        }

        log.Info("Creating a new Deployment", "Deployment.Namespace", dep.Namespace, "Deployment.Name", dep.Name)
        if err = r.Create(ctx, dep); err != nil {
            log.Error(err, "Failed to create new Deployment")
            return ctrl.Result{}, err
        }
        r.Recorder.Event(memcached, corev1.EventTypeNormal, "Created", "Deployment created successfully")
        return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
    } else if err != nil {
        log.Error(err, "Failed to get Deployment")
        return ctrl.Result{}, err
    }

    // Step 4: Ensure the Deployment size matches the spec
    size := memcached.Spec.Size
    if *found.Spec.Replicas != size {
        found.Spec.Replicas = &size
        if err = r.Update(ctx, found); err != nil {
            log.Error(err, "Failed to update Deployment", "Deployment.Namespace", found.Namespace, "Deployment.Name", found.Name)
            return ctrl.Result{}, err
        }
        r.Recorder.Eventf(memcached, corev1.EventTypeNormal, "Scaled", "Scaled deployment to %d replicas", size)
        return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
    }

    // Step 5: Update status with ready replicas
    memcached.Status.ReadyReplicas = found.Status.ReadyReplicas
    podList := &corev1.PodList{}
    listOpts := []client.ListOption{
        client.InNamespace(memcached.Namespace),
        client.MatchingLabels(labelsForMemcached(memcached.Name)),
    }
    if err = r.List(ctx, podList, listOpts...); err != nil {
        log.Error(err, "Failed to list pods")
        return ctrl.Result{}, err
    }
    podNames := getPodNames(podList.Items)
    memcached.Status.Nodes = podNames

    meta.SetStatusCondition(&memcached.Status.Conditions, metav1.Condition{
        Type:    "Available",
        Status:  metav1.ConditionTrue,
        Reason:  "Reconciling",
        Message: fmt.Sprintf("Deployment has %d ready replicas", found.Status.ReadyReplicas),
    })

    if err := r.Status().Update(ctx, memcached); err != nil {
        log.Error(err, "Failed to update Memcached status")
        return ctrl.Result{}, err
    }

    return ctrl.Result{RequeueAfter: 1 * time.Minute}, nil
}

// deploymentForMemcached returns a Deployment object for the Memcached CR
func (r *MemcachedReconciler) deploymentForMemcached(m *cachev1alpha1.Memcached) (*appsv1.Deployment, error) {
    labels := labelsForMemcached(m.Name)
    replicas := m.Spec.Size

    dep := &appsv1.Deployment{
        ObjectMeta: metav1.ObjectMeta{
            Name:      m.Name,
            Namespace: m.Namespace,
            Labels:    labels,
        },
        Spec: appsv1.DeploymentSpec{
            Replicas: &replicas,
            Selector: &metav1.LabelSelector{
                MatchLabels: labels,
            },
            Template: corev1.PodTemplateSpec{
                ObjectMeta: metav1.ObjectMeta{
                    Labels: labels,
                },
                Spec: corev1.PodSpec{
                    Containers: []corev1.Container{{
                        Name:    "memcached",
                        Image:   m.Spec.Image,
                        Command: []string{"memcached", "-m=64", "-o", "modern", "-v"},
                        Ports: []corev1.ContainerPort{{
                            ContainerPort: m.Spec.ContainerPort,
                            Name:          "memcached",
                        }},
                    }},
                },
            },
        },
    }

    // Set the owning CR as the owner of the Deployment
    if err := ctrl.SetControllerReference(m, dep, r.Scheme); err != nil {
        return nil, err
    }
    return dep, nil
}

func labelsForMemcached(name string) map[string]string {
    return map[string]string{
        "app.kubernetes.io/name":       "memcached",
        "app.kubernetes.io/instance":   name,
        "app.kubernetes.io/managed-by": "memcached-operator",
    }
}

func getPodNames(pods []corev1.Pod) []string {
    podNames := make([]string, len(pods))
    for i, pod := range pods {
        podNames[i] = pod.Name
    }
    return podNames
}

func (r *MemcachedReconciler) cleanupResources(ctx context.Context, m *cachev1alpha1.Memcached) error {
    log := log.FromContext(ctx)
    log.Info("Running cleanup for Memcached", "name", m.Name)
    // External cleanup logic goes here (e.g., deregister from service mesh)
    return nil
}

// SetupWithManager sets up the controller with the Manager
func (r *MemcachedReconciler) SetupWithManager(mgr ctrl.Manager) error {
    return ctrl.NewControllerManagedBy(mgr).
        For(&cachev1alpha1.Memcached{}).
        Owns(&appsv1.Deployment{}).
        Complete(r)
}
```

### 5.3 조정 패턴

```
                Idempotent Reconciliation Flow
                ================================

    ┌──────────┐    ┌──────────────┐    ┌──────────────┐
    │  Observe  │───▶│   Compare    │───▶│     Act      │
    │  current  │    │  current vs  │    │  create/     │
    │  state    │    │  desired     │    │  update/     │
    └──────────┘    └──────────────┘    │  delete      │
         ▲                               └──────┬───────┘
         │                                      │
         └──────────────────────────────────────┘
                    (requeue)
```

핵심 원칙:

1. **멱등성(Idempotency)** -- 동일한 입력으로 Reconcile을 여러 번 실행해도 동일한 결과를 생성
2. **레벨 트리거(Level-triggered)** -- 이벤트가 아닌 현재 상태에 반응
3. **엣지 무관(Edge-agnostic)** -- 특정 이벤트에 의해 조정이 트리거된다고 가정하지 않음

---

## 6. 리더 선출

### 이론: Leader Election — 하나가 활성, 여럿이 대기

Operator는 HA를 위해 여러 레플리카로 실행되어야 하지만, 한 번에 **하나만** 조정해야 합니다 — 그렇지 않으면 두 레플리카가 같은 StatefulSet을 만들려고 경쟁합니다. controller-runtime의 해결책은 **leader election**입니다 — 레플리카들이 클러스터의 `Lease` 객체를 두고 경쟁하고, Lease를 가진 자가 리더이며 reconcile 루프를 실행합니다. 대기는 Lease를 watch하다가 만료되면 인계받습니다(기본 15초 TTL, 10초 갱신, 2초 재시도).

```go
mgr, _ := manager.New(cfg, manager.Options{
    LeaderElection:   true,
    LeaderElectionID: "my-operator-lock",
    LeaderElectionNamespace: "my-operator-system",
})
```

Lease 객체는 etcd에 살므로, etcd를 안전하게 만드는 동일한 합의(1강 §B)가 leader election을 안전하게 만듭니다 — 네트워크 파티션 하에서 etcd 멤버 쿼럼에 도달할 수 있는 측만 lease를 가질 수 있습니다.

이는 kube-controller-manager가 자신에 대해 사용하는 동일한 패턴입니다. Deployment 컨트롤러는 고가용성입니다 — 세 controller-manager 레플리카, 한 명의 선출된 리더가 reconcile 루프 실행, 두 개의 warm 대기.

### 6.1 왜 리더 선출이 필요한가?

고가용성을 위해 여러 오퍼레이터 레플리카(replica)를 실행할 때, 한 번에 하나의 인스턴스만 능동적으로 조정해야 합니다. 리더 선출(leader election)은 Kubernetes Lease 객체를 사용하여 단일 쓰기(single-writer) 의미를 보장합니다.

### 6.2 작동 방식

```
   Pod-A (Leader)           Pod-B (Standby)          Pod-C (Standby)
   ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
   │ Reconciling  │         │   Watching   │         │   Watching   │
   │ actively     │         │   lease      │         │   lease      │
   └──────┬───────┘         └──────┬───────┘         └──────┬───────┘
          │                        │                        │
          ▼                        ▼                        ▼
   ┌─────────────────────────────────────────────────────────┐
   │              Lease: memcached-operator-lock              │
   │              holder: pod-a                               │
   │              renewTime: 2024-01-15T10:00:00Z             │
   └─────────────────────────────────────────────────────────┘
```

### 6.3 구성

```go
// Leader election is configured in the Manager options
mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
    LeaderElection:          true,
    LeaderElectionID:        "memcached-operator-lock",
    LeaderElectionNamespace: "memcached-operator-system",
    // Tune timings for faster failover (defaults shown)
    LeaseDuration: durationPtr(15 * time.Second),
    RenewDeadline: durationPtr(10 * time.Second),
    RetryPeriod:   durationPtr(2 * time.Second),
})

func durationPtr(d time.Duration) *time.Duration {
    return &d
}
```

고가용성을 위해 오퍼레이터 배포는 `replicas: 2`를 지정해야 합니다:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: memcached-operator-controller-manager
  namespace: memcached-operator-system
spec:
  replicas: 2
  selector:
    matchLabels:
      control-plane: controller-manager
  template:
    spec:
      containers:
      - name: manager
        image: controller:latest
        args:
        - --leader-elect
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8081
          initialDelaySeconds: 15
          periodSeconds: 20
        readinessProbe:
          httpGet:
            path: /readyz
            port: 8081
          initialDelaySeconds: 5
          periodSeconds: 10
```

---

## 7. 파이널라이저

### 이론: Owner Reference와 Finalizer — 소유권을 명시적으로

`PostgresCluster` 리소스는 자식 객체를 소유합니다 — StatefulSet, 여러 Service, 자격 증명용 Secret, 스토리지용 PVC. 두 메커니즘이 그것들을 묶습니다:

**Owner Reference**는 자식에서 부모를 가리키는 메타데이터입니다:

```yaml
metadata:
  name: my-cluster-sts
  ownerReferences:
    - apiVersion: example.com/v1
      kind: PostgresCluster
      name: my-cluster
      uid: 12345...
      controller: true
      blockOwnerDeletion: true
```

`PostgresCluster`를 삭제하면, **garbage collector** 컨트롤러(kube-controller-manager에 내장)가 매달린 owner reference를 보고 자식을 cascade-delete합니다. StatefulSet에 대한 삭제 로직을 작성하지 않습니다 — 생성 시 `ownerReferences`를 설정했기 때문에 GC가 처리합니다. 이것이 Deployment가 ReplicaSet을 삭제하고 ReplicaSet이 Pod를 삭제하는 방법입니다 — 모두 "무료로".

**Finalizer**는 그 반대입니다 — `metadata.finalizers` 아래의 문자열 목록이 제거될 때까지 삭제를 차단합니다. 사용자가 `kubectl delete postgrescluster my-cluster`를 실행하면:

1. 쿠버네티스가 `metadata.deletionTimestamp`(소프트 삭제 마커)를 설정.
2. Garbage collection이 알아차리지만 finalizer 목록이 비어 있지 않으므로 대기.
3. 컨트롤러의 reconciler가 `deletionTimestamp != nil`을 보고 정리 실행(예: 최종 백업 수행, 모니터링에서 등록 해제, 클라우드 관리 디스크 해제).
4. 정리 후, 컨트롤러가 목록에서 자신의 finalizer 제거.
5. Finalizer가 비어 있으면, GC가 실제로 객체를 삭제.

Finalizer는 "삭제 전 동기 정리"를 하는 유일한 올바른 방법입니다 — 그것 없이는 객체가 반응할 기회를 얻기 전에 사라집니다.

흔한 패턴 — 첫 reconcile에서 finalizer를 등록(업데이트를 통해)하고, 모든 reconcile의 맨 위에서 `deletionTimestamp`를 검사하여 delete-handling으로 분기.

### 7.1 파이널라이저란?

파이널라이저(finalizer)는 삭제 전 훅(pre-deletion hook)을 나타내는 리소스의 키입니다. 파이널라이저가 있는 리소스가 삭제되면 Kubernetes는 `deletionTimestamp`를 설정하지만 모든 파이널라이저가 제거될 때까지 객체를 삭제하지 않습니다.

### 7.2 파이널라이저 라이프사이클

```
User: kubectl delete memcached my-cache
    │
    ▼
API Server sets deletionTimestamp
    │
    ▼
Operator sees deletionTimestamp is non-zero
    │
    ▼
Operator runs cleanup logic
    (e.g., delete external resources, revoke credentials)
    │
    ▼
Operator removes finalizer from object
    │
    ▼
API Server garbage-collects the object
```

### 7.3 구현 패턴

```go
const myFinalizer = "cache.example.com/finalizer"

func (r *MemcachedReconciler) handleFinalizer(ctx context.Context, m *cachev1alpha1.Memcached) (ctrl.Result, error) {
    if m.ObjectMeta.DeletionTimestamp.IsZero() {
        // Not being deleted -- ensure finalizer is present
        if !controllerutil.ContainsFinalizer(m, myFinalizer) {
            controllerutil.AddFinalizer(m, myFinalizer)
            if err := r.Update(ctx, m); err != nil {
                return ctrl.Result{}, err
            }
        }
        return ctrl.Result{}, nil
    }

    // Being deleted -- run cleanup
    if controllerutil.ContainsFinalizer(m, myFinalizer) {
        // Cleanup: delete PVCs that are not garbage-collected
        if err := r.deleteOrphanedPVCs(ctx, m); err != nil {
            return ctrl.Result{}, err
        }

        // Cleanup: remove external DNS record
        if err := r.removeExternalDNS(ctx, m); err != nil {
            return ctrl.Result{}, err
        }

        // Remove the finalizer to allow garbage collection
        controllerutil.RemoveFinalizer(m, myFinalizer)
        if err := r.Update(ctx, m); err != nil {
            return ctrl.Result{}, err
        }
    }

    return ctrl.Result{}, nil
}
```

### 7.4 일반적인 파이널라이저 함정

| 함정 | 결과 | 해결책 |
|---|---|---|
| 파이널라이저 정리가 영원히 멈춤 | 객체가 Terminating 상태에서 멈춤 | 타임아웃과 폴백 로직 추가 |
| 파이널라이저는 추가했지만 정리 로직 제거 | 객체를 영구적으로 삭제할 수 없음 | 항상 삭제 경로를 테스트 |
| 정리 중 `IsNotFound`를 확인하지 않음 | 이미 삭제된 리소스에서 정리 실패 | NotFound 오류 무시 |
| 삭제 중 spec 변경 | 유효성 검증 웹훅이 업데이트를 거부 | 삭제 중에는 metadata/status만 수정 |

---

## 8. 소유자 참조

### 8.1 소유자 참조 작동 방식

소유자 참조(owner reference)는 Kubernetes 객체 간의 부모-자식 관계를 생성합니다. 부모가 삭제되면 가비지 컬렉터(garbage collector)가 자동으로 모든 자식을 삭제합니다.

```go
// Setting owner reference with controller-runtime
func (r *MemcachedReconciler) deploymentForMemcached(m *cachev1alpha1.Memcached) (*appsv1.Deployment, error) {
    dep := &appsv1.Deployment{
        // ... deployment spec ...
    }
    // This sets the Memcached CR as the owner of the Deployment
    if err := ctrl.SetControllerReference(m, dep, r.Scheme); err != nil {
        return nil, err
    }
    return dep, nil
}
```

### 8.2 소유자 참조 필드

```yaml
# The child resource (Deployment) will have:
metadata:
  ownerReferences:
  - apiVersion: cache.example.com/v1alpha1
    kind: Memcached
    name: my-cache
    uid: d9607e19-f88f-11e6-a518-42010a800195
    controller: true        # This owner is the managing controller
    blockOwnerDeletion: true # Block deletion until child is cleaned up
```

### 8.3 네임스페이스 간 소유권

소유자 참조(owner reference)는 네임스페이스 경계를 넘을 수 없습니다. 네임스페이스 간 관계의 경우 레이블과 파이널라이저를 대신 사용하세요:

```go
// Label the resource with the owner's identity
labels := map[string]string{
    "managed-by":      "memcached-operator",
    "owner-name":      m.Name,
    "owner-namespace": m.Namespace,
}
```

### 8.4 Owns() 감시

컨트롤러 설정에서 `Owns()`를 사용하면 controller-runtime이 자동으로 자식 리소스를 감시하고 이벤트를 부모에게 매핑합니다:

```go
func (r *MemcachedReconciler) SetupWithManager(mgr ctrl.Manager) error {
    return ctrl.NewControllerManagedBy(mgr).
        For(&cachev1alpha1.Memcached{}).          // Watch the primary resource
        Owns(&appsv1.Deployment{}).                // Watch owned Deployments
        Owns(&corev1.Service{}).                   // Watch owned Services
        Owns(&corev1.ConfigMap{}).                 // Watch owned ConfigMaps
        WithOptions(controller.Options{
            MaxConcurrentReconciles: 2,
        }).
        Complete(r)
}
```

---

## 9. Operator Lifecycle Manager (OLM)

### 9.1 OLM이란?

OLM은 오퍼레이터 자체의 라이프사이클 -- 설치, 업그레이드, RBAC, 의존성 해결 -- 을 관리합니다. 오퍼레이터를 버전이 지정된 패키지로 취급하는 일급 시민(first-class citizen)으로 다룹니다.

### 9.2 OLM 아키텍처

```
┌──────────────────────────────────────────────────┐
│                  OLM Components                   │
│                                                   │
│  ┌──────────────┐    ┌──────────────────────┐    │
│  │   Catalog    │    │   OLM Operator       │    │
│  │   Operator   │    │   (installs CSVs)    │    │
│  │ (indexes)    │    │                      │    │
│  └──────┬───────┘    └──────────┬───────────┘    │
│         │                       │                │
│         ▼                       ▼                │
│  ┌──────────────┐    ┌──────────────────────┐    │
│  │ CatalogSource│    │ ClusterServiceVersion│    │
│  │ (package     │───▶│ (CSV - describes     │    │
│  │  index)      │    │  an operator version)│    │
│  └──────────────┘    └──────────────────────┘    │
│                              │                   │
│                              ▼                   │
│                      ┌───────────────┐           │
│                      │ Subscription  │           │
│                      │ (auto-update  │           │
│                      │  channel)     │           │
│                      └───────────────┘           │
└──────────────────────────────────────────────────┘
```

### 9.3 OLM 번들 빌드

```bash
# Generate the ClusterServiceVersion (CSV)
operator-sdk generate kustomize manifests

# Build the bundle (CRDs + CSV + metadata)
make bundle IMG=example.com/memcached-operator:v0.1.0

# Bundle directory structure
# bundle/
# ├── manifests/
# │   ├── cache.example.com_memcacheds.yaml    # CRD
# │   └── memcached-operator.clusterserviceversion.yaml  # CSV
# ├── metadata/
# │   └── annotations.yaml
# └── tests/
#     └── scorecard/
#         └── config.yaml

# Build and push the bundle image
make bundle-build bundle-push BUNDLE_IMG=example.com/memcached-operator-bundle:v0.1.0

# Build a catalog containing the bundle
make catalog-build catalog-push CATALOG_IMG=example.com/memcached-operator-catalog:v0.1.0
```

### 9.4 OLM을 통한 오퍼레이터 설치

```bash
# Install OLM itself (if not pre-installed)
operator-sdk olm install

# Create a CatalogSource
kubectl apply -f - <<EOF
apiVersion: operators.coreos.com/v1alpha1
kind: CatalogSource
metadata:
  name: memcached-operator-catalog
  namespace: olm
spec:
  sourceType: grpc
  image: example.com/memcached-operator-catalog:v0.1.0
  displayName: Memcached Operator Catalog
  updateStrategy:
    registryPoll:
      interval: 10m
EOF

# Create a Subscription to install the operator
kubectl apply -f - <<EOF
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: memcached-operator
  namespace: operators
spec:
  channel: alpha
  name: memcached-operator
  source: memcached-operator-catalog
  sourceNamespace: olm
  installPlanApproval: Automatic
EOF
```

### 9.5 업그레이드 채널

```yaml
# CSV defines the upgrade path
apiVersion: operators.coreos.com/v1alpha1
kind: ClusterServiceVersion
metadata:
  name: memcached-operator.v0.2.0
spec:
  replaces: memcached-operator.v0.1.0  # Upgrade from v0.1.0
  version: 0.2.0
  # skips can be used to skip intermediate versions
  # skips:
  # - memcached-operator.v0.1.1
```

---

## 10. 모범 사례와 안티패턴

### 10.1 모범 사례

**멱등한 조정(Idempotent reconciliation)**: Reconcile을 호출할 때마다 동일한 클러스터 상태에서 동일한 결과를 생성해야 합니다.

```go
// GOOD: Create-or-update pattern
func (r *Reconciler) reconcileDeployment(ctx context.Context, m *cachev1alpha1.Memcached) error {
    dep := &appsv1.Deployment{
        ObjectMeta: metav1.ObjectMeta{
            Name:      m.Name,
            Namespace: m.Namespace,
        },
    }
    op, err := controllerutil.CreateOrUpdate(ctx, r.Client, dep, func() error {
        // Mutate the deployment to match desired state
        dep.Spec.Replicas = &m.Spec.Size
        dep.Spec.Template.Spec.Containers[0].Image = m.Spec.Image
        return ctrl.SetControllerReference(m, dep, r.Scheme)
    })
    if err != nil {
        return err
    }
    log.FromContext(ctx).Info("Deployment reconciled", "operation", op)
    return nil
}
```

**status condition 사용**: 표준 condition 타입을 사용하여 의미 있는 status를 보고합니다.

```go
meta.SetStatusCondition(&m.Status.Conditions, metav1.Condition{
    Type:               "Degraded",
    Status:             metav1.ConditionTrue,
    Reason:             "InsufficientReplicas",
    Message:            "Only 1 of 3 desired replicas are ready",
    ObservedGeneration: m.Generation,
})
```

**이벤트 발생**: 사용자에게 보이는 작업에 이벤트 레코더를 사용합니다.

```go
r.Recorder.Event(m, corev1.EventTypeNormal, "Upgraded", "Memcached version upgraded to 1.6.18")
r.Recorder.Event(m, corev1.EventTypeWarning, "BackupFailed", "Scheduled backup failed: connection timeout")
```

### 10.2 안티패턴

| 안티패턴 | 문제 | 해결책 |
|---|---|---|
| Reconcile에서 goroutine으로 I/O 수행 | 추적되지 않는 작업, 리소스 누수 | 모든 작업을 Reconcile 내에서 유지 |
| `watch.Interface` 직접 사용 | 캐시를 우회하고 API 부하 생성 | 컨트롤러 설정을 통한 인포머(informer) 사용 |
| 컨트롤러 구조체에 상태 저장 | 재시작 시 손실, 리더 선출 중단 | CR status나 ConfigMap에 상태 저장 |
| 외부 호출에 속도 제한 없음 | 오퍼레이터 재시작 시 썬더링 허드(thundering herd) | 워크큐 속도 제한기 사용 |
| `Generation` vs `ResourceVersion` 무시 | 모든 status 업데이트에서 조정 수행 | `GenerationChangedPredicate`로 필터링 |
| 오퍼레이터 Pod에 리소스 제한 미설정 | 부하 시 오퍼레이터가 OOMKilled | request와 limit 설정 |

### 10.3 Predicate로 이벤트 필터링

```go
import "sigs.k8s.io/controller-runtime/pkg/predicate"

func (r *MemcachedReconciler) SetupWithManager(mgr ctrl.Manager) error {
    return ctrl.NewControllerManagedBy(mgr).
        For(&cachev1alpha1.Memcached{}, builder.WithPredicates(
            predicate.GenerationChangedPredicate{}, // Only spec changes
        )).
        Owns(&appsv1.Deployment{}).
        Complete(r)
}
```

### 10.4 오퍼레이터 테스팅

```go
// Using envtest for integration testing
package controller

import (
    "context"
    "time"

    . "github.com/onsi/ginkgo/v2"
    . "github.com/onsi/gomega"
    appsv1 "k8s.io/api/apps/v1"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/types"

    cachev1alpha1 "github.com/example/memcached-operator/api/v1alpha1"
)

var _ = Describe("Memcached controller", func() {
    const (
        timeout  = time.Second * 30
        interval = time.Second * 1
    )

    Context("When creating a Memcached CR", func() {
        It("Should create a Deployment with correct replica count", func() {
            ctx := context.Background()
            memcached := &cachev1alpha1.Memcached{
                ObjectMeta: metav1.ObjectMeta{
                    Name:      "test-memcached",
                    Namespace: "default",
                },
                Spec: cachev1alpha1.MemcachedSpec{
                    Size:  3,
                    Image: "memcached:1.6-alpine",
                },
            }
            Expect(k8sClient.Create(ctx, memcached)).Should(Succeed())

            deploymentKey := types.NamespacedName{Name: "test-memcached", Namespace: "default"}
            createdDeployment := &appsv1.Deployment{}

            Eventually(func() bool {
                err := k8sClient.Get(ctx, deploymentKey, createdDeployment)
                return err == nil
            }, timeout, interval).Should(BeTrue())

            Expect(*createdDeployment.Spec.Replicas).Should(Equal(int32(3)))
        })
    })
})
```

---

## 연습문제

### 연습문제 1: 오퍼레이터 스캐폴딩

operator-sdk 또는 kubebuilder를 사용하여 커스텀 `Redis` 리소스를 관리하는 새 오퍼레이터 프로젝트를 스캐폴딩하세요. CRD에는 `replicas` (int32), `version` (string), `persistenceEnabled` (bool) spec 필드가 있어야 합니다. 적절한 유효성 검증 마커를 포함한 완전한 `redis_types.go` 파일을 작성하세요.

<details>
<summary>정답 보기</summary>

```bash
# Scaffold the project
mkdir redis-operator && cd redis-operator
operator-sdk init --domain example.com --repo github.com/example/redis-operator
operator-sdk create api --group database --version v1alpha1 --kind Redis --resource --controller
```

```go
// api/v1alpha1/redis_types.go
package v1alpha1

import (
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type RedisSpec struct {
    // +kubebuilder:validation:Minimum=1
    // +kubebuilder:validation:Maximum=6
    // +kubebuilder:default:=1
    Replicas int32 `json:"replicas"`

    // +kubebuilder:validation:Pattern=`^[0-9]+\.[0-9]+\.[0-9]+$`
    // +kubebuilder:default:="7.2.4"
    Version string `json:"version,omitempty"`

    // +kubebuilder:default:=false
    PersistenceEnabled bool `json:"persistenceEnabled,omitempty"`
}

type RedisStatus struct {
    Conditions    []metav1.Condition `json:"conditions,omitempty"`
    ReadyReplicas int32             `json:"readyReplicas,omitempty"`
    Phase         string            `json:"phase,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Replicas",type="integer",JSONPath=".spec.replicas"
// +kubebuilder:printcolumn:name="Version",type="string",JSONPath=".spec.version"
// +kubebuilder:printcolumn:name="Ready",type="integer",JSONPath=".status.readyReplicas"
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
type Redis struct {
    metav1.TypeMeta   `json:",inline"`
    metav1.ObjectMeta `json:"metadata,omitempty"`
    Spec              RedisSpec   `json:"spec,omitempty"`
    Status            RedisStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true
type RedisList struct {
    metav1.TypeMeta `json:",inline"`
    metav1.ListMeta `json:"metadata,omitempty"`
    Items           []Redis `json:"items"`
}

func init() {
    SchemeBuilder.Register(&Redis{}, &RedisList{})
}
```

</details>

### 연습문제 2: 조정 루프 구현

연습문제 1의 Redis 오퍼레이터를 위한 완전한 Reconcile 함수를 작성하세요. 컨트롤러는 다음을 수행해야 합니다: (a) 지정된 수의 레플리카로 StatefulSet 생성, (b) StatefulSet을 위한 헤드리스 Service 생성, (c) ready 레플리카 수와 Phase 필드(Pending/Running/Failed)로 status 업데이트.

<details>
<summary>정답 보기</summary>

```go
func (r *RedisReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    log := log.FromContext(ctx)

    redis := &databasev1alpha1.Redis{}
    if err := r.Get(ctx, req.NamespacedName, redis); err != nil {
        if apierrors.IsNotFound(err) {
            return ctrl.Result{}, nil
        }
        return ctrl.Result{}, err
    }

    // Reconcile the headless Service
    svc := &corev1.Service{
        ObjectMeta: metav1.ObjectMeta{
            Name:      redis.Name + "-headless",
            Namespace: redis.Namespace,
        },
    }
    _, err := controllerutil.CreateOrUpdate(ctx, r.Client, svc, func() error {
        svc.Spec.ClusterIP = "None"
        svc.Spec.Selector = map[string]string{"app": redis.Name}
        svc.Spec.Ports = []corev1.ServicePort{{
            Port:     6379,
            Name:     "redis",
            Protocol: corev1.ProtocolTCP,
        }}
        return ctrl.SetControllerReference(redis, svc, r.Scheme)
    })
    if err != nil {
        return ctrl.Result{}, err
    }

    // Reconcile the StatefulSet
    sts := &appsv1.StatefulSet{
        ObjectMeta: metav1.ObjectMeta{
            Name:      redis.Name,
            Namespace: redis.Namespace,
        },
    }
    _, err = controllerutil.CreateOrUpdate(ctx, r.Client, sts, func() error {
        replicas := redis.Spec.Replicas
        labels := map[string]string{"app": redis.Name}
        sts.Spec.Replicas = &replicas
        sts.Spec.ServiceName = redis.Name + "-headless"
        sts.Spec.Selector = &metav1.LabelSelector{MatchLabels: labels}
        sts.Spec.Template = corev1.PodTemplateSpec{
            ObjectMeta: metav1.ObjectMeta{Labels: labels},
            Spec: corev1.PodSpec{
                Containers: []corev1.Container{{
                    Name:  "redis",
                    Image: "redis:" + redis.Spec.Version,
                    Ports: []corev1.ContainerPort{{ContainerPort: 6379}},
                }},
            },
        }
        return ctrl.SetControllerReference(redis, sts, r.Scheme)
    })
    if err != nil {
        log.Error(err, "Failed to reconcile StatefulSet")
        return ctrl.Result{}, err
    }

    // Update status
    existingSts := &appsv1.StatefulSet{}
    if err := r.Get(ctx, types.NamespacedName{Name: redis.Name, Namespace: redis.Namespace}, existingSts); err != nil {
        return ctrl.Result{}, err
    }
    redis.Status.ReadyReplicas = existingSts.Status.ReadyReplicas
    if existingSts.Status.ReadyReplicas == redis.Spec.Replicas {
        redis.Status.Phase = "Running"
    } else if existingSts.Status.ReadyReplicas > 0 {
        redis.Status.Phase = "Pending"
    } else {
        redis.Status.Phase = "Pending"
    }
    if err := r.Status().Update(ctx, redis); err != nil {
        return ctrl.Result{}, err
    }

    return ctrl.Result{RequeueAfter: 30 * time.Second}, nil
}
```

</details>

### 연습문제 3: 파이널라이저 구현

Redis 오퍼레이터에 CR이 삭제될 때 다음 정리를 수행하는 파이널라이저를 추가하세요: (a) `redis-cli BGSAVE`를 실행하는 Job을 생성하여 최종 백업 수행, (b) Job이 완료될 때까지 대기, (c) 파이널라이저 제거. 백업 Job이 실패하는 경우를 처리하세요.

<details>
<summary>정답 보기</summary>

```go
const redisFinalizer = "database.example.com/backup-finalizer"

func (r *RedisReconciler) handleDeletion(ctx context.Context, redis *databasev1alpha1.Redis) (ctrl.Result, error) {
    log := log.FromContext(ctx)

    if !redis.ObjectMeta.DeletionTimestamp.IsZero() {
        if controllerutil.ContainsFinalizer(redis, redisFinalizer) {
            // Check if backup Job already exists
            backupJob := &batchv1.Job{}
            jobName := redis.Name + "-final-backup"
            err := r.Get(ctx, types.NamespacedName{Name: jobName, Namespace: redis.Namespace}, backupJob)

            if apierrors.IsNotFound(err) {
                // Create the backup Job
                job := &batchv1.Job{
                    ObjectMeta: metav1.ObjectMeta{
                        Name:      jobName,
                        Namespace: redis.Namespace,
                    },
                    Spec: batchv1.JobSpec{
                        Template: corev1.PodTemplateSpec{
                            Spec: corev1.PodSpec{
                                RestartPolicy: corev1.RestartPolicyNever,
                                Containers: []corev1.Container{{
                                    Name:    "backup",
                                    Image:   "redis:7.2",
                                    Command: []string{"redis-cli", "-h", redis.Name + "-headless", "BGSAVE"},
                                }},
                            },
                        },
                        BackoffLimit: int32Ptr(3),
                    },
                }
                if err := r.Create(ctx, job); err != nil {
                    log.Error(err, "Failed to create backup job")
                    return ctrl.Result{}, err
                }
                log.Info("Created final backup job", "job", jobName)
                return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
            } else if err != nil {
                return ctrl.Result{}, err
            }

            // Check Job status
            if backupJob.Status.Succeeded > 0 {
                log.Info("Backup completed successfully")
            } else if backupJob.Status.Failed >= 3 {
                log.Info("Backup failed after 3 attempts, proceeding with deletion")
                r.Recorder.Event(redis, corev1.EventTypeWarning, "BackupFailed",
                    "Final backup failed, proceeding with deletion")
            } else {
                // Job still running
                return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
            }

            // Remove the finalizer
            controllerutil.RemoveFinalizer(redis, redisFinalizer)
            if err := r.Update(ctx, redis); err != nil {
                return ctrl.Result{}, err
            }
        }
        return ctrl.Result{}, nil
    }

    // Not being deleted -- ensure finalizer is present
    if !controllerutil.ContainsFinalizer(redis, redisFinalizer) {
        controllerutil.AddFinalizer(redis, redisFinalizer)
        if err := r.Update(ctx, redis); err != nil {
            return ctrl.Result{}, err
        }
    }
    return ctrl.Result{}, nil
}

func int32Ptr(i int32) *int32 { return &i }
```

</details>

### 연습문제 4: 리더 선출 구성

고가용성을 위해 3개의 레플리카로 오퍼레이터를 배포해야 합니다. 다음을 수행하는 Deployment 매니페스트와 manager 구성 코드를 작성하세요: (a) 리더 선출 활성화, (b) 10초 리스 기간과 7초 갱신 기한 구성, (c) health 및 readiness 프로브 추가, (d) 적절한 리소스 request와 limit 설정.

<details>
<summary>정답 보기</summary>

```go
// Manager setup with leader election tuning
mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
    Scheme:                 scheme,
    LeaderElection:         true,
    LeaderElectionID:       "redis-operator-lock",
    LeaseDuration:          durationPtr(10 * time.Second),
    RenewDeadline:          durationPtr(7 * time.Second),
    RetryPeriod:            durationPtr(2 * time.Second),
    HealthProbeBindAddress: ":8081",
    Metrics: metricsserver.Options{
        BindAddress: ":8080",
    },
})
if err != nil {
    setupLog.Error(err, "unable to start manager")
    os.Exit(1)
}

mgr.AddHealthzCheck("healthz", healthz.Ping)
mgr.AddReadyzCheck("readyz", healthz.Ping)
```

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis-operator-controller-manager
  namespace: redis-operator-system
spec:
  replicas: 3
  selector:
    matchLabels:
      control-plane: controller-manager
  template:
    metadata:
      labels:
        control-plane: controller-manager
    spec:
      serviceAccountName: redis-operator-controller-manager
      terminationGracePeriodSeconds: 10
      containers:
      - name: manager
        image: example.com/redis-operator:v0.1.0
        command:
        - /manager
        args:
        - --leader-elect
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
          periodSeconds: 20
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /readyz
            port: 8081
          initialDelaySeconds: 5
          periodSeconds: 10
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 256Mi
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
          readOnlyRootFilesystem: true
          runAsNonRoot: true
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchLabels:
                  control-plane: controller-manager
              topologyKey: kubernetes.io/hostname
```

</details>

### 연습문제 5: OLM 번들

Memcached 오퍼레이터 v0.2.0의 ClusterServiceVersion(CSV) 매니페스트를 생성하세요. 다음을 포함해야 합니다: (a) v0.1.0을 대체, (b) `certificates.cert-manager.io/v1`에 대한 필수 API 의존성 선언, (c) 오퍼레이터의 필수 RBAC 권한 지정, (d) 컨트롤러 배포를 포함한 설치 전략.

<details>
<summary>정답 보기</summary>

```yaml
apiVersion: operators.coreos.com/v1alpha1
kind: ClusterServiceVersion
metadata:
  name: memcached-operator.v0.2.0
  namespace: placeholder
  annotations:
    alm-examples: |
      [
        {
          "apiVersion": "cache.example.com/v1alpha1",
          "kind": "Memcached",
          "metadata": {"name": "memcached-sample"},
          "spec": {"size": 3, "containerPort": 11211}
        }
      ]
    capabilities: "Full Lifecycle"
    categories: "Database"
spec:
  displayName: Memcached Operator
  description: Manages Memcached clusters on Kubernetes
  version: 0.2.0
  replaces: memcached-operator.v0.1.0
  maturity: beta
  minKubeVersion: "1.25.0"

  maintainers:
  - name: Example Inc.
    email: ops@example.com

  customresourcedefinitions:
    owned:
    - name: memcacheds.cache.example.com
      version: v1alpha1
      kind: Memcached
      displayName: Memcached
      description: A Memcached cluster
      statusDescriptors:
      - path: readyReplicas
        displayName: Ready Replicas
        description: Number of ready replicas

    required:
    - name: certificates.cert-manager.io
      version: v1
      kind: Certificate
      displayName: Certificate
      description: TLS certificate management

  install:
    strategy: deployment
    spec:
      clusterPermissions:
      - serviceAccountName: memcached-operator-controller-manager
        rules:
        - apiGroups: ["cache.example.com"]
          resources: ["memcacheds"]
          verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
        - apiGroups: ["cache.example.com"]
          resources: ["memcacheds/status"]
          verbs: ["get", "update", "patch"]
        - apiGroups: ["cache.example.com"]
          resources: ["memcacheds/finalizers"]
          verbs: ["update"]
        - apiGroups: ["apps"]
          resources: ["deployments"]
          verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
        - apiGroups: [""]
          resources: ["pods", "services", "configmaps"]
          verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
        - apiGroups: [""]
          resources: ["events"]
          verbs: ["create", "patch"]
        - apiGroups: ["coordination.k8s.io"]
          resources: ["leases"]
          verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
      deployments:
      - name: memcached-operator-controller-manager
        spec:
          replicas: 2
          selector:
            matchLabels:
              control-plane: controller-manager
          template:
            metadata:
              labels:
                control-plane: controller-manager
            spec:
              serviceAccountName: memcached-operator-controller-manager
              containers:
              - name: manager
                image: example.com/memcached-operator:v0.2.0
                args:
                - --leader-elect
                ports:
                - containerPort: 8080
                  name: metrics
                resources:
                  requests:
                    cpu: 100m
                    memory: 128Mi
                  limits:
                    cpu: 500m
                    memory: 256Mi

  installModes:
  - type: OwnNamespace
    supported: true
  - type: SingleNamespace
    supported: true
  - type: MultiNamespace
    supported: false
  - type: AllNamespaces
    supported: true
```

</details>

---

**이전**: [커스텀 리소스 정의](./10_Custom_Resource_Definitions.md) | **다음**: [어드미션 컨트롤러](./12_Admission_Controllers.md)

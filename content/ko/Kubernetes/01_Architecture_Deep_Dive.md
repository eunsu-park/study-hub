# 01. 아키텍처 심층 분석

**이전**: [개요](./00_Overview.md) | **다음**: [워크로드 리소스](./02_Workload_Resources.md)

## 학습 목표
- 쿠버네티스 컨트롤 플레인(Control Plane)과 노드 컴포넌트를 깊이 이해한다
- kubectl에서 etcd까지 API 요청의 전체 라이프사이클을 추적한다
- etcd 데이터 모델과 쿠버네티스 오브젝트가 저장되는 방식을 설명한다
- 필터링 및 스코어링 단계를 포함한 스케줄러(Scheduler) 알고리즘을 설명한다
- 컨트롤 루프(Control Loop)와 조정(Reconciliation)이 선언적 모델을 어떻게 구동하는지 설명한다

---

쿠버네티스는 선언적 API와 실제 상태를 원하는 상태로 지속적으로 조정하는 독립적인
컨트롤러 집합을 중심으로 설계된 분산 시스템입니다. 아키텍처를 이해하는 것은
효과적인 운영, 디버깅, 확장에 필수적입니다. 이 레슨에서는 모든 주요 컴포넌트,
요청 라이프사이클, 그리고 쿠버네티스를 자가 치유(Self-Healing)하게 만드는
컨트롤 루프 패턴을 분석합니다.

## 목차
1. [상위 수준 아키텍처](#1-상위-수준-아키텍처)
2. [컨트롤 플레인 컴포넌트](#2-컨트롤-플레인-컴포넌트)
3. [노드 컴포넌트](#3-노드-컴포넌트)
4. [API 요청 라이프사이클](#4-api-요청-라이프사이클)
5. [etcd 데이터 모델](#5-etcd-데이터-모델)
6. [쿠버네티스 오브젝트 모델 (GVR / GVK)](#6-쿠버네티스-오브젝트-모델-gvr--gvk)
7. [인증과 인가 흐름](#7-인증과-인가-흐름)
8. [스케줄러 알고리즘](#8-스케줄러-알고리즘)
9. [컨트롤 루프와 조정](#9-컨트롤-루프와-조정)
10. [연습문제](#연습문제)

---

## 1. 상위 수준 아키텍처

쿠버네티스는 허브 앤 스포크(Hub-and-Spoke) 토폴로지를 따릅니다. **컨트롤 플레인(Control Plane)**이
허브이며, API 서버를 단일 진입점으로 노출합니다. **노드(Node)**(워커 머신)는
스포크이며, 각각 API 서버와 통신하는 에이전트(kubelet)를 실행합니다.

```
┌──────────────────────────────────────────────────────────┐
│                     Control Plane                        │
│  ┌────────────┐  ┌───────┐  ┌───────────┐  ┌─────────┐ │
│  │ API Server │──│ etcd  │  │ Scheduler │  │ CM / CCM│ │
│  └─────┬──────┘  └───────┘  └───────────┘  └─────────┘ │
│        │                                                 │
└────────┼─────────────────────────────────────────────────┘
         │  Watch / List / Update
    ┌────┴────┬────────────┬────────────┐
    │ Node 1  │  Node 2    │  Node N    │
    │ kubelet │  kubelet   │  kubelet   │
    │ kproxy  │  kproxy    │  kproxy    │
    │ runtime │  runtime   │  runtime   │
    └─────────┴────────────┴────────────┘
```

### 1.1 설계 원칙

- **선언적 > 명령적**: 사용자가 원하는 상태를 선언하면, 컨트롤러가 수렴합니다.
- **레벨 트리거, 엣지 트리거 아님**: 컨트롤러는 이벤트가 아닌 현재 세계의 상태에
  반응합니다. 컨트롤러가 재시작되면 전체 상태를 다시 읽고 거기서부터 계속합니다.
- **데이터 경로에 단일 장애점 없음**: 컨트롤 플레인이 일시적으로 다운되어도
  실행 중인 워크로드에는 영향을 주지 않습니다. 파드(Pod)는 계속 실행되며,
  변경 사항만 지연됩니다.

### 1.2 프로덕션에서의 클러스터 토폴로지

프로덕션급 클러스터는 일반적으로 다음과 같이 실행됩니다:

| 컴포넌트 | 권장 수량 | 비고 |
|-----------|------------------|-------|
| API 서버 | 3+ (LB 뒤) | 무상태; 수평 확장 |
| etcd | 3 또는 5 (홀수) | Raft 쿼럼은 과반수 필요 |
| 스케줄러(Scheduler) | 1 활성 + 대기 | 리스(Lease)를 통한 리더 선출 |
| 컨트롤러 매니저(Controller Manager) | 1 활성 + 대기 | 리스(Lease)를 통한 리더 선출 |
| 워커 노드 | 가변적 | 클러스터당 최대 5,000개 |

### 1.3 Minikube로 탐색하기

```bash
# 로컬 클러스터 시작
minikube start --nodes 2 --driver=docker

# 노드 목록 확인
kubectl get nodes -o wide

# 컨트롤 플레인 파드 확인
kubectl get pods -n kube-system

# API 서버 검사
kubectl describe pod -n kube-system kube-apiserver-minikube
```

---

## 2. 컨트롤 플레인 컴포넌트

### 2.1 kube-apiserver

API 서버는 클러스터의 프론트 도어입니다. `kubectl`, 대시보드, 인클러스터
컨트롤러 등 모든 작업이 API 서버를 통해 이루어집니다.

주요 책임:
- **RESTful API**: 모든 쿠버네티스 오브젝트에 대한 CRUD 작업
- **어드미션 제어(Admission Control)**: 변형(Mutating) 및 검증(Validating) 웹훅
- **인증 및 인가(Authentication & Authorization)**: 플러그인 기반 (7절 참조)
- **감시 메커니즘(Watch Mechanism)**: 변경 알림을 위한 장기 HTTP 스트림
- **OpenAPI 스키마 제공**: 클라이언트 측 검증 활성화

```bash
# API 서버의 커맨드라인 인수 확인
kubectl -n kube-system get pod kube-apiserver-minikube -o jsonpath='{.spec.containers[0].command}' | python3 -m json.tool
```

API 서버는 **무상태(Stateless)**입니다. 모든 영속 상태는 etcd에 저장됩니다. 이는
고가용성을 위해 로드 밸런서 뒤에서 여러 API 서버 레플리카를 실행할 수 있음을 의미합니다.

### 2.2 etcd

etcd는 클러스터의 단일 진실 공급원(Single Source of Truth) 역할을 하는
분산, 강일관성 키-값 저장소입니다.

속성:
- **Raft 합의(Consensus)**: `n`개 멤버 중 `(n-1)/2`개 노드 장애 허용
- **기본적으로 직렬화 가능한 읽기(Serializable Reads)**: 선형화 가능 읽기(Linearizable Reads)로 설정 가능
- **감시 지원(Watch Support)**: 클라이언트가 정렬된 변경 알림을 수신
- **MVCC**: 다중 버전 동시성 제어로 효율적인 감시 및 압축 가능

```bash
# minikube 내부에서 etcd 직접 쿼리 (학습 목적으로만)
minikube ssh

# etcd는 /registry/<resource>/<namespace>/<name> 하에 데이터 저장
# 예시: /registry/pods/default/my-pod

# etcd 상태 확인 (etcdctl이 사용 가능한 경우)
ETCDCTL_API=3 etcdctl \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/var/lib/minikube/certs/etcd/ca.crt \
  --cert=/var/lib/minikube/certs/etcd/server.crt \
  --key=/var/lib/minikube/certs/etcd/server.key \
  endpoint health
```

### 2.3 kube-scheduler

스케줄러(Scheduler)는 노드가 할당되지 않은 새로 생성된 파드(Pod)를 감시하고
각 파드에 가장 적합한 노드를 선택합니다. 알고리즘은 8절에서 자세히 다룹니다.

```bash
# 스케줄러 설정 확인
kubectl -n kube-system describe pod kube-scheduler-minikube

# 스케줄러 로그 확인
kubectl -n kube-system logs kube-scheduler-minikube --tail=20
```

### 2.4 kube-controller-manager

컨트롤러 매니저(Controller Manager)는 각각 특정 컨트롤 루프를 구현하는
컨트롤러 모음을 실행합니다. 예시:

| 컨트롤러 | 감시 대상 | 관리 대상 |
|------------|---------|---------|
| 디플로이먼트(Deployment) | 디플로이먼트 | 레플리카셋(ReplicaSets) |
| 레플리카셋(ReplicaSet) | 레플리카셋 | 파드(Pods) |
| 노드(Node) | 노드 | 테인트(Taints), 퇴거(Evictions) |
| 잡(Job) | 잡 | 파드(Pods) |
| 엔드포인트(Endpoint) | 서비스, 파드 | 엔드포인트(Endpoints) |
| 서비스어카운트(ServiceAccount) | 네임스페이스 | 기본 SA + 토큰 |
| 가비지 컬렉터(Garbage Collector) | 소유자 참조(Owner References) | 연쇄 삭제(Cascading Deletion) |

모든 컨트롤러는 하나의 프로세스를 공유하지만 논리적으로는 독립적입니다. 각 컨트롤러는
인포머(Informer)(캐시된 감시 스트림)가 공급하는 **작업 큐(Work Queue)**에서 작동합니다.

```bash
# 컨트롤러 매니저 내부에서 실행 중인 모든 컨트롤러 나열
kubectl -n kube-system get pod kube-controller-manager-minikube \
  -o jsonpath='{.spec.containers[0].command}' | tr ',' '\n' | grep controllers
```

### 2.5 cloud-controller-manager

클라우드 환경에서는 별도의 바이너리가 클라우드 특화 로직을 처리합니다:
- 노드 라이프사이클 (VM이 삭제된 경우 감지)
- 라우트(Route) 설정
- 로드 밸런서(Load Balancer) 프로비저닝
- 볼륨 연결 (레거시; 현재는 CSI)

이 분리를 통해 클라우드 공급자가 자체 릴리스 주기에 맞춰 배포할 수 있습니다.

---

## 3. 노드 컴포넌트

### 3.1 kubelet

kubelet은 모든 노드의 주요 에이전트입니다. 노드를 API 서버에 등록하고
PodSpec에 기술된 컨테이너가 실행 중이고 정상인지 확인합니다.

책임:
- 파드(Pod) 라이프사이클 관리 (시작, 중지, 재시작)
- 활성(Liveness), 준비(Readiness), 시작(Startup) 프로브 실행
- 리소스 보고 (CPU, 메모리, 스토리지, PID)
- 컨테이너 로그 관리
- CSI 및 디바이스 플러그인 인터페이스
- 정적 파드(Static Pod) 관리 (로컬 디렉토리에서 읽기)

```bash
# minikube에서 kubelet 상태 확인
minikube ssh -- systemctl status kubelet

# kubelet 로그 확인
minikube ssh -- journalctl -u kubelet --no-pager --tail=30
```

### 3.2 kube-proxy

kube-proxy는 각 노드에서 네트워크 규칙을 프로그래밍하여 서비스(Service) 추상화를
구현합니다. 서비스 및 엔드포인트슬라이스(EndpointSlice) 오브젝트를 감시하고 다음을 설정합니다:

- **iptables 모드** (기본): 서비스 IP에서 파드 IP로의 변환을 위한 NAT 규칙 생성
- **IPVS 모드**: L4 로드 밸런싱을 위한 Linux IPVS 커널 모듈 사용
- **nftables 모드** (v1.29+): iptables 대신 nftables 사용

```bash
# kube-proxy 모드 확인
kubectl -n kube-system get configmap kube-proxy -o yaml | grep mode

# kube-proxy가 생성한 iptables 규칙 나열 (노드에서)
minikube ssh -- sudo iptables -t nat -L KUBE-SERVICES -n | head -20
```

### 3.3 컨테이너 런타임(Container Runtime)

쿠버네티스는 **컨테이너 런타임 인터페이스(Container Runtime Interface, CRI)**를 통해
컨테이너 런타임과 통신합니다. 주요 런타임:

| 런타임 | 설명 |
|---------|------------|
| containerd | 업계 표준, Docker에서 추출 |
| CRI-O | 쿠버네티스용으로 설계된 OCI 네이티브 런타임 |
| Docker (cri-dockerd 경유) | 레거시; 심(Shim) 필요 |

```bash
# minikube가 사용하는 런타임 확인
minikube ssh -- crictl info | head -5

# CRI를 통해 실행 중인 컨테이너 나열
minikube ssh -- crictl ps
```

---

## 4. API 요청 라이프사이클

`kubectl apply -f pod.yaml`을 실행하면, 요청은 정확한 파이프라인을 거칩니다:

```
kubectl → HTTP request → API Server
                           │
                    ┌──────┴───────┐
                    │ Authentication│  (당신은 누구인가?)
                    ├──────────────┤
                    │ Authorization │  (이 작업을 할 수 있는가?)
                    ├──────────────┤
                    │ Admission     │  (Mutating → Validating)
                    ├──────────────┤
                    │ Validation    │  (스키마 + 커스텀)
                    ├──────────────┤
                    │ etcd Write    │  (오브젝트 영속화)
                    ├──────────────┤
                    │ Post-hooks    │  (인포머 알림)
                    └──────────────┘
```

### 4.1 단계별 추적

1. **kubectl**이 kubeconfig를 읽고, API 서버 URL과 자격 증명을 확인합니다
2. **HTTP 요청** 구성: `POST /api/v1/namespaces/default/pods`
3. **TLS 종료**: API 서버가 클라이언트 인증서(또는 토큰)를 검증합니다
4. **인증(Authentication)**: 설정된 인증기 중 하나가 신원을 검증합니다
5. **인가(Authorization)**: RBAC(또는 다른 인가기)이 사용자가 `default` 네임스페이스에
   파드를 생성할 수 있는지 확인합니다
6. **변형 어드미션(Mutating Admission)**: 웹훅이 사이드카를 주입하거나, 기본값을 설정하거나, 레이블을 추가할 수 있습니다
7. **오브젝트 기본값**: API 서버가 누락된 필드를 채웁니다 (예: `restartPolicy`)
8. **검증 어드미션(Validating Admission)**: 웹훅이 요청을 거부할 수 있습니다 (예: 정책 검사)
9. **스키마 검증**: OpenAPI 스키마에 대해 오브젝트가 검증됩니다
10. **etcd 쓰기**: 오브젝트가 직렬화(protobuf)되어 etcd에 기록됩니다
11. **감시 알림(Watch Notifications)**: 모든 감시자(스케줄러, 컨트롤러)에게 알림이 전송됩니다

### 4.2 라이프사이클 관찰

```bash
# 전체 요청을 보기 위해 상세 출력 활성화
kubectl apply -f pod.yaml -v=8

# 감사 로깅으로 API 호출 추적 (API 서버 플래그 필요)
# --audit-log-path=/var/log/kube-audit.log
# --audit-policy-file=/etc/kubernetes/audit-policy.yaml
```

감사 정책 예시:

```yaml
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
  - level: RequestResponse
    resources:
      - group: ""
        resources: ["pods"]
    verbs: ["create", "update", "delete"]
  - level: Metadata
    resources:
      - group: ""
        resources: ["services", "configmaps"]
  - level: None
    resources:
      - group: ""
        resources: ["events"]
```

---

## 5. etcd 데이터 모델

### 이론: etcd와 Raft 합의 알고리즘

모든 쿠버네티스 객체는 **etcd**에 저장됩니다. etcd는 합의를 위해 **Raft**를 사용하는 분산 키-값 저장소입니다. Raft가 부분 장애 상황에서 etcd의 정합성을 보장합니다 — 컨트롤 플레인 노드가 쓰기 도중 크래시해도, 동일한 파드에 대해 두 API 서버가 서로 다른 상태를 보고하는 일은 발생하지 않습니다.

Raft는 (보통 3개 또는 5개) etcd 멤버 중에서 단일 **리더(leader)**를 선출합니다. 모든 쓰기는 리더를 거쳐 복제 로그(replicated log)에 추가됩니다. 쓰기는 **쿼럼(quorum, 과반)**의 멤버가 로그 엔트리를 영속화한 후에야 응답됩니다. 이로부터 두 가지 핵심 속성이 보장됩니다:

- **선형성(Linearizability)**: 쓰기가 응답된 이후의 모든 읽기는 그 쓰기를 본다. etcd 관점에서 stale read는 존재하지 않습니다.
- **스플릿 브레인(Split-brain) 방지**: 소수 파티션은 진행할 수 없습니다(쿼럼에 도달 불가). 따라서 분리된 두 절반이 모두 리더를 선출하고 쓰기를 받는 일은 일어나지 않습니다.

왜 etcd 멤버를 **홀수**로 두는가? 3-멤버 클러스터는 1대 장애를 견딥니다(2/3가 여전히 쿼럼). 4-멤버 클러스터도 1대 장애만 견디면서(과반은 3/4 필요) 동일한 내결함성에 대해 쓰기 지연은 두 배가 됩니다. 따라서 홀수(3, 5, 7)가 엄밀히 더 우수합니다. 프로덕션 클러스터는 거의 항상 3 또는 5를 운영합니다.

etcd 내부에서 쿠버네티스 객체는 `/registry/pods/default/my-pod` 같은 계층 키 아래에 protobuf 인코딩 값으로 저장됩니다. etcd와 직접 대화하는 클라이언트는 API 서버뿐이며, 그 외 모든 컴포넌트는 API 서버와 통신합니다. 이것이 **단일 진실 원천(single source of truth)** 원칙을 문자 그대로 구현한 것입니다.

### 5.1 키 구조

etcd는 쿠버네티스 오브젝트를 계층적 키 체계로 저장합니다:

```
/registry/<api-group>/<resource>/<namespace>/<name>
```

예시:
```
/registry/pods/default/nginx
/registry/deployments/kube-system/coredns
/registry/services/specs/default/kubernetes
/registry/clusterroles/cluster-admin
```

클러스터 범위 리소스는 네임스페이스 컴포넌트를 생략합니다:
```
/registry/nodes/worker-1
/registry/namespaces/production
```

### 5.2 직렬화 형식

기본적으로 오브젝트는 효율성을 위해 **프로토콜 버퍼(Protocol Buffers)** 형식으로 저장됩니다.
API 서버가 JSON(클라이언트 대면)과 protobuf(스토리지) 간의 변환을 처리합니다.

### 5.3 리소스 버전(Resource Versions)

etcd의 모든 오브젝트에는 etcd **수정 리비전(Modified Revision)**에 매핑되는
`resourceVersion` 필드가 있습니다. 이는 다음에 사용됩니다:

- **낙관적 동시성(Optimistic Concurrency)**: 업데이트에 현재 `resourceVersion`을 포함해야 함;
  충돌 시 `409 Conflict` 반환
- **감시 북마크(Watch Bookmarks)**: 클라이언트가 특정 리비전부터 감시를 재개
- **목록 페이지네이션(List Pagination)**: `continue` 토큰이 리비전 정보를 인코딩

```bash
# resourceVersion 확인
kubectl get pod nginx -o jsonpath='{.metadata.resourceVersion}'

# 충돌하는 업데이트 시도 (다른 사람이 수정한 경우 실패)
kubectl get pod nginx -o yaml > pod.yaml
# pod.yaml 편집 후:
kubectl replace -f pod.yaml
```

### 5.4 압축과 조각 모음(Compaction and Defragmentation)

etcd는 모든 리비전의 이력을 유지합니다. 시간이 지나면 디스크 공간을 소비합니다.

```bash
# etcd 데이터베이스 크기 확인
ETCDCTL_API=3 etcdctl \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/var/lib/minikube/certs/etcd/ca.crt \
  --cert=/var/lib/minikube/certs/etcd/server.crt \
  --key=/var/lib/minikube/certs/etcd/server.key \
  endpoint status --write-out=table
```

API 서버는 자동으로 오래된 리비전을 압축합니다 (기본값: 5분).

---

## 6. 쿠버네티스 오브젝트 모델 (GVR / GVK)

### 6.1 그룹, 버전, 리소스 (GVR)

쿠버네티스의 모든 REST 엔드포인트는 GVR로 식별됩니다:

| 구성 요소 | 예시 | 설명 |
|-----------|---------|-------------|
| 그룹(Group) | `apps` | API 그룹 (코어의 경우 빈 문자열) |
| 버전(Version) | `v1` | API 버전 |
| 리소스(Resource) | `deployments` | 복수형 리소스 이름 |

REST 경로: `/apis/{group}/{version}/namespaces/{ns}/{resource}/{name}`

코어 그룹은 `/api/v1/...`을 사용합니다 (경로에 그룹 없음).

### 6.2 그룹, 버전, 종류 (GVK)

GVK는 오브젝트의 **Go 타입**을 식별합니다:

| 구성 요소 | 예시 |
|-----------|---------|
| 그룹(Group) | `apps` |
| 버전(Version) | `v1` |
| 종류(Kind) | `Deployment` |

GVK와 GVR 간의 매핑은 **REST 매퍼(REST Mapper)**가 관리합니다.

```bash
# 모든 API 리소스 검색 (GVR + Kind)
kubectl api-resources

# 특정 리소스에 대한 상세 정보
kubectl api-resources | grep -i deployment

# OpenAPI 스키마 탐색
kubectl get --raw /openapi/v2 | python3 -m json.tool | head -50
```

### 6.3 GVR을 사용한 Go 클라이언트 작성

```go
package main

import (
	"context"
	"fmt"
	"os"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/clientcmd"
)

func main() {
	// Build config from kubeconfig
	config, err := clientcmd.BuildConfigFromFlags("", os.Getenv("KUBECONFIG"))
	if err != nil {
		panic(err)
	}

	// Create dynamic client
	client, err := dynamic.NewForConfig(config)
	if err != nil {
		panic(err)
	}

	// Define the GVR for Deployments
	gvr := schema.GroupVersionResource{
		Group:    "apps",
		Version:  "v1",
		Resource: "deployments",
	}

	// List deployments in the default namespace
	deployments, err := client.Resource(gvr).Namespace("default").List(
		context.TODO(),
		metav1.ListOptions{},
	)
	if err != nil {
		panic(err)
	}

	for _, d := range deployments.Items {
		fmt.Printf("Deployment: %s (replicas: %v)\n",
			d.GetName(),
			d.Object["spec"].(map[string]interface{})["replicas"],
		)
	}
}
```

### 6.4 커스텀 리소스(Custom Resources)

커스텀 리소스 정의(Custom Resource Definitions, CRDs)는 새로운 GVR/GVK 쌍으로 API를 확장합니다:

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: certificates.cert-manager.io
spec:
  group: cert-manager.io
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
              properties:
                secretName:
                  type: string
                issuerRef:
                  type: object
                  properties:
                    name:
                      type: string
                    kind:
                      type: string
  scope: Namespaced
  names:
    plural: certificates
    singular: certificate
    kind: Certificate
    shortNames:
      - cert
```

---

## 7. 인증과 인가 흐름

### 7.1 인증 (AuthN)

API 서버는 순서대로 평가되는 여러 인증 전략을 지원합니다:

| 방법 | 메커니즘 | 일반적 사용 |
|--------|-----------|-----------|
| X.509 클라이언트 인증서 | TLS 클라이언트 인증서 | kubeadm 클러스터 |
| 베어러 토큰(Bearer Tokens) | `Authorization: Bearer <token>` | 서비스 어카운트 |
| OIDC 토큰 | ID 공급자의 JWT | 인간 사용자 (SSO) |
| 웹훅 토큰 리뷰(Webhook Token Review) | 외부 인증 서비스 | 커스텀 통합 |
| 부트스트랩 토큰(Bootstrap Tokens) | 단기 토큰 | 노드 부트스트래핑 |

```bash
# 현재 신원 확인
kubectl auth whoami

# kubeconfig 인증 정보 확인
kubectl config view --minify -o jsonpath='{.users[0]}'
```

### 7.2 서비스 어카운트 토큰(Service Account Tokens)

모든 파드는 `/var/run/secrets/kubernetes.io/serviceaccount/token`에
마운트된 프로젝티드 서비스 어카운트 토큰을 받습니다.

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: my-app
  namespace: default
automountServiceAccountToken: true
---
apiVersion: v1
kind: Pod
metadata:
  name: my-app-pod
spec:
  serviceAccountName: my-app
  containers:
    - name: app
      image: curlimages/curl:8.5.0
      command:
        - sh
        - -c
        - |
          TOKEN=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)
          curl -s -k -H "Authorization: Bearer $TOKEN" \
            https://kubernetes.default.svc/api/v1/namespaces/default/pods
```

### 7.3 인가 (AuthZ)

인증 후, API 서버는 인가를 확인합니다. 기본 모드는
**RBAC**(역할 기반 접근 제어, Role-Based Access Control)입니다.

RBAC 오브젝트:
- **롤(Role)** / **클러스터롤(ClusterRole)**: 권한 정의 (리소스에 대한 동사)
- **롤바인딩(RoleBinding)** / **클러스터롤바인딩(ClusterRoleBinding)**: 주체에 롤 바인딩

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: production
  name: pod-reader
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "watch", "list"]
  - apiGroups: [""]
    resources: ["pods/log"]
    verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-pods
  namespace: production
subjects:
  - kind: ServiceAccount
    name: monitoring-agent
    namespace: production
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
```

```bash
# 사용자가 특정 작업을 수행할 수 있는지 확인
kubectl auth can-i create deployments --namespace=production

# 특정 서비스 어카운트로 확인
kubectl auth can-i get pods \
  --as=system:serviceaccount:production:monitoring-agent \
  --namespace=production

# 네임스페이스의 모든 롤 나열
kubectl get roles -n production

# ClusterRole 설명
kubectl describe clusterrole cluster-admin
```

### 7.4 어드미션 제어(Admission Control)

인가 후, 요청은 어드미션 컨트롤러를 거칩니다:

**변형 어드미션(Mutating Admission)** (오브젝트 수정):
- `MutatingAdmissionWebhook`
- `DefaultStorageClass`
- `PodPreset` (더 이상 사용하지 않음)

**검증 어드미션(Validating Admission)** (오브젝트 거부):
- `ValidatingAdmissionWebhook`
- `ValidatingAdmissionPolicy` (CEL 기반, v1.30 GA)
- `ResourceQuota`
- `LimitRanger`

```yaml
# 예시: ValidatingAdmissionPolicy (웹훅 불필요)
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
metadata:
  name: require-labels
spec:
  failurePolicy: Fail
  matchConstraints:
    resourceRules:
      - apiGroups: ["apps"]
        apiVersions: ["v1"]
        operations: ["CREATE", "UPDATE"]
        resources: ["deployments"]
  validations:
    - expression: "has(object.metadata.labels) && 'app' in object.metadata.labels"
      message: "All deployments must have an 'app' label"
```

---

## 8. 스케줄러 알고리즘

### 이론: 스케줄러 — 2단계 옵티마이저

`nodeName`이 설정되지 않은 Pod를 만들면, 그 Pod는 etcd에 "Pending" 상태로 들어갑니다. 스케줄러는 미스케줄된 파드를 워치하다가, 각각에 대해 2단계 알고리즘을 실행합니다:

**1단계 — 필터링(Predicates).** 스케줄러는 각 노드를 일련의 하드 제약(hard constraint)에 대해 평가합니다:
- 노드에 파드의 요청을 만족할 만큼 CPU와 메모리가 있는가?
- 파드가 노드의 테인트(taint)를 톨러레이트(tolerate)하는가?
- 파드의 노드 셀렉터/어피니티(affinity)가 이 노드의 레이블과 매치되는가?
- 파드가 클레임한 볼륨이 실제로 이 노드의 영역(zone)에서 attach 가능한가?

조건 하나라도 만족하지 못하는 노드는 제거됩니다. 필터링을 통과한 노드가 0개라면 파드는 Pending 상태로 남고, 스케줄러는 `FailedScheduling` 이벤트를 기록합니다.

**2단계 — 스코어링(Priorities).** 살아남은 노드들은 소프트 선호(soft preference) 기준으로 순위가 매겨집니다:
- `LeastAllocated`: 여유 자원이 많은 노드를 선호(spread).
- `BalancedResourceAllocation`: CPU와 메모리 사용률이 균형 잡힌 노드를 선호.
- `ImageLocality`: 해당 컨테이너 이미지가 이미 캐시된 노드를 선호.
- `InterPodAffinity`: 파드 어피니티 규칙을 만족하는 노드를 선호(예: 캐시 파드를 앱 파드 근처에 배치).

각 스코어러는 0–100을 반환하고, 스케줄러는 이를 가중 합산하여 가장 높은 점수의 노드를 고릅니다. 결정은 **바인딩(binding)**으로 커밋됩니다 — 단일 API 호출(`POST /pods/{name}/binding`)로 파드 객체에 `spec.nodeName`을 기록합니다. 그 노드의 kubelet은 워치 이벤트를 보고 컨테이너를 시작합니다.

높은 우선순위 파드가 스케줄될 수 없으면 **선점(preemption)**이 작동합니다 — 스케줄러가 더 낮은 우선순위 파드를 찾아 축출하여 자리를 만듭니다. 이것이 프로덕션에서 파드 우선순위 클래스가 중요한 이유입니다 — 경합 시 누가 축출되는지를 결정합니다.

### 8.1 개요

스케줄러는 두 단계로 파드를 노드에 할당합니다:

1. **필터링(Filtering)** (프레디케이트): 파드를 실행할 수 없는 노드를 제거
2. **스코어링(Scoring)** (우선순위): 남은 노드를 순위 매기고 최적의 것을 선택

```
Unscheduled Pod
      │
      ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Filtering  │───▶│   Scoring    │───▶│  Binding    │
│ (eliminate) │    │ (rank 0-100) │    │ (assign)    │
└─────────────┘    └──────────────┘    └─────────────┘
  N nodes            M ≤ N nodes          1 node
```

### 8.2 필터링 플러그인

| 플러그인 | 확인 내용 |
|--------|---------------|
| NodeResourcesFit | CPU/메모리 요청이 가용 용량에 맞는지 |
| NodeAffinity | 노드 셀렉터 및 어피니티 규칙이 일치하는지 |
| TaintToleration | 파드가 노드 테인트(Taint)를 허용하는지 |
| PodTopologySpread | 토폴로지 제약 조건이 충족 가능한지 |
| VolumeBinding | 필요한 PV가 이 노드에서 바인딩 가능한지 |
| InterPodAffinity | 파드 어피니티/안티어피니티 제약 조건 |
| NodePorts | 요청된 호스트 포트가 사용 가능한지 |
| NodeUnschedulable | 노드가 차단(Cordon)되지 않았는지 |

### 8.3 스코어링 플러그인

| 플러그인 | 전략 |
|--------|----------|
| NodeResourcesBalancedAllocation | 균형 잡힌 CPU/메모리 사용량 선호 |
| NodeResourcesFit (LeastAllocated) | 여유 리소스가 더 많은 노드 선호 |
| InterPodAffinity | 파드 어피니티 선호도 기반 점수 |
| TaintToleration | 허용되지 않은 테인트가 적은 노드 선호 |
| ImageLocality | 이미 컨테이너 이미지가 있는 노드 선호 |
| PodTopologySpread | 토폴로지 도메인 간 고른 분배 선호 |

각 플러그인은 0~100의 점수를 반환합니다. 스케줄러는 가중 점수의 합을 구하고
총합이 가장 높은 노드를 선택합니다.

### 8.4 스케줄러 프로파일

```yaml
apiVersion: kubescheduler.config.k8s.io/v1
kind: KubeSchedulerConfiguration
profiles:
  - schedulerName: default-scheduler
    plugins:
      score:
        enabled:
          - name: NodeResourcesFit
            weight: 2
          - name: PodTopologySpread
            weight: 3
        disabled:
          - name: ImageLocality
      filter:
        enabled:
          - name: NodeResourcesFit
          - name: TaintToleration
```

### 8.5 스케줄링에 영향 주기

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-workload
spec:
  # 노드 셀렉터 (단순)
  nodeSelector:
    accelerator: nvidia-a100

  # 노드 어피니티 (표현력 있는)
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: topology.kubernetes.io/zone
                operator: In
                values: ["us-east-1a", "us-east-1b"]
      preferredDuringSchedulingIgnoredDuringExecution:
        - weight: 80
          preference:
            matchExpressions:
              - key: node.kubernetes.io/instance-type
                operator: In
                values: ["p4d.24xlarge"]

    # 파드 안티어피니티
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        - labelSelector:
            matchLabels:
              app: gpu-workload
          topologyKey: kubernetes.io/hostname

  # 톨러레이션(Tolerations)
  tolerations:
    - key: "nvidia.com/gpu"
      operator: "Exists"
      effect: "NoSchedule"

  containers:
    - name: trainer
      image: pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
      resources:
        requests:
          cpu: "4"
          memory: "16Gi"
          nvidia.com/gpu: "1"
        limits:
          nvidia.com/gpu: "1"
```

---

## 9. 컨트롤 루프와 조정

### 이론: 폴링이 아닌 워치(Watch) — 컨트롤러들이 직접 통신하지 않고 협력하는 방법

순진하게 생각하면, 컨트롤러와 kubelet이 끊임없이 API 서버를 폴링할 것이라 예상할 수 있습니다 — "내가 처리할 새 파드가 있어?" 그러나 이런 방식은 수천 개 노드와 수백만 개 객체로 확장되지 않습니다. 대신 모든 장기 실행 쿠버네티스 컴포넌트는 단일 HTTP 연결을 열고 **워치(watch)** 요청을 보냅니다:

```
GET /api/v1/pods?watch=1&resourceVersion=12345
```

API 서버는 이 연결을 계속 열어두고, `resourceVersion=12345` 이후 변경이 발생할 때마다 **델타(delta)** 이벤트(`ADDED`, `MODIFIED`, `DELETED`)를 스트리밍합니다. 클라이언트는 로컬 캐시(**인포머**)를 유지하며 스트림으로부터 이를 갱신합니다. 이 결과:

- 모든 노드의 kubelet은 "나에게 할당된 파드" 실시간 미러를 가지고, 할당 후 밀리초 단위로 반응합니다.
- Deployment 컨트롤러는 "모든 Deployment와 그 자식 ReplicaSet"의 미러를 가지고, 레플리카 수가 바뀌는 순간 조정을 트리거합니다.
- API 서버를 폴링하는 일은 드뭅니다. 대부분 일회성 **list**(전체 스냅샷) 후 무기한 워치를 유지하는 패턴입니다.

워치가 끊어지면(네트워크 일시 장애 등), 클라이언트는 마지막으로 알고 있던 `resourceVersion`부터 다시 list합니다. 이것이 클러스터가 컨트롤 플레인 일시 장애를 견디는 방식입니다 — 컴포넌트들은 etcd의 진실(truth)로부터 자신의 뷰를 재구성하고 재시작 없이 작업을 계속합니다.

### 이론: 컨트롤러 — 레벨 트리거 조정 루프

`kube-controller-manager`의 모든 컨트롤러(Deployment, ReplicaSet, Node, Job, Endpoints, ...)는 동일한 루프를 구현합니다:

```
loop forever:
    desired = informer 캐시에서 읽기 (API 서버 / etcd 미러)
    actual  = 세상을 관찰 (또는 status 필드 읽기)
    if desired != actual:
        수렴하도록 액션 (객체 생성/수정/삭제)
    else:
        아무것도 하지 않음
```

이 모델을 견고하게 만드는 두 속성:

- **레벨 트리거(level-triggered), 엣지 트리거 아님.** 컨트롤러는 "이벤트를 봤으니 한 번 반응할게"가 아니라 "현재 상태가 X, 원하는 상태가 Y, X = Y가 되도록 행동할게"라고 말합니다. 컨트롤러가 크래시 후 재시작해도 상태를 다시 읽고 이어서 진행합니다 — 놓친 이벤트를 복구할 필요가 없습니다.
- **멱등(idempotent) 액션.** "ReplicaSet R이 없으면 생성" 요청은 두 번째 호출에서 효과가 없습니다. 따라서 컨트롤러의 액션이 중복되더라도(재시도, 재시작, 워크 큐 분할 등으로) 시스템은 동일한 상태로 수렴합니다.

`controller-manager`는 보통 모든 내장 컨트롤러를 단일 프로세스에서 실행하지만, etcd의 리더 선출 lease 덕분에 **한 번에 하나의 레플리카만 활성**입니다. 대기 중인 레플리카는 lease가 만료되면 인계받습니다. 커스텀 컨트롤러(Operator, 11강)도 controller-runtime 라이브러리를 통해 동일한 패턴을 사용합니다.

이 루프는 *바로 그* 쿠버네티스 패러다임입니다. Deployment, 오토스케일러, 인그레스 컨트롤러, cert-manager, ArgoCD — 모두 서로 다른 desired-state 스키마에 동일한 알고리즘을 적용한 것입니다.

### 9.1 조정 패턴(Reconciliation Pattern)

모든 쿠버네티스 컨트롤러는 동일한 패턴을 따릅니다:

```
          ┌──────────┐
          │  Observe  │  (API 서버에서 현재 상태 읽기)
          └────┬─────┘
               │
          ┌────▼─────┐
          │   Diff    │  (원하는 상태 vs. 실제 상태 비교)
          └────┬─────┘
               │
          ┌────▼─────┐
          │   Act     │  (수렴을 위한 조치 수행)
          └────┬─────┘
               │
               └───────── (루프)
```

이 패턴은 **레벨 트리거(Level-Triggered)**입니다: 컨트롤러는 모든 이벤트를 볼 필요가
없습니다. 무엇을 할지 결정하려면 현재 상태만 필요합니다.

### 9.2 인포머와 작업 큐(Informers and Work Queues)

컨트롤러는 API 서버를 폴링하지 않습니다. 대신 **인포머(Informers)**를 사용합니다:

1. 인포머가 리소스 타입에 대해 `List` 후 `Watch`를 수행
2. 수신된 오브젝트가 **로컬 캐시**(스레드 안전 저장소)에 저장
3. 이벤트가 키(`namespace/name`)를 큐에 넣는 **이벤트 핸들러** 트리거
4. **워커 고루틴(Worker Goroutine)**이 키를 디큐하고 `Reconcile` 함수를 호출
5. `Reconcile`이 캐시에서 원하는 상태를 읽고 조치를 취함

```go
package main

import (
	"context"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/workqueue"
)

func main() {
	config, _ := clientcmd.BuildConfigFromFlags("", "/path/to/kubeconfig")
	clientset, _ := kubernetes.NewForConfig(config)

	// Create a shared informer factory (resync every 30s)
	factory := informers.NewSharedInformerFactory(clientset, 30*time.Second)
	podInformer := factory.Core().V1().Pods().Informer()

	// Work queue
	queue := workqueue.NewRateLimitingQueue(
		workqueue.DefaultControllerRateLimiter(),
	)

	// Event handlers enqueue keys
	podInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			key, _ := cache.MetaNamespaceKeyFunc(obj)
			queue.Add(key)
			fmt.Printf("Pod added: %s\n", key)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			key, _ := cache.MetaNamespaceKeyFunc(newObj)
			queue.Add(key)
		},
		DeleteFunc: func(obj interface{}) {
			key, _ := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
			queue.Add(key)
			fmt.Printf("Pod deleted: %s\n", key)
		},
	})

	// Start the informer
	ctx := context.Background()
	factory.Start(ctx.Done())
	factory.WaitForCacheSync(ctx.Done())

	// Worker loop
	go wait.Until(func() {
		for {
			key, shutdown := queue.Get()
			if shutdown {
				return
			}
			// Reconcile logic goes here
			fmt.Printf("Reconciling: %s\n", key)
			queue.Done(key)
		}
	}, time.Second, ctx.Done())

	// Run until interrupted
	select {}
}
```

### 9.3 예시: 디플로이먼트 컨트롤러 조정

디플로이먼트(Deployment)가 업데이트되면 (예: 이미지 변경):

1. 디플로이먼트 컨트롤러가 업데이트된 디플로이먼트를 감지
2. 새 파드 템플릿으로 새 레플리카셋(ReplicaSet) 생성
3. 새 레플리카셋을 스케일업하고 이전 것을 스케일다운
4. 레플리카셋 컨트롤러가 필요에 따라 파드를 생성/삭제
5. 스케줄러가 새 파드를 노드에 할당
6. Kubelet이 이미지를 풀하고 컨테이너를 시작

각 단계는 자체 리소스에서 작동하는 별도의 컨트롤러이지만,
조율된 롤아웃(Rollout)으로 구성됩니다.

### 9.4 소유자 참조와 가비지 컬렉션(Owner References and Garbage Collection)

쿠버네티스는 `ownerReferences`를 통해 오브젝트 소유권을 추적합니다:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-deploy-7d4b8c6f5-abc12
  ownerReferences:
    - apiVersion: apps/v1
      kind: ReplicaSet
      name: my-deploy-7d4b8c6f5
      uid: a1b2c3d4-e5f6-7890-abcd-ef1234567890
      controller: true
      blockOwnerDeletion: true
```

부모 오브젝트가 삭제되면, **가비지 컬렉터(Garbage Collector)** 컨트롤러가
모든 종속 오브젝트를 삭제합니다 (연쇄 삭제). 이 동작을 제어할 수 있습니다:

```bash
# 전경 삭제 (종속 항목이 먼저 삭제될 때까지 대기)
kubectl delete deployment my-deploy --cascade=foreground

# 종속 항목 고아화 (레플리카셋과 파드는 계속 실행)
kubectl delete deployment my-deploy --cascade=orphan

# 백그라운드 삭제 (기본값: 부모를 즉시 삭제, GC가 정리)
kubectl delete deployment my-deploy
```

### 9.5 리더 선출(Leader Election)

HA 구성에서는 컨트롤러의 인스턴스 중 하나만 활성이어야 합니다. 쿠버네티스는
리더 선출을 위해 리스(Lease) 오브젝트를 사용합니다:

```bash
# 스케줄러 리스 보유자 확인
kubectl get lease -n kube-system kube-scheduler -o yaml

# 컨트롤러 매니저 리스 확인
kubectl get lease -n kube-system kube-controller-manager -o yaml
```

---

## 연습문제

### 연습문제 1: API 요청 추적

간단한 파드를 생성하고 `kubectl -v=8`을 사용하여 전체 API 요청을 추적합니다.
생성 작업의 HTTP 메서드, URL 경로, 응답 코드를 식별하세요.

```bash
# 상세 출력으로 이 파드 생성
cat <<EOF > /tmp/trace-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: trace-test
spec:
  containers:
    - name: nginx
      image: nginx:1.25
EOF
```

<details>
<summary>정답 보기</summary>

```bash
kubectl apply -f /tmp/trace-pod.yaml -v=8

# 다음과 유사한 라인을 찾으세요:
# I0115 POST https://192.168.49.2:8443/api/v1/namespaces/default/pods 201 Created

# 핵심 관찰 사항:
# - 메서드: POST (새 리소스 생성)
# - 경로: /api/v1/namespaces/default/pods (코어 그룹, v1, pods 리소스)
# - 응답: 201 Created (오브젝트가 etcd에 영속화됨)
# - Content-Type: application/json (kubectl은 기본적으로 JSON 전송)
# - 응답 본문에는 uid, resourceVersion, creationTimestamp 등
#   서버가 할당한 필드를 포함한 전체 파드 스펙이 포함됩니다

# 정리
kubectl delete pod trace-test
```

</details>

### 연습문제 2: etcd 키 탐색

`kubectl get --raw`을 사용하여 API를 탐색하고 다양한 리소스가 어떻게 구성되는지
이해합니다. 모든 API 그룹을 나열하고 디플로이먼트의 GVR을 찾으세요.

<details>
<summary>정답 보기</summary>

```bash
# 모든 API 그룹 나열
kubectl get --raw /apis | python3 -c "
import json, sys
data = json.load(sys.stdin)
for g in data['groups']:
    print(f\"{g['name']:40s} preferred: {g['preferredVersion']['groupVersion']}\")
"

# apps/v1 그룹의 리소스 나열
kubectl get --raw /apis/apps/v1 | python3 -c "
import json, sys
data = json.load(sys.stdin)
for r in data['resources']:
    if '/' not in r['name']:  # skip subresources
        print(f\"  {r['name']:30s} kind={r['kind']:20s} namespaced={r['namespaced']}\")
"

# 디플로이먼트의 GVR은:
# Group:    apps
# Version:  v1
# Resource: deployments
# REST 경로: /apis/apps/v1/namespaces/{ns}/deployments/{name}

# 특정 디플로이먼트를 가져와서 확인
kubectl get --raw /apis/apps/v1/namespaces/kube-system/deployments/coredns | python3 -m json.tool | head -20
```

</details>

### 연습문제 3: RBAC 구성

`default` 네임스페이스에서 파드와 파드 로그만 읽을 수 있는 서비스어카운트(ServiceAccount)를
생성합니다. `kubectl auth can-i`를 사용하여 권한을 확인하세요.

<details>
<summary>정답 보기</summary>

```yaml
# /tmp/rbac-exercise.yaml로 저장
apiVersion: v1
kind: ServiceAccount
metadata:
  name: pod-log-reader
  namespace: default
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-log-reader-role
  namespace: default
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["pods/log"]
    verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: pod-log-reader-binding
  namespace: default
subjects:
  - kind: ServiceAccount
    name: pod-log-reader
    namespace: default
roleRef:
  kind: Role
  name: pod-log-reader-role
  apiGroup: rbac.authorization.k8s.io
```

```bash
kubectl apply -f /tmp/rbac-exercise.yaml

# 권한 테스트
kubectl auth can-i get pods \
  --as=system:serviceaccount:default:pod-log-reader
# yes

kubectl auth can-i get pods/log \
  --as=system:serviceaccount:default:pod-log-reader
# yes

kubectl auth can-i create pods \
  --as=system:serviceaccount:default:pod-log-reader
# no

kubectl auth can-i get deployments \
  --as=system:serviceaccount:default:pod-log-reader
# no

kubectl auth can-i get pods \
  --as=system:serviceaccount:default:pod-log-reader \
  --namespace=kube-system
# no (롤이 default 네임스페이스로 제한됨)
```

</details>

### 연습문제 4: 스케줄러 관찰

특정 노드 어피니티(Node Affinity)와 리소스 요청을 가진 파드를 생성합니다.
이벤트(Events)를 검사하여 스케줄러의 결정을 관찰하세요.

<details>
<summary>정답 보기</summary>

```yaml
# /tmp/scheduler-exercise.yaml로 저장
apiVersion: v1
kind: Pod
metadata:
  name: scheduler-test
spec:
  affinity:
    nodeAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
        - weight: 100
          preference:
            matchExpressions:
              - key: kubernetes.io/os
                operator: In
                values: ["linux"]
  containers:
    - name: nginx
      image: nginx:1.25
      resources:
        requests:
          cpu: "100m"
          memory: "128Mi"
        limits:
          cpu: "200m"
          memory: "256Mi"
```

```bash
kubectl apply -f /tmp/scheduler-exercise.yaml

# 스케줄러 이벤트 확인
kubectl describe pod scheduler-test | grep -A 5 "Events:"

# 다음과 같은 이벤트를 볼 수 있습니다:
# Type    Reason     Age   From               Message
# ----    ------     ---   ----               -------
# Normal  Scheduled  10s   default-scheduler  Successfully assigned default/scheduler-test to minikube
# Normal  Pulling    9s    kubelet            Pulling image "nginx:1.25"

# 선택된 노드 확인
kubectl get pod scheduler-test -o jsonpath='{.spec.nodeName}'

# 이 특정 파드에 대한 스케줄러 로그 확인
kubectl -n kube-system logs kube-scheduler-minikube --tail=20 | grep scheduler-test

# 정리
kubectl delete pod scheduler-test
```

</details>

### 연습문제 5: 컨트롤러 조정

3개의 레플리카를 가진 레플리카셋(ReplicaSet)을 생성한 다음, 파드 하나를 수동으로 삭제합니다.
원하는 수를 유지하기 위해 레플리카셋 컨트롤러가 파드를 재생성하는 것을 관찰하세요.

<details>
<summary>정답 보기</summary>

```yaml
# /tmp/reconcile-exercise.yaml로 저장
apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: reconcile-test
spec:
  replicas: 3
  selector:
    matchLabels:
      app: reconcile-demo
  template:
    metadata:
      labels:
        app: reconcile-demo
    spec:
      containers:
        - name: nginx
          image: nginx:1.25
```

```bash
kubectl apply -f /tmp/reconcile-exercise.yaml

# 모든 파드가 실행 중일 때까지 대기
kubectl get pods -l app=reconcile-demo -w

# 파드 이름 확인
kubectl get pods -l app=reconcile-demo -o name

# 파드 하나 삭제
POD=$(kubectl get pods -l app=reconcile-demo -o jsonpath='{.items[0].metadata.name}')
kubectl delete pod $POD

# 즉시 확인: 레플리카셋 컨트롤러가 대체 파드를 생성합니다
kubectl get pods -l app=reconcile-demo

# 레플리카셋 이벤트 확인
kubectl describe rs reconcile-test | grep -A 10 "Events:"

# 다음을 볼 수 있습니다:
# Normal  SuccessfulCreate  Created pod: reconcile-test-xxxxx (원본)
# Normal  SuccessfulCreate  Created pod: reconcile-test-yyyyy (대체)

# ownerReferences가 RS가 파드를 소유하고 있음을 보여줍니다
kubectl get pod -l app=reconcile-demo -o jsonpath='{.items[0].metadata.ownerReferences[0].kind}'
# ReplicaSet

# 정리
kubectl delete rs reconcile-test
```

</details>

---

**이전**: [개요](./00_Overview.md) | **다음**: [워크로드 리소스](./02_Workload_Resources.md)

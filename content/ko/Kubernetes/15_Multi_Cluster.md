# 15. 멀티 클러스터(Multi-Cluster)

**이전**: [옵저버빌리티](./14_Observability.md) | **다음**: [Kubernetes API 프로그래밍](./16_Kubernetes_API_Programming.md)

## 학습 목표

이 레슨을 완료하면 다음을 수행할 수 있습니다:

1. 멀티 클러스터 아키텍처(복제형, 페더레이션, 허브-스포크)를 비교하고 적절한 모델 선택
2. Kubernetes Federation v2를 사용하여 클러스터 간 워크로드 배포 및 관리
3. Submariner와 멀티 클러스터 서비스 디스커버리(Service Discovery)를 통한 크로스 클러스터 네트워킹 구성
4. Istio 멀티 클러스터 구성을 활용한 멀티 클러스터 서비스 메시(Service Mesh) 설정
5. ArgoCD ApplicationSets를 사용한 멀티 클러스터 환경의 GitOps 구현

---

단일 Kubernetes 클러스터에는 확실한 한계가 있습니다 -- etcd 성능은 약 5,000노드를 초과하면 저하되고, 컨트롤 플레인(Control Plane) 장애의 영향 범위가 모든 워크로드에 미치며, 규정 요건에 따라 지리적 데이터 상주가 요구될 수 있습니다. 멀티 클러스터 아키텍처는 워크로드를 독립적인 클러스터에 분산하여 이러한 제약을 해결합니다. 하지만 멀티 클러스터는 자체적인 복잡성을 도입합니다: 서비스 디스커버리, 크로스 클러스터 네트워킹, 일관된 구성, 통합 옵저버빌리티. 이 레슨에서는 멀티 클러스터 규모에서 Kubernetes를 운영하기 위한 아키텍처, 도구, 패턴을 다룹니다.

## 목차

- [이론과 원리](#이론과-원리)
- [1. 멀티 클러스터 아키텍처](#1-멀티-클러스터-아키텍처)
- [2. Kubernetes Federation v2](#2-kubernetes-federation-v2)
- [3. 멀티 클러스터 서비스 디스커버리](#3-멀티-클러스터-서비스-디스커버리)
- [4. Submariner를 활용한 크로스 클러스터 네트워킹](#4-submariner를-활용한-크로스-클러스터-네트워킹)
- [5. Liqo를 활용한 클러스터 공유](#5-liqo를-활용한-클러스터-공유)
- [6. 멀티 클러스터 서비스 메시 (Istio)](#6-멀티-클러스터-서비스-메시-istio)
- [7. 멀티 클러스터 GitOps (ArgoCD ApplicationSets)](#7-멀티-클러스터-gitops-argocd-applicationsets)
- [8. 멀티 클러스터 보안](#8-멀티-클러스터-보안)
- [연습문제](#연습문제)

---

## 1. 멀티 클러스터 아키텍처

### 이론: 클러스터가 늘어나는 이유

조직을 한 클러스터 너머로 미는 네 압력:

- **스케일 천장.** etcd 성능은 약 5,000 노드와 약 150,000 파드를 넘어가면 저하됩니다. 공식 "지원" 최댓값은 잘 문서화되어 있지만, 실무에서 많은 팀은 예측 가능한 업그레이드와 disaster-recovery 특성을 위해 1,000–2,000 노드에서 샤드합니다. 그 천장을 넘으면 워크로드를 여러 클러스터에 분할합니다.
- **Blast radius.** 잘못 구성된 어드미션 웹훅, 손상된 etcd, 나쁜 클러스터 업그레이드가 클러스터의 모든 워크로드를 다운시킬 수 있습니다. "prod"를 3개 지역 클러스터로 분할하면 사고가 사용자의 1/3에 영향을 미치지, 전체가 아닙니다. 이는 가장 중요한 비-스케일 이유입니다.
- **지리적 지연.** Tokyo의 사용자는 us-east-1에서 서비스되어서는 안 됩니다. 다중 지역 애플리케이션은 지역당 클러스터(또는 클러스터 집합)를 필요로 하며, 트래픽은 지연이나 출처로 라우팅됩니다.
- **규제와 데이터 거주성.** EU 고객 데이터는 EU에 머물러야 합니다 — 일부 관할은 로컬 컨트롤 플레인을 요구합니다. 대륙을 가로지르는 단일 클러스터는 운영적으로 어색하며 불법일 수 있습니다.

흔한 경로 — 한 클러스터로 시작, 업그레이드가 잘못되어 blast radius 고통을 겪고, "prod" + "staging" + "dev"로 분할. 그다음 지연을 위해 prod를 지역별로 분할. 그다음 공유 서비스(로깅, 모니터링, 내부 도구)를 위한 hub 추가. 계획하기 전에 다섯 클러스터가 됩니다.

### 이론: 세 토폴로지 — Replicated, Federated, Hub-Spoke

클러스터 간 연결 모델은 세 가지 원형을 가집니다:

**Replicated (독립 클러스터).** 각 클러스터는 완전히 자율 — 동일한 워크로드를 여러 클러스터에 배포하고 엣지에서 트래픽을 라우팅(DNS, 글로벌 로드 밸런서). 클러스터 간 상태 없음, 클러스터 간 컨트롤 플레인 없음. 이는 **가장 단순한** 모델이며 각 지역이 자체 사용자를 서비스하는 stateless 서비스에 동작합니다. 운영 고통 — 모든 config, 모든 시크릿, 모든 관측 가능성 대시보드의 N개 사본. GitOps(§D)로 완화.

**Federated (Federation v2 / KubeFed).** 중앙 호스트 클러스터가 *연합된* 리소스 버전(`FederatedDeployment`, `FederatedService`)을 보유 — 컨트롤러가 그것들을 멤버 클러스터로 투영합니다. 하나의 매니페스트를 작성하면 선택된 모든 클러스터에 배포됩니다. 장점 — 중앙화된 API, 동적 배치 정책(예: "70% us-east, 30% us-west"). 단점 — 호스트 클러스터가 단일 실패 지점이 됨; 투영 lag이 관찰 가능; "왜 전파되지 않았는가?" 디버깅은 자체 학문. Federation v2는 2026년 유지보수 모드 — 대부분의 프로덕션 팀은 GitOps + service mesh를 대신 사용합니다.

**Hub-Spoke.** 한 "hub" 클러스터가 공유 플랫폼 서비스(CI/CD 오케스트레이션, 관측 가능성 집계, 중앙 정책 강제)를 실행 — 워크로드 "spoke" 클러스터는 애플리케이션 워크로드만 실행. Hub는 작지만 중요 — spoke는 크지만 플랫폼 관점에서 stateless. 이 모델은 벤더가 hub를 제공하는 OpenShift / Rancher / Anthos / EKS Anywhere를 채택하는 엔터프라이즈에서 지배적입니다.

올바른 선택은 무엇을 공유하는지에 달려 있습니다 — 클러스터 간 아무것도 → replicated; 클러스터 간 동기화된 리소스 → federated; 클러스터 간 플랫폼 서비스 → hub-spoke.

### 1.1 왜 멀티 클러스터인가?

| 동인 | 단일 클러스터 문제 | 멀티 클러스터 솔루션 |
|---|---|---|
| 영향 범위(Blast Radius) | 컨트롤 플레인 장애 = 전체 워크로드 영향 | 독립적인 장애 도메인 |
| 확장 제한 | ~5,000노드에서 etcd 성능 저하 | 클러스터 간 부하 분산 |
| 규정 준수(Compliance) | 데이터가 특정 리전에 상주해야 함 | 리전별 클러스터 |
| 팀 격리 | 노이지 네이버(Noisy Neighbor), RBAC 복잡성 | 팀/테넌트별 전용 클러스터 |
| 가용성(Availability) | 단일 리전 장애 = 전체 다운타임 | 리전 간 액티브-액티브(Active-Active) |
| 업그레이드 안전성 | 클러스터 업그레이드가 모든 워크로드에 영향 | 클러스터 간 롤링 업그레이드 |

### 1.2 아키텍처 패턴

```
패턴 1: 복제형 (Replicated, Standalone)
==================================
각 클러스터가 동일한 배포로 독립적으로 실행.
로드 밸런서가 클러스터 간 트래픽 분산.

  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │  Cluster A    │    │  Cluster B    │    │  Cluster C    │
  │  (us-east)    │    │  (us-west)    │    │  (eu-west)    │
  │               │    │               │    │               │
  │  ┌──────────┐ │    │  ┌──────────┐ │    │  ┌──────────┐ │
  │  │ App v1.2 │ │    │  │ App v1.2 │ │    │  │ App v1.2 │ │
  │  └──────────┘ │    │  └──────────┘ │    │  └──────────┘ │
  └──────────────┘    └──────────────┘    └──────────────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            │
                    ┌───────▼───────┐
                    │  Global LB    │
                    │  (DNS/Anycast) │
                    └───────────────┘

패턴 2: 페더레이션 (Federated)
====================
컨트롤 플레인이 멤버 클러스터에 리소스를 분배.

                    ┌──────────────┐
                    │  Federation   │
                    │  Control Plane│
                    └───────┬──────┘
                            │ distribute
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
       ┌──────────┐  ┌──────────┐  ┌──────────┐
       │ Cluster A │  │ Cluster B │  │ Cluster C │
       └──────────┘  └──────────┘  └──────────┘

패턴 3: 허브-스포크 (Hub-Spoke)
====================
관리 클러스터(허브)가 워크로드 클러스터(스포크)를 제어.

              ┌─────────────────────┐
              │     Hub Cluster      │
              │  (management plane)  │
              │  - ArgoCD            │
              │  - Policy engine     │
              │  - Fleet management  │
              └──────────┬──────────┘
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │  Spoke 1  │  │  Spoke 2  │  │  Spoke 3  │
    │ (prod-us) │  │(prod-eu)  │  │ (staging) │
    └──────────┘  └──────────┘  └──────────┘
```

### 1.3 아키텍처 선택

| 패턴 | 복잡도 | 사용 사례 | 크로스 클러스터 통신 |
|---|---|---|---|
| 복제형(Replicated) | 낮음 | 독립적인 리전별 배포 | 외부 LB를 통해서만 |
| 페더레이션(Federated) | 높음 | 클러스터 간 통합 API | Federation API |
| 허브-스포크(Hub-Spoke) | 중간 | 중앙 관리, GitOps | 관리 플레인만 |

---

## 2. Kubernetes Federation v2

> **사용 중단 공지**: KubeFed(kubernetes-sigs/kubefed)는 **2023년에 아카이브**되었으며 더 이상 활발하게 유지 관리되지 않습니다. 새 프로젝트에는 사용하지 마세요. 멀티 클러스터 관리를 위해서는 아래 [섹션 2.3](#23-kubefed의-활성-대안)에 나열된 활성 대안을 선호하세요: **Cluster API(CAPI)**, **Open Cluster Management(OCM)**, 또는 **Rancher Fleet**.

### 2.1 KubeFed 아키텍처

KubeFed(Kubernetes Federation v2)는 여러 클러스터에 Kubernetes 리소스를 분배하기 위한 컨트롤 플레인을 제공합니다:

```
┌─────────────────────────────────────────────┐
│              Host Cluster                    │
│                                             │
│  ┌────────────────────────────────────────┐  │
│  │         KubeFed Controller             │  │
│  │                                        │  │
│  │  ┌──────────────┐  ┌───────────────┐   │  │
│  │  │ FederatedType │  │   Placement   │   │  │
│  │  │ Controller    │  │   Controller  │   │  │
│  │  └──────────────┘  └───────────────┘   │  │
│  │  ┌──────────────┐  ┌───────────────┐   │  │
│  │  │   Override    │  │   Scheduling  │   │  │
│  │  │  Controller   │  │  Controller   │   │  │
│  │  └──────────────┘  └───────────────┘   │  │
│  └────────────────────────────────────────┘  │
│                                             │
│  ┌────────────────────────────────────────┐  │
│  │  Federated Resources                   │  │
│  │  - FederatedDeployment                 │  │
│  │  - FederatedService                    │  │
│  │  - FederatedConfigMap                  │  │
│  │  - FederatedNamespace                  │  │
│  └────────────────────────────────────────┘  │
└──────────────┬──────────────┬───────────────┘
               │              │
         ┌─────▼─────┐  ┌────▼──────┐
         │ Member     │  │ Member    │
         │ Cluster A  │  │ Cluster B │
         └───────────┘  └───────────┘
```

### 2.2 설치

```bash
# 호스트 클러스터에 KubeFed 설치
helm repo add kubefed-charts https://raw.githubusercontent.com/kubernetes-sigs/kubefed/master/charts
helm install kubefed kubefed-charts/kubefed \
  --namespace kube-federation-system \
  --create-namespace

# 멤버 클러스터 조인
kubefedctl join cluster-a \
  --cluster-context=cluster-a-context \
  --host-cluster-context=host-cluster-context \
  --v=2

kubefedctl join cluster-b \
  --cluster-context=cluster-b-context \
  --host-cluster-context=host-cluster-context \
  --v=2

# 확인
kubectl get kubefedclusters -n kube-federation-system
```

### 2.3 페더레이션 디플로이먼트(Federated Deployment)

```yaml
apiVersion: types.kubefed.io/v1beta1
kind: FederatedDeployment
metadata:
  name: web-app
  namespace: production
spec:
  template:
    metadata:
      labels:
        app: web-app
    spec:
      replicas: 3
      selector:
        matchLabels:
          app: web-app
      template:
        metadata:
          labels:
            app: web-app
        spec:
          containers:
          - name: web-app
            image: example.com/web-app:v1.2
            ports:
            - containerPort: 8080
            resources:
              requests:
                cpu: 200m
                memory: 256Mi

  placement:
    clusters:
    - name: cluster-a
    - name: cluster-b
    # 또는 clusterSelector 사용:
    # clusterSelector:
    #   matchLabels:
    #     region: us

  overrides:
  - clusterName: cluster-a
    clusterOverrides:
    - path: "/spec/replicas"
      value: 5                    # 기본 클러스터에서 더 많은 레플리카
  - clusterName: cluster-b
    clusterOverrides:
    - path: "/spec/replicas"
      value: 2                    # 보조 클러스터에서 더 적은 레플리카
    - path: "/spec/template/spec/containers/0/image"
      value: "example.com/web-app:v1.1"  # 카나리: cluster-b에서 이전 버전
```

### 2.4 페더레이션 서비스(Federated Service)

```yaml
apiVersion: types.kubefed.io/v1beta1
kind: FederatedService
metadata:
  name: web-app
  namespace: production
spec:
  template:
    spec:
      selector:
        app: web-app
      ports:
      - port: 80
        targetPort: 8080
      type: ClusterIP
  placement:
    clusters:
    - name: cluster-a
    - name: cluster-b
```

### 2.5 페더레이션 네임스페이스(FederatedNamespace)

```yaml
# 네임스페이스를 먼저 페더레이션해야 합니다
apiVersion: types.kubefed.io/v1beta1
kind: FederatedNamespace
metadata:
  name: production
  namespace: production
spec:
  placement:
    clusters:
    - name: cluster-a
    - name: cluster-b
    - name: cluster-c
```

### 2.3 KubeFed의 활성 대안

KubeFed가 2023년에 아카이브되었으므로, 커뮤니티는 멀티 클러스터 관리를 위한 세 가지 활성 프로젝트로 수렴되었습니다:

#### Cluster API(CAPI)

Cluster API는 Kubernetes 스타일의 선언적 API를 사용하여 클러스터 생명주기 관리(프로비저닝, 업그레이드, 삭제)를 표준화합니다. 워크로드를 페더레이션하지는 않지만, 페더레이션 도구가 구축되는 인프라 기반을 제공합니다.

```bash
# clusterctl(CAPI CLI) 설치
curl -L https://github.com/kubernetes-sigs/cluster-api/releases/latest/download/clusterctl-linux-amd64 -o clusterctl
chmod +x clusterctl && mv clusterctl /usr/local/bin/

# AWS 공급자로 관리 클러스터 초기화
clusterctl init --infrastructure aws

# 클러스터 정의 생성 및 적용
clusterctl generate cluster my-cluster \
  --kubernetes-version v1.29.0 \
  --control-plane-machine-count=3 \
  --worker-machine-count=3 | kubectl apply -f -
```

주요 개념: `Cluster`, `Machine`, `MachineDeployment`, `ClusterClass`(토폴로지 기반 템플릿). 공급자: AWS, Azure, GCP, vSphere, OpenStack 등.

#### Open Cluster Management(OCM)

OCM(open-cluster-management.io)은 멀티 클러스터 거버넌스, 배치, 애드온 관리를 위한 허브-스포크 모델을 제공합니다. Red Hat Advanced Cluster Management(RHACM)의 기반입니다.

```bash
# OCM CLI 설치
brew install open-cluster-management/tap/clusteradm    # macOS
# 또는: curl -L https://raw.githubusercontent.com/open-cluster-management-io/clusteradm/main/install.sh | bash

# 허브 클러스터 초기화
clusteradm init --wait

# 관리형 클러스터 참가 (스포크 클러스터에서 실행)
clusteradm join --hub-token <token> --hub-apiserver <hub-url> --cluster-name cluster1

# 허브에서 참가 요청 승인
clusteradm accept --clusters cluster1
```

주요 개념: `ManagedCluster`, `ManagedClusterSet`, `Placement`, `ManifestWork`(스포크에 워크로드 푸시), `AddOn`(생명주기 관리 플러그인).

#### Rancher Fleet

Fleet(fleet.rancher.io)은 수천 개의 클러스터로 확장되는 Kubernetes용 GitOps 네이티브 지속적 배포 도구입니다. Rancher와 함께 번들로 제공되지만 독립적으로도 사용할 수 있습니다.

```bash
# Fleet 독립 설치 (Rancher 없이)
helm repo add fleet https://rancher.github.io/fleet-helm-charts/
helm install -n cattle-fleet-system --create-namespace fleet-crd fleet/fleet-crd
helm install -n cattle-fleet-system fleet fleet/fleet

# Git에서 배포하기 위한 GitRepo 리소스 생성
kubectl apply -f - <<EOF
apiVersion: fleet.cattle.io/v1alpha1
kind: GitRepo
metadata:
  name: my-app
  namespace: fleet-local
spec:
  repo: https://github.com/myorg/my-app
  branch: main
  targets:
    - clusterSelector:
        matchLabels:
          env: production
EOF
```

주요 개념: `GitRepo`, `Bundle`, `ClusterGroup`, `ClusterRegistrationToken`. Fleet은 허브-스포크 모델을 사용합니다: Fleet Manager 클러스터가 레이블 셀렉터를 기반으로 등록된 다운스트림 클러스터에 번들을 푸시합니다.

| 도구 | 주요 용도 | 워크로드 페더레이션 | 클러스터 생명주기 |
|------|-----------|--------------------|--------------------|
| Cluster API | 클러스터 프로비저닝 | 아니오 | 예 |
| OCM | 거버넌스 + 배치 | ManifestWork를 통해 | CAPI 애드온을 통해 |
| Rancher Fleet | GitOps 전달 | 예 | Rancher/CAPI를 통해 |
| (KubeFed — 아카이브) | 페더레이션 리소스 | 예 (사용 중단) | 아니오 |

---

## 3. 멀티 클러스터 서비스 디스커버리

### 이론: 클러스터 간 서비스 디스커버리 — DNS + Identity 문제

한 클러스터 내에서 Pod는 `redis.cache.svc.cluster.local`을 호출하고 CoreDNS가 해석합니다(3강 §D). 클러스터 간에는 이것이 깨집니다 — `cluster.local`은 클러스터별. 이를 동작하게 만드는 세 패턴:

**1. Multi-Cluster Services API (KEP-1645).** 표준 CRD `ServiceExport`가 Service를 export 가능으로 표시 — 각 클러스터의 컨트롤러가 그것을 `redis.cache.svc.clusterset.local` 같은 글로벌 DNS 이름 아래로 미러링. clusterset의 어떤 클러스터의 Pod든 해석하고 도달 가능. 구현 — AWS Cloud Map, GKE Multi-cluster Services, Submariner.

**2. Service Mesh Multi-Cluster (Istio, Linkerd, Cilium Cluster Mesh).** 각 클러스터의 사이드카나 eBPF 프로그램이 *모든* 피어 클러스터의 Service를 압니다. `redis.cache.svc.cluster.local`로의 호출이 투명하게 원격 클러스터의 Pod에 도달할 수 있습니다. 강한 식별(사이드카 간 mTLS)이 게이트 — 데이터 플레인이 연결성을 처리합니다. 운영적으로 무겁지만 가장 강력 — 클러스터를 가로지르는 트래픽 분할, 페일오버, 위치 인식 라우팅을 얻습니다.

**3. 클러스터 인식 DNS + 평면 L3 (Submariner).** Submariner가 클러스터 노드 간 암호화된 IPsec 터널을 빌드하여, 클러스터 A의 Pod가 클러스터 B의 Pod IP에서 직접 도달 가능(NAT 없음). 다중 클러스터 DNS 뷰(Lighthouse)와 결합하여, 클러스터 간 클러스터 내 경험을 얻습니다. Service mesh보다 가벼움 — mTLS나 L7 기능을 주지 않음.

근본 통찰 — 클러스터 간 연결성은 **단순한 네트워킹 문제가 아닙니다.** 식별(누가 호출자, 누가 callee), DNS(서로를 어떻게 찾는지), 신뢰(서로를 검증하는지) 모두 필요합니다. Service mesh는 셋 모두를 묶고, 다른 것들은 조각으로부터 합성합니다.

### 3.1 디스커버리 문제

단일 클러스터에서는 DNS 해석(`service-name.namespace.svc.cluster.local`)이 서비스 디스커버리를 처리합니다. 클러스터 간에는 각 클러스터가 자체 DNS와 네트워크 공간을 가지고 있기 때문에 이것이 작동하지 않습니다.

### 3.2 Kubernetes Multi-Cluster Services API (MCS)

MCS API(KEP-1645)는 클러스터 간 서비스를 내보내고 가져오는 표준화된 방법을 제공합니다:

```yaml
# Cluster A에서: 서비스 내보내기
apiVersion: multicluster.x-k8s.io/v1alpha1
kind: ServiceExport
metadata:
  name: web-app
  namespace: production
---
# Cluster B에서: 서비스가 다음으로 사용 가능:
# web-app.production.svc.clusterset.local
```

### 3.3 ClusterSet 아키텍처

```
┌──────────────────────────────────────────────────────────┐
│                     ClusterSet                            │
│                                                          │
│  ┌──────────────┐         ┌──────────────┐              │
│  │  Cluster A    │         │  Cluster B    │              │
│  │               │         │               │              │
│  │  Service:     │         │  Service:     │              │
│  │  web-app      │         │  web-app      │              │
│  │  (exported)   │         │  (exported)   │              │
│  └──────┬───────┘         └──────┬───────┘              │
│         │                        │                       │
│         └────────────┬───────────┘                       │
│                      │                                   │
│              ┌───────▼───────┐                           │
│              │ ServiceImport │                           │
│              │               │                           │
│              │ DNS:          │                           │
│              │ web-app.      │                           │
│              │ production.   │                           │
│              │ svc.clusterset│                           │
│              │ .local        │                           │
│              └───────────────┘                           │
└──────────────────────────────────────────────────────────┘
```

### 3.4 DNS 기반 서비스 디스커버리

더 간단한 설정에서는 외부 DNS를 사용하여 크로스 클러스터 서비스 디스커버리를 수행할 수 있습니다:

```yaml
# ExternalDNS로 Route53/CloudDNS에 서비스 등록
apiVersion: v1
kind: Service
metadata:
  name: web-app
  namespace: production
  annotations:
    external-dns.alpha.kubernetes.io/hostname: web-app.us-east.example.com
    external-dns.alpha.kubernetes.io/ttl: "60"
spec:
  type: LoadBalancer
  selector:
    app: web-app
  ports:
  - port: 80
    targetPort: 8080
```

```bash
# 가중치 라우팅을 사용한 글로벌 DNS 구성
# Route53 예시:
# web-app.example.com -> CNAME
#   - web-app.us-east.example.com (weight: 70)
#   - web-app.eu-west.example.com (weight: 30)
```

---

## 4. Submariner를 활용한 크로스 클러스터 네트워킹

### 4.1 Submariner란?

Submariner는 Kubernetes 클러스터 간에 보안 네트워크 터널을 생성하여, 클러스터 간 직접적인 파드-투-파드(Pod-to-Pod) 및 파드-투-서비스(Pod-to-Service) 통신을 가능하게 합니다. Globalnet 컴포넌트를 통해 중복되는 CIDR 범위를 처리합니다.

### 4.2 아키텍처

```
┌──────────────────────┐         ┌──────────────────────┐
│    Cluster A          │         │    Cluster B          │
│    10.244.0.0/16     │         │    10.245.0.0/16     │
│                      │         │                      │
│  ┌────────────────┐  │  IPsec  │  ┌────────────────┐  │
│  │   Gateway Node  │◀─────────▶│  │   Gateway Node  │  │
│  │   (submariner-  │  │ tunnel │  │   (submariner-  │  │
│  │    gateway)     │  │         │  │    gateway)     │  │
│  └────────────────┘  │         │  └────────────────┘  │
│                      │         │                      │
│  ┌────────────────┐  │         │  ┌────────────────┐  │
│  │ Route Agent    │  │         │  │ Route Agent    │  │
│  │ (all nodes)    │  │         │  │ (all nodes)    │  │
│  └────────────────┘  │         │  └────────────────┘  │
│                      │         │                      │
│  ┌────────────────┐  │         │  ┌────────────────┐  │
│  │ Lighthouse     │  │         │  │ Lighthouse     │  │
│  │ (DNS discovery)│  │         │  │ (DNS discovery)│  │
│  └────────────────┘  │         │  └────────────────┘  │
└──────────────────────┘         └──────────────────────┘
                │                         │
                └────────────┬────────────┘
                             │
                     ┌───────▼───────┐
                     │  Broker        │
                     │  (metadata     │
                     │   exchange)    │
                     └───────────────┘
```

### 4.3 설치

```bash
# subctl CLI 설치
curl -Ls https://get.submariner.io | bash
export PATH=$PATH:~/.local/bin

# 브로커 배포 (임의의 클러스터 또는 전용 브로커 클러스터에서)
subctl deploy-broker --kubeconfig broker-cluster.kubeconfig

# 클러스터를 브로커에 조인
subctl join --kubeconfig cluster-a.kubeconfig broker-info.subm \
  --clusterid cluster-a \
  --natt=false

subctl join --kubeconfig cluster-b.kubeconfig broker-info.subm \
  --clusterid cluster-b \
  --natt=false

# 연결 확인
subctl show all
subctl diagnose all
subctl verify --kubeconfig cluster-a.kubeconfig \
  --toconfig cluster-b.kubeconfig \
  --only connectivity
```

### 4.4 서비스 내보내기

```bash
# Cluster A에서 서비스 내보내기
subctl export service web-app -n production

# 또는 선언적으로
kubectl apply -f - <<EOF
apiVersion: multicluster.x-k8s.io/v1alpha1
kind: ServiceExport
metadata:
  name: web-app
  namespace: production
EOF

# Cluster B에서 다음으로 접근 가능:
# web-app.production.svc.clusterset.local
```

### 4.5 크로스 클러스터 연결 테스트

```bash
# Cluster B의 파드에서 Cluster A의 서비스에 접근
kubectl exec -it test-pod -- curl http://web-app.production.svc.clusterset.local

# DNS 해석 확인
kubectl exec -it test-pod -- nslookup web-app.production.svc.clusterset.local

# Submariner 연결 상태 확인
kubectl get clusters.submariner.io -n submariner-operator
kubectl get endpoints.submariner.io -n submariner-operator
kubectl get gateways.submariner.io -n submariner-operator
```

### 4.6 Globalnet (중복 CIDR)

클러스터들이 중복되는 파드 또는 서비스 CIDR을 가지고 있을 때, Submariner의 Globalnet 컴포넌트가 글로벌 가상 IP를 할당합니다:

```bash
# 조인 시 Globalnet 활성화
subctl join broker-info.subm \
  --clusterid cluster-a \
  --globalnet \
  --globalnet-cidr 242.0.0.0/16

# 각 클러스터에 고유한 글로벌 CIDR이 할당됨
# 내보낸 서비스와 파드에 글로벌 IP가 할당됨
kubectl get globalingressips -n production
```

---

## 5. Liqo를 활용한 클러스터 공유

### 5.1 Liqo란?

Liqo는 원격 클러스터를 나타내는 가상 노드(Virtual Node)를 생성하여 원활한 멀티 클러스터 리소스 공유를 가능하게 합니다. 가상 노드에 스케줄링된 파드는 투명하게 원격 클러스터로 오프로드됩니다.

### 5.2 아키텍처

```
┌──────────────────────────────────┐
│         Home Cluster              │
│                                   │
│  ┌─────────┐  ┌───────────────┐  │
│  │  Node 1  │  │  Virtual Node │  │
│  │  (real)  │  │  (cluster-b)  │──────▶ Pods run in Cluster B
│  └─────────┘  └───────────────┘  │
│  ┌─────────┐  ┌───────────────┐  │
│  │  Node 2  │  │  Virtual Node │  │
│  │  (real)  │  │  (cluster-c)  │──────▶ Pods run in Cluster C
│  └─────────┘  └───────────────┘  │
│                                   │
│  스케줄러가 가상 노드를 용량을 가진   │
│  일반 노드로 인식                   │
└──────────────────────────────────┘
```

### 5.3 설치 및 피어링(Peering)

```bash
# 양쪽 클러스터에 Liqo 설치
curl -sL https://get.liqo.io | bash

# 또는 Helm으로
helm repo add liqo https://helm.liqo.io
helm install liqo liqo/liqo \
  --namespace liqo-system \
  --create-namespace \
  --set controllerManager.config.enableResourceEnforcement=true

# Cluster B에서 피어링 명령 생성
liqoctl generate peer-command

# Cluster A에서 출력된 명령 실행하여 피어링 설정
liqoctl peer --remote-kubeconfig cluster-b.kubeconfig

# 피어링 확인
kubectl get foreignclusters
liqoctl status
```

### 5.4 워크로드 오프로딩

```yaml
# 네임스페이스 오프로딩 활성화
apiVersion: offloading.liqo.io/v1beta1
kind: NamespaceOffloading
metadata:
  name: offloading
  namespace: production
spec:
  namespaceMappingStrategy: EnforceSameName
  podOffloadingStrategy: LocalAndRemote  # 또는 Remote
  clusterSelector:
    nodeSelectorTerms:
    - matchExpressions:
      - key: liqo.io/remote-cluster-id
        operator: In
        values:
        - cluster-b
        - cluster-c
```

```yaml
# 표준 노드 어피니티 또는 Liqo 스케줄러를 사용하여 파드를 오프로드
apiVersion: apps/v1
kind: Deployment
metadata:
  name: distributed-app
  namespace: production
spec:
  replicas: 6
  selector:
    matchLabels:
      app: distributed-app
  template:
    metadata:
      labels:
        app: distributed-app
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: liqo.io/type
                operator: In
                values:
                - virtual-node  # 원격 클러스터에만 스케줄링
      containers:
      - name: app
        image: example.com/app:v1
```

### 5.5 리소스 공유 구성

```bash
# 리소스 공유 쿼터 구성
kubectl annotate foreigncluster cluster-b \
  liqo.io/cpu-sharing-percentage="50" \
  liqo.io/memory-sharing-percentage="50"

# 원격 클러스터에서 사용 가능한 리소스 확인
kubectl describe node liqo-cluster-b
# Capacity:
#   cpu:     8     (원격 클러스터 16 CPU의 50%)
#   memory:  16Gi  (원격 클러스터 32Gi의 50%)
```

---

## 6. 멀티 클러스터 서비스 메시 (Istio)

### 6.1 Istio 멀티 클러스터 모델

Istio는 여러 멀티 클러스터 배포 모델을 지원합니다:

```
모델 1: 멀티 프라이머리 (Multi-Primary, 각 클러스터에 자체 컨트롤 플레인)
===============================================================

  ┌──────────────────┐     ┌──────────────────┐
  │  Cluster A        │     │  Cluster B        │
  │  ┌──────────────┐ │     │  ┌──────────────┐ │
  │  │  istiod      │◀──────▶│  istiod      │ │
  │  │  (primary)   │ │ sync│  │  (primary)   │ │
  │  └──────────────┘ │     │  └──────────────┘ │
  │  ┌──────────────┐ │     │  ┌──────────────┐ │
  │  │  east-west   │◀──────▶│  east-west   │ │
  │  │  gateway     │ │ data│  │  gateway     │ │
  │  └──────────────┘ │     │  └──────────────┘ │
  └──────────────────┘     └──────────────────┘

모델 2: 프라이머리-리모트 (Primary-Remote, 하나의 컨트롤 플레인이 여러 클러스터 관리)
=================================================================

  ┌──────────────────┐     ┌──────────────────┐
  │  Cluster A        │     │  Cluster B        │
  │  ┌──────────────┐ │     │  (istiod 없음)     │
  │  │  istiod      │──────▶│                  │
  │  │  (primary)   │ │ push│  ┌──────────────┐ │
  │  └──────────────┘ │ config│  east-west   │ │
  │  ┌──────────────┐ │     │  │  gateway     │ │
  │  │  east-west   │◀──────▶│              │ │
  │  │  gateway     │ │ data│  └──────────────┘ │
  │  └──────────────┘ │     └──────────────────┘
  └──────────────────┘
```

### 6.2 멀티 프라이머리 설정

```bash
# 전제 조건: 클러스터 간 공유 루트 CA
# 공유 루트 인증서 생성
mkdir -p certs
cd certs

# 루트 CA 생성
make -f istio-1.20.0/tools/certs/Makefile.selfsigned.mk root-ca

# 각 클러스터에 대한 중간 CA 생성
make -f istio-1.20.0/tools/certs/Makefile.selfsigned.mk cluster-a-cacerts
make -f istio-1.20.0/tools/certs/Makefile.selfsigned.mk cluster-b-cacerts

# 각 클러스터에 인증서를 시크릿으로 설치
kubectl --context=cluster-a create namespace istio-system
kubectl --context=cluster-a create secret generic cacerts -n istio-system \
  --from-file=cluster-a/ca-cert.pem \
  --from-file=cluster-a/ca-key.pem \
  --from-file=cluster-a/root-cert.pem \
  --from-file=cluster-a/cert-chain.pem

kubectl --context=cluster-b create namespace istio-system
kubectl --context=cluster-b create secret generic cacerts -n istio-system \
  --from-file=cluster-b/ca-cert.pem \
  --from-file=cluster-b/ca-key.pem \
  --from-file=cluster-b/root-cert.pem \
  --from-file=cluster-b/cert-chain.pem
```

### 6.3 각 클러스터에 Istio 설치

```yaml
# Cluster A IstioOperator
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  name: istio-cluster-a
spec:
  profile: default
  values:
    global:
      meshID: mesh1
      multiCluster:
        clusterName: cluster-a
      network: network-a
  meshConfig:
    defaultConfig:
      proxyMetadata:
        ISTIO_META_DNS_CAPTURE: "true"
        ISTIO_META_DNS_AUTO_ALLOCATE: "true"
  components:
    ingressGateways:
    - name: istio-eastwestgateway
      label:
        istio: eastwestgateway
        app: istio-eastwestgateway
        topology.istio.io/network: network-a
      enabled: true
      k8s:
        env:
        - name: ISTIO_META_REQUESTED_NETWORK_VIEW
          value: network-a
        service:
          ports:
          - name: status-port
            port: 15021
            targetPort: 15021
          - name: tls
            port: 15443
            targetPort: 15443
          - name: tls-istiod
            port: 15012
            targetPort: 15012
          - name: tls-webhook
            port: 15017
            targetPort: 15017
```

```bash
# 양쪽 클러스터에 설치
istioctl install --context=cluster-a -f cluster-a-operator.yaml
istioctl install --context=cluster-b -f cluster-b-operator.yaml

# 이스트-웨스트 게이트웨이를 통해 서비스 노출
kubectl --context=cluster-a apply -f samples/multicluster/expose-services.yaml
kubectl --context=cluster-b apply -f samples/multicluster/expose-services.yaml

# 리모트 시크릿 교환 (각 클러스터가 상대 클러스터를 알아야 함)
istioctl create-remote-secret --context=cluster-a --name=cluster-a | \
  kubectl apply -f - --context=cluster-b

istioctl create-remote-secret --context=cluster-b --name=cluster-b | \
  kubectl apply -f - --context=cluster-a

# 멀티 클러스터 확인
istioctl remote-clusters --context=cluster-a
```

### 6.4 크로스 클러스터 트래픽 관리

```yaml
# 로컬리티 기반 로드 밸런싱을 위한 DestinationRule
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: web-app-dr
  namespace: production
spec:
  host: web-app.production.svc.cluster.local
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 10s
      baseEjectionTime: 30s
    loadBalancer:
      localityLbSetting:
        enabled: true
        distribute:
        - from: "us-east/*"
          to:
            "us-east/*": 80
            "us-west/*": 20
        - from: "us-west/*"
          to:
            "us-west/*": 80
            "us-east/*": 20
        failover:
        - from: us-east
          to: us-west
        - from: us-west
          to: us-east
```

### 6.5 멀티 클러스터 VirtualService

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: web-app-vs
  namespace: production
spec:
  hosts:
  - web-app.production.svc.cluster.local
  http:
  - match:
    - headers:
        x-region:
          exact: us-east
    route:
    - destination:
        host: web-app.production.svc.cluster.local
        subset: cluster-a
      weight: 100
  - route:
    - destination:
        host: web-app.production.svc.cluster.local
      weight: 100  # 기본값: 로컬리티 기반 라우팅
```

---

## 7. 멀티 클러스터 GitOps (ArgoCD ApplicationSets)

### 이론: 멀티 클러스터를 위한 GitOps — Pull 모델이 Hub로부터 당신을 구합니다

GitOps 모델(Argo CD, Flux)에서, 각 클러스터는 Git 리포지토리에서 자신의 desired 상태를 pull하는 에이전트를 실행합니다. Git repo가 진실의 원천 — 에이전트가 매치되도록 클러스터를 조정합니다.

멀티 클러스터에 대해 이는 아름답게 스케일됩니다:

- **하나의 repo, 여러 클러스터.** `clusters/` 디렉토리가 클러스터당 하나의 하위 디렉토리를 가짐 — 각 에이전트는 자기 디렉토리만 pull. 클러스터 추가 = 디렉토리 추가 + 에이전트 부트스트랩.
- **실패할 중앙 컨트롤 플레인 없음.** Hub 클러스터가 다운되어도, spoke 에이전트는 독립적으로 HA인 Git에 대해 계속 조정합니다. 이는 Federation v2의 push 모델에 대한 근본적 이점입니다.
- **Argo CD ApplicationSet**이 템플릿 + 생성기(클러스터 목록, Git 디렉토리, pull request)로부터 많은 클러스터의 Argo `Application` 리소스를 생성. 하나의 템플릿, N 클러스터, 자동 멤버십 추적.

멘탈 모델 — GitOps는 "중앙 컨트롤러가 spoke에 config를 푸시"(Federation v2) 패턴을 "spoke가 공유 소스로부터 config를 pull"(Argo)로 대체합니다. 같은 최종 상태, 매우 다른 실패 모드 — pull 모델은 Git 서버 너머의 중앙 단일 실패 지점이 없습니다.

배포뿐 아니라 클러스터 간 *연결성*이 필요한 워크로드에 대해, GitOps는 service mesh와 합성됩니다 — GitOps가 메시 + 워크로드를 각 클러스터에 배포 — 메시가 클러스터 간 트래픽을 처리합니다.

### 7.1 ArgoCD 멀티 클러스터 아키텍처

```
┌─────────────────────────────────────────┐
│          Management Cluster              │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │         ArgoCD                   │    │
│  │                                  │    │
│  │  ┌──────────────────────────┐   │    │
│  │  │   ApplicationSet         │   │    │
│  │  │   Controller             │   │    │
│  │  │                          │   │    │
│  │  │   Generators:            │   │    │
│  │  │   - List                 │   │    │
│  │  │   - Cluster              │   │    │
│  │  │   - Git                  │   │    │
│  │  │   - Matrix               │   │    │
│  │  └──────────┬───────────────┘   │    │
│  │             │ generates          │    │
│  │             ▼                    │    │
│  │  ┌────┐ ┌────┐ ┌────┐ ┌────┐   │    │
│  │  │App │ │App │ │App │ │App │   │    │
│  │  │ 1  │ │ 2  │ │ 3  │ │ 4  │   │    │
│  │  └──┬─┘ └──┬─┘ └──┬─┘ └──┬─┘   │    │
│  └─────┼──────┼──────┼──────┼─────┘    │
│        │      │      │      │           │
└────────┼──────┼──────┼──────┼───────────┘
         │      │      │      │
    ┌────▼──┐ ┌─▼────┐ │  ┌───▼───┐
    │Prod-US│ │Prod-EU│ │  │Staging│
    └───────┘ └──────┘  │  └───────┘
                   ┌────▼───┐
                   │Prod-AP │
                   └────────┘
```

### 7.2 ArgoCD에 클러스터 등록

```bash
# ArgoCD에 대상 클러스터 등록
argocd cluster add cluster-a-context --name prod-us
argocd cluster add cluster-b-context --name prod-eu
argocd cluster add cluster-c-context --name staging

# 등록된 클러스터 확인
argocd cluster list

# 또는 Secret으로 선언적 등록
kubectl apply -f - <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: prod-us-cluster
  namespace: argocd
  labels:
    argocd.argoproj.io/secret-type: cluster
    environment: production
    region: us-east
type: Opaque
stringData:
  name: prod-us
  server: https://prod-us.example.com:6443
  config: |
    {
      "bearerToken": "<token>",
      "tlsClientConfig": {
        "insecure": false,
        "caData": "<base64-ca>"
      }
    }
EOF
```

### 7.3 클러스터 제너레이터를 사용한 ApplicationSet

```yaml
# 모든 프로덕션 클러스터에 동일한 앱 배포
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: web-app
  namespace: argocd
spec:
  generators:
  - clusters:
      selector:
        matchLabels:
          environment: production
      values:
        revision: main
  template:
    metadata:
      name: 'web-app-{{name}}'
    spec:
      project: default
      source:
        repoURL: https://github.com/example/web-app.git
        targetRevision: '{{values.revision}}'
        path: k8s/overlays/{{metadata.labels.region}}
      destination:
        server: '{{server}}'
        namespace: production
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
        syncOptions:
        - CreateNamespace=true
        retry:
          limit: 5
          backoff:
            duration: 5s
            factor: 2
            maxDuration: 3m
```

### 7.4 Git 제너레이터를 사용한 ApplicationSet

```yaml
# Git의 디렉토리 구조에서 애플리케이션 생성
# 레포지토리 구조:
# clusters/
#   prod-us/
#     config.json    {"cluster": "prod-us", "region": "us-east", "env": "prod"}
#   prod-eu/
#     config.json    {"cluster": "prod-eu", "region": "eu-west", "env": "prod"}
#   staging/
#     config.json    {"cluster": "staging", "region": "us-east", "env": "staging"}

apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: cluster-configs
  namespace: argocd
spec:
  generators:
  - git:
      repoURL: https://github.com/example/cluster-configs.git
      revision: main
      directories:
      - path: clusters/*
      - path: clusters/experimental-*
        exclude: true
  template:
    metadata:
      name: '{{path.basename}}-config'
    spec:
      project: default
      source:
        repoURL: https://github.com/example/cluster-configs.git
        targetRevision: main
        path: '{{path}}'
      destination:
        server: 'https://{{path.basename}}.example.com:6443'
        namespace: kube-system
```

### 7.5 매트릭스 제너레이터를 사용한 ApplicationSet

```yaml
# 매트릭스: 클러스터 x 애플리케이션 (데카르트 곱)
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: platform-services
  namespace: argocd
spec:
  generators:
  - matrix:
      generators:
      # 첫 번째 차원: 클러스터
      - clusters:
          selector:
            matchLabels:
              environment: production
      # 두 번째 차원: 애플리케이션
      - list:
          elements:
          - app: monitoring
            chart: kube-prometheus-stack
            repoURL: https://prometheus-community.github.io/helm-charts
            version: "55.0.0"
          - app: logging
            chart: loki-stack
            repoURL: https://grafana.github.io/helm-charts
            version: "2.10.0"
          - app: ingress
            chart: ingress-nginx
            repoURL: https://kubernetes.github.io/ingress-nginx
            version: "4.9.0"
  template:
    metadata:
      name: '{{app}}-{{name}}'
    spec:
      project: platform
      source:
        repoURL: '{{repoURL}}'
        chart: '{{chart}}'
        targetRevision: '{{version}}'
        helm:
          valueFiles:
          - values/{{metadata.labels.region}}.yaml
      destination:
        server: '{{server}}'
        namespace: '{{app}}'
      syncPolicy:
        automated:
          selfHeal: true
        syncOptions:
        - CreateNamespace=true
```

### 7.6 클러스터 간 점진적 롤아웃

```yaml
# 롤링 동기화 전략을 사용한 ApplicationSet
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: web-app-progressive
  namespace: argocd
spec:
  generators:
  - list:
      elements:
      - cluster: staging
        server: https://staging.example.com:6443
        order: "1"
      - cluster: prod-us
        server: https://prod-us.example.com:6443
        order: "2"
      - cluster: prod-eu
        server: https://prod-eu.example.com:6443
        order: "3"
  strategy:
    type: RollingSync
    rollingSync:
      steps:
      - matchExpressions:
        - key: order
          operator: In
          values: ["1"]    # 스테이징에 먼저 배포
      - matchExpressions:
        - key: order
          operator: In
          values: ["2"]    # 그 다음 prod-us
        maxUpdate: 1
      - matchExpressions:
        - key: order
          operator: In
          values: ["3"]    # 마지막으로 prod-eu
        maxUpdate: 1
  template:
    metadata:
      name: 'web-app-{{cluster}}'
      labels:
        order: '{{order}}'
    spec:
      project: default
      source:
        repoURL: https://github.com/example/web-app.git
        path: k8s/base
        targetRevision: main
      destination:
        server: '{{server}}'
        namespace: production
```

---

## 8. 멀티 클러스터 보안

### 8.1 아이덴티티와 신뢰(Trust)

크로스 클러스터 통신에는 공유 신뢰 도메인(Trust Domain)이 필요합니다. 옵션:

```
옵션 1: 공유 루트 CA
========================
    ┌────────────┐
    │  Root CA    │
    └──────┬─────┘
     ┌─────┼─────┐
     ▼     ▼     ▼
  ┌────┐ ┌────┐ ┌────┐
  │Int │ │Int │ │Int │
  │CA-A│ │CA-B│ │CA-C│
  └────┘ └────┘ └────┘
    │      │      │
  Cluster Cluster Cluster
    A      B      C

옵션 2: SPIFFE/SPIRE
=======================
각 클러스터가 SPIRE 에이전트를 실행.
SPIFFE ID: spiffe://trust-domain/ns/production/sa/web-app
페더레이션을 통한 크로스 클러스터 신뢰.
```

### 8.2 멀티 클러스터용 네트워크 정책(Network Policy)

```yaml
# 알려진 원격 클러스터에서의 트래픽만 허용
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-cross-cluster
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: web-app
  policyTypes:
  - Ingress
  ingress:
  # 로컬 네임스페이스에서 허용
  - from:
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: production
  # Submariner 게이트웨이 CIDR에서 허용
  - from:
    - ipBlock:
        cidr: 242.0.0.0/16  # Globalnet CIDR
  # Istio 이스트-웨스트 게이트웨이에서 허용
  - from:
    - namespaceSelector:
        matchLabels:
          istio: system
      podSelector:
        matchLabels:
          istio: eastwestgateway
```

### 8.3 멀티 클러스터 관리를 위한 RBAC

```yaml
# 클러스터 범위 권한이 있는 ArgoCD 프로젝트
apiVersion: argoproj.io/v1alpha1
kind: AppProject
metadata:
  name: platform
  namespace: argocd
spec:
  description: Platform services across all clusters
  sourceRepos:
  - 'https://github.com/example/*'
  destinations:
  - namespace: monitoring
    server: '*'  # 등록된 모든 클러스터
  - namespace: logging
    server: '*'
  - namespace: ingress
    server: '*'
  clusterResourceWhitelist:
  - group: ''
    kind: Namespace
  - group: rbac.authorization.k8s.io
    kind: ClusterRole
  - group: rbac.authorization.k8s.io
    kind: ClusterRoleBinding
  namespaceResourceWhitelist:
  - group: '*'
    kind: '*'
  roles:
  - name: platform-admin
    description: Platform team admin
    policies:
    - p, proj:platform:platform-admin, applications, *, platform/*, allow
    groups:
    - platform-team
```

### 8.4 클러스터 간 시크릿(Secret) 관리

```yaml
# 중앙 시크릿 스토어와 External Secrets Operator 사용
apiVersion: external-secrets.io/v1beta1
kind: ClusterSecretStore
metadata:
  name: aws-secrets-manager
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets-sa
            namespace: external-secrets
---
# ArgoCD를 통해 모든 클러스터에 배포되는 동일한 ExternalSecret
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: database-credentials
  namespace: production
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: ClusterSecretStore
  target:
    name: db-credentials
  data:
  - secretKey: username
    remoteRef:
      key: production/database
      property: username
  - secretKey: password
    remoteRef:
      key: production/database
      property: password
```

---

## 연습문제

### 연습문제 1: 멀티 클러스터 아키텍처 설계

여러분의 회사는 북미, 유럽, 아시아태평양 지역의 고객에게 서비스하는 전자상거래 플랫폼을 운영합니다. 요구사항: (a) 데이터 상주 -- EU 고객 데이터는 EU에 있어야 함, (b) 지연시간 -- 100ms 미만 응답 시간, (c) 가용성 -- 전체 리전 장애에도 서비스 유지, (d) 비용 -- 크로스 리전 트래픽 최소화. 멀티 클러스터 아키텍처를 설계하세요: 패턴(복제형/페더레이션/허브-스포크)을 선택하고, 클러스터 토폴로지를 설명하고, 트래픽 라우팅 방법을 설명하고, 각 관심사(서비스 디스커버리, 네트워킹, 배포)에 사용할 도구를 지정하세요.

<details>
<summary>정답 보기</summary>

**아키텍처: 허브-스포크 관리를 갖춘 복제형(Replicated with Hub-Spoke Management)**

```
                     ┌──────────────────────┐
                     │  Management Cluster   │
                     │  (us-east)            │
                     │  - ArgoCD             │
                     │  - Monitoring (Thanos) │
                     │  - Policy (Kyverno)    │
                     └──────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  prod-us      │  │  prod-eu      │  │  prod-ap      │
    │  (us-east)    │  │  (eu-west)    │  │  (ap-south)   │
    │               │  │               │  │               │
    │  Full app     │  │  Full app     │  │  Full app     │
    │  stack        │  │  stack        │  │  stack        │
    │  US database  │  │  EU database  │  │  AP database  │
    └──────────────┘  └──────────────┘  └──────────────┘
```

**클러스터 토폴로지:** 총 4개 클러스터. 1개 관리 허브 + 3개 리전 워크로드 클러스터. 각 워크로드 클러스터는 데이터 상주 규정 준수를 위해 자체 데이터베이스를 포함한 전체 애플리케이션 스택을 실행합니다.

**트래픽 라우팅:** 글로벌 DNS(Route53 지연시간 기반 라우팅 또는 Cloudflare)가 사용자를 가장 가까운 리전 클러스터로 안내합니다. 각 리전 내에서는 Istio 서비스 메시가 로컬리티 기반 로드 밸런싱으로 트래픽 관리를 처리합니다. 크로스 리전 페일오버의 경우, DNS 헬스 체크가 비정상 리전을 제거하고 다음으로 가까운 리전으로 리다이렉트합니다.

**도구 선택:**
- 배포: 허브의 ArgoCD with ApplicationSets (클러스터 제너레이터)
- 서비스 디스커버리: 리전 내 -- Kubernetes DNS. 크로스 리전 -- Route53 지연시간 기반 DNS
- 네트워킹: 각 클러스터가 독립적 (크로스 클러스터 파드 네트워킹 불필요). 크로스 리전 통신은 공용 TLS 엔드포인트를 통한 API 호출 사용
- 데이터 동기화: 비EU 데이터를 위한 데이터베이스 복제 (예: CockroachDB 멀티 리전 또는 PostgreSQL 논리적 복제). EU 데이터는 eu-west에만 유지
- 옵저버빌리티: 글로벌 Prometheus 페더레이션을 위한 Thanos. 멀티 테넌트 수집이 가능한 Loki
- 시크릿 관리: 리전별 AWS Secrets Manager 인스턴스와 External Secrets Operator

</details>

### 연습문제 2: Submariner 크로스 클러스터 설정

Submariner를 사용하여 두 클러스터를 연결하는 전체 절차를 작성하세요: (a) 파드 CIDR 10.244.0.0/16, 서비스 CIDR 10.96.0.0/12인 Cluster A, (b) 파드 CIDR 10.244.0.0/16 (중복), 서비스 CIDR 10.96.0.0/12인 Cluster B. 포함 항목: 브로커 배포, Globalnet을 활성화하여 두 클러스터 조인, Cluster A에서 서비스 내보내기, Cluster B에서 접근하는 명령어. 검증 명령어도 제공하세요.

<details>
<summary>정답 보기</summary>

```bash
# 단계 1: subctl 설치
curl -Ls https://get.submariner.io | bash
export PATH=$PATH:~/.local/bin

# 단계 2: Cluster A에 브로커 배포 (또는 전용 브로커 클러스터)
subctl deploy-broker --kubeconfig ~/.kube/cluster-a.kubeconfig

# broker-info.subm 파일이 연결 세부사항과 함께 생성됨

# 단계 3: Globalnet으로 Cluster A 조인 (CIDR이 중복되므로 Globalnet 필수)
subctl join --kubeconfig ~/.kube/cluster-a.kubeconfig \
  broker-info.subm \
  --clusterid cluster-a \
  --globalnet \
  --globalnet-cidr 242.1.0.0/16 \
  --cable-driver libreswan \
  --natt=false

# 단계 4: Globalnet으로 Cluster B 조인
subctl join --kubeconfig ~/.kube/cluster-b.kubeconfig \
  broker-info.subm \
  --clusterid cluster-b \
  --globalnet \
  --globalnet-cidr 242.2.0.0/16 \
  --cable-driver libreswan \
  --natt=false

# 단계 5: 연결 확인
subctl show all --kubeconfig ~/.kube/cluster-a.kubeconfig
subctl diagnose all --kubeconfig ~/.kube/cluster-a.kubeconfig
subctl verify --kubeconfig ~/.kube/cluster-a.kubeconfig \
  --toconfig ~/.kube/cluster-b.kubeconfig \
  --only connectivity,service-discovery

# 단계 6: Cluster A에서 서비스 내보내기
kubectl --kubeconfig ~/.kube/cluster-a.kubeconfig \
  apply -f - <<EOF
apiVersion: multicluster.x-k8s.io/v1alpha1
kind: ServiceExport
metadata:
  name: database
  namespace: production
EOF

# 단계 7: Cluster B에서 접근
kubectl --kubeconfig ~/.kube/cluster-b.kubeconfig run test \
  --image=busybox --rm -it --restart=Never -- \
  nslookup database.production.svc.clusterset.local

kubectl --kubeconfig ~/.kube/cluster-b.kubeconfig run test \
  --image=busybox --rm -it --restart=Never -- \
  wget -qO- http://database.production.svc.clusterset.local:5432

# 단계 8: Cluster B에서 ServiceImport 생성 확인
kubectl --kubeconfig ~/.kube/cluster-b.kubeconfig \
  get serviceimports -n production

# 단계 9: Globalnet IP 할당 확인
kubectl --kubeconfig ~/.kube/cluster-a.kubeconfig \
  get globalingressips -n production

kubectl --kubeconfig ~/.kube/cluster-b.kubeconfig \
  get globalingressips -n production

# 단계 10: 게이트웨이 상태 확인
kubectl --kubeconfig ~/.kube/cluster-a.kubeconfig \
  get gateways.submariner.io -n submariner-operator -o wide
kubectl --kubeconfig ~/.kube/cluster-a.kubeconfig \
  get clusters.submariner.io -n submariner-operator
```

</details>

### 연습문제 3: ArgoCD ApplicationSet

다음 조건의 ArgoCD ApplicationSet을 작성하세요: (a) `environment: production` 레이블이 있는 모든 클러스터에 `payment-service` 배포, (b) 리전별로 다른 Helm 값 파일 사용 (values/us-east.yaml, values/eu-west.yaml), (c) 스테이징에 먼저 배포하고 동기화를 기다린 후 프로덕션 클러스터에 하나씩 롤아웃, (d) 자동 동기화 정책(self-heal 및 prune 포함). 전체 ApplicationSet YAML과 Git 레포지토리 구조를 작성하세요.

<details>
<summary>정답 보기</summary>

Git 레포지토리 구조:

```
payment-service/
├── Chart.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
├── values.yaml              # 기본 값
└── values/
    ├── staging.yaml          # 스테이징 오버라이드
    ├── us-east.yaml          # US 프로덕션 오버라이드
    ├── eu-west.yaml          # EU 프로덕션 오버라이드
    └── ap-south.yaml         # AP 프로덕션 오버라이드
```

```yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: payment-service
  namespace: argocd
spec:
  generators:
  - list:
      elements:
      - cluster: staging
        server: https://staging.example.com:6443
        region: staging
        order: "1"
        environment: staging
      - cluster: prod-us
        server: https://prod-us.example.com:6443
        region: us-east
        order: "2"
        environment: production
      - cluster: prod-eu
        server: https://prod-eu.example.com:6443
        region: eu-west
        order: "3"
        environment: production
      - cluster: prod-ap
        server: https://prod-ap.example.com:6443
        region: ap-south
        order: "4"
        environment: production
  strategy:
    type: RollingSync
    rollingSync:
      steps:
      # 단계 1: 스테이징에 배포
      - matchExpressions:
        - key: order
          operator: In
          values: ["1"]
      # 단계 2: 첫 번째 프로덕션 클러스터 (US)
      - matchExpressions:
        - key: order
          operator: In
          values: ["2"]
        maxUpdate: 1
      # 단계 3: 나머지 프로덕션 클러스터 하나씩
      - matchExpressions:
        - key: order
          operator: In
          values: ["3", "4"]
        maxUpdate: 1
  template:
    metadata:
      name: 'payment-service-{{cluster}}'
      labels:
        order: '{{order}}'
        environment: '{{environment}}'
    spec:
      project: default
      source:
        repoURL: https://github.com/example/payment-service.git
        targetRevision: main
        path: .
        helm:
          valueFiles:
          - values.yaml
          - values/{{region}}.yaml
      destination:
        server: '{{server}}'
        namespace: payment
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
        syncOptions:
        - CreateNamespace=true
        - PrunePropagationPolicy=foreground
        retry:
          limit: 5
          backoff:
            duration: 10s
            factor: 2
            maxDuration: 5m
```

</details>

### 연습문제 4: 멀티 클러스터 Istio 서비스 메시

서로 다른 네트워크에 있는 두 클러스터에서 Istio 멀티 프라이머리를 설정하는 단계를 설명하세요. 작성 항목: (a) 양쪽 클러스터의 IstioOperator 구성, (b) 리모트 시크릿 생성 및 교환 명령어, (c) 페일오버가 있는 로컬리티 기반 로드 밸런싱을 구현하는 DestinationRule (80% 로컬, 20% 리모트, 로컬이 비정상이면 전체 페일오버), (d) 헤더 기반 트래픽 라우팅 VirtualService (x-canary: true는 cluster-b로만 라우팅).

<details>
<summary>정답 보기</summary>

**(a) Cluster A용 IstioOperator:**

```yaml
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  name: istio-cluster-a
spec:
  profile: default
  values:
    global:
      meshID: shared-mesh
      multiCluster:
        clusterName: cluster-a
      network: network-a
  meshConfig:
    defaultConfig:
      proxyMetadata:
        ISTIO_META_DNS_CAPTURE: "true"
  components:
    ingressGateways:
    - name: istio-eastwestgateway
      label:
        istio: eastwestgateway
        topology.istio.io/network: network-a
      enabled: true
      k8s:
        env:
        - name: ISTIO_META_REQUESTED_NETWORK_VIEW
          value: network-a
        service:
          ports:
          - name: status-port
            port: 15021
          - name: tls
            port: 15443
          - name: tls-istiod
            port: 15012
          - name: tls-webhook
            port: 15017
```

Cluster B용 IstioOperator는 동일하지만 `clusterName: cluster-b`와 `network: network-b`로 변경합니다.

**(b) 리모트 시크릿 교환:**

```bash
# 양쪽 클러스터에 Istio 설치
istioctl install --context=cluster-a -f cluster-a-operator.yaml -y
istioctl install --context=cluster-b -f cluster-b-operator.yaml -y

# 이스트-웨스트 게이트웨이에서 서비스 노출
kubectl --context=cluster-a apply -n istio-system -f \
  samples/multicluster/expose-services.yaml
kubectl --context=cluster-b apply -n istio-system -f \
  samples/multicluster/expose-services.yaml

# 리모트 시크릿 생성 및 교환
istioctl create-remote-secret --context=cluster-a --name=cluster-a | \
  kubectl apply -f - --context=cluster-b

istioctl create-remote-secret --context=cluster-b --name=cluster-b | \
  kubectl apply -f - --context=cluster-a

# 확인
istioctl remote-clusters --context=cluster-a
istioctl remote-clusters --context=cluster-b
```

**(c) 로컬리티 페일오버가 있는 DestinationRule:**

```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: web-app-locality
  namespace: production
spec:
  host: web-app.production.svc.cluster.local
  trafficPolicy:
    outlierDetection:
      consecutive5xxErrors: 3
      interval: 10s
      baseEjectionTime: 30s
      maxEjectionPercent: 100
    loadBalancer:
      localityLbSetting:
        enabled: true
        distribute:
        - from: "us-east/us-east-1/*"
          to:
            "us-east/us-east-1/*": 80
            "eu-west/eu-west-1/*": 20
        - from: "eu-west/eu-west-1/*"
          to:
            "eu-west/eu-west-1/*": 80
            "us-east/us-east-1/*": 20
        failover:
        - from: us-east
          to: eu-west
        - from: eu-west
          to: us-east
```

**(d) 카나리 헤더 라우팅이 있는 VirtualService:**

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: web-app-canary
  namespace: production
spec:
  hosts:
  - web-app.production.svc.cluster.local
  http:
  - match:
    - headers:
        x-canary:
          exact: "true"
    route:
    - destination:
        host: web-app.production.svc.cluster.local
      headers:
        request:
          set:
            x-preferred-locality: "eu-west/eu-west-1"  # cluster-b의 리전으로 라우팅
  - route:
    - destination:
        host: web-app.production.svc.cluster.local
```

참고: Istio의 로컬리티 기반 라우팅과 `x-preferred-locality` 헤더(또는 서브셋 기반 라우팅)를 결합하면 카나리 트래픽을 cluster-b로 보낼 수 있습니다. 엄격한 클러스터 타겟팅을 위해서는 로컬리티 레이블이 있는 서브셋을 사용하세요.

</details>

### 연습문제 5: 멀티 클러스터 옵저버빌리티

멀티 클러스터 옵저버빌리티 스택을 설계하세요. 다음 구성을 작성하세요: (a) 3개 클러스터에서 Prometheus 메트릭을 페더레이션하는 Thanos (Thanos 사이드카, 스토어 게이트웨이, 쿼리 컴포넌트 포함), (b) 모든 클러스터에서 클러스터 레이블과 함께 로그를 수신하는 멀티 테넌트 모드의 Loki, (c) 클러스터 간 비교 메트릭을 보여주는 Grafana 대시보드 (클러스터별 에러율, 지연시간, 리소스 사용량). 클러스터 식별을 위한 외부 레이블을 포함한 PromQL 쿼리를 포함하세요.

<details>
<summary>정답 보기</summary>

**(a) Thanos 페더레이션:**

```yaml
# Thanos 사이드카 (각 클러스터의 Prometheus와 함께 배포)
# Prometheus Helm 값에 추가:
prometheus:
  prometheusSpec:
    externalLabels:
      cluster: prod-us    # 클러스터마다 다름
      region: us-east
    thanos:
      image: quay.io/thanos/thanos:v0.34.0
      objectStorageConfig:
        existingSecret:
          name: thanos-objstore-config
          key: objstore.yml
    retention: 24h  # 짧은 보존 기간, Thanos가 장기 저장

---
# Thanos 스토어 게이트웨이 (중앙 클러스터)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: thanos-store-gateway
  namespace: monitoring
spec:
  replicas: 2
  selector:
    matchLabels:
      app: thanos-store
  template:
    spec:
      containers:
      - name: thanos-store
        image: quay.io/thanos/thanos:v0.34.0
        args:
        - store
        - --data-dir=/var/thanos/store
        - --objstore.config-file=/etc/thanos/objstore.yml
        - --index-cache-size=500MB
        ports:
        - containerPort: 10901
          name: grpc
        - containerPort: 10902
          name: http
        volumeMounts:
        - name: objstore-config
          mountPath: /etc/thanos
      volumes:
      - name: objstore-config
        secret:
          secretName: thanos-objstore-config
---
# Thanos 쿼리 (중앙 클러스터)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: thanos-query
  namespace: monitoring
spec:
  replicas: 2
  selector:
    matchLabels:
      app: thanos-query
  template:
    spec:
      containers:
      - name: thanos-query
        image: quay.io/thanos/thanos:v0.34.0
        args:
        - query
        - --http-address=0.0.0.0:9090
        - --store=dnssrv+_grpc._tcp.thanos-store-gateway.monitoring.svc
        - --store=thanos-sidecar-prod-us.monitoring.svc:10901
        - --store=thanos-sidecar-prod-eu.monitoring.svc:10901
        - --store=thanos-sidecar-prod-ap.monitoring.svc:10901
        - --query.replica-label=prometheus_replica
        ports:
        - containerPort: 9090
          name: http
```

오브젝트 스토어 구성:

```yaml
# thanos-objstore-config Secret
type: S3
config:
  bucket: thanos-metrics
  endpoint: s3.us-east-1.amazonaws.com
  region: us-east-1
```

**(b) 멀티 테넌트 Loki:**

```yaml
# 멀티 테넌트 모드의 Loki 구성
auth_enabled: true  # 멀티 테넌시 활성화

# 각 클러스터의 Promtail이 클러스터 레이블 추가
# promtail.yaml (클러스터별)
clients:
- url: http://loki-central.monitoring.svc:3100/loki/api/v1/push
  tenant_id: prod-us      # 클러스터마다 다름
  external_labels:
    cluster: prod-us
    region: us-east
```

**(c) Grafana 대시보드 쿼리:**

```promql
# 클러스터별 에러율 (Thanos 쿼리를 통해)
sum by (cluster) (
  rate(http_requests_total{status=~"5.."}[5m])
) /
sum by (cluster) (
  rate(http_requests_total[5m])
) * 100

# 클러스터별 P99 지연시간
histogram_quantile(0.99,
  sum by (cluster, le) (
    rate(http_request_duration_seconds_bucket[5m])
  )
)

# 클러스터별 CPU 사용량
sum by (cluster) (
  rate(container_cpu_usage_seconds_total{container!=""}[5m])
)

# 클러스터별 메모리 사용량 (GiB)
sum by (cluster) (
  container_memory_working_set_bytes{container!=""}
) / 1024 / 1024 / 1024

# 클러스터별 파드 수
count by (cluster) (kube_pod_info)

# 클러스터별 노드 수
count by (cluster) (kube_node_info)

# 크로스 클러스터 비교: 디플로이먼트 레플리카 불일치
sum by (cluster, deployment) (kube_deployment_spec_replicas)
-
sum by (cluster, deployment) (kube_deployment_status_ready_replicas)
```

크로스 클러스터 로그 분석을 위한 LogQL:

```logql
# 클러스터별 에러율 (Loki)
sum by (cluster) (rate({level="error"}[5m]))

# 특정 클러스터의 에러
{cluster="prod-us"} | json | level="error"
```

</details>

---

**이전**: [옵저버빌리티](./14_Observability.md) | **다음**: [Kubernetes API 프로그래밍](./16_Kubernetes_API_Programming.md)

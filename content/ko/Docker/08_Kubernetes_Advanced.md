# 08. Kubernetes 고급

**이전**: [Kubernetes 보안](./07_Kubernetes_Security.md) | **다음**: [Helm 패키지 관리](./09_Helm_Package_Management.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Ingress 리소스를 구성하여 외부 HTTP/HTTPS 트래픽을 클러스터 서비스로 라우팅할 수 있습니다
2. Kubernetes Gateway API 리소스 모델(GatewayClass, Gateway, HTTPRoute)을 설명하고 Ingress와 비교할 수 있다
3. 안정적인 네트워크 ID를 가진 StatefulSet을 사용하여 상태 있는 애플리케이션을 배포할 수 있습니다
4. PersistentVolume과 PersistentVolumeClaim으로 영구 스토리지를 프로비저닝할 수 있습니다
5. 파일 마운트 및 동적 업데이트를 포함한 고급 ConfigMap과 Secret 패턴을 적용할 수 있습니다
6. 노드 수준 에이전트에 DaemonSet을, 배치 워크로드에 Job/CronJob을 사용할 수 있습니다
7. 노드 어피니티(node affinity), 테인트(taints), 톨러레이션(tolerations), 토폴로지 스프레드로 고급 스케줄링을 구현할 수 있습니다

---

기본 Kubernetes 프리미티브인 Pod, Deployment, Service는 많은 사용 사례를 처리하지만, 프로덕션 워크로드는 더 많은 것을 요구합니다. 데이터베이스는 안정적인 스토리지와 네트워크 ID가 필요하고, 웹 애플리케이션은 TLS를 갖춘 외부 HTTP 라우팅이 필요하며, 클러스터 전체 에이전트는 모든 노드에서 실행되어야 합니다. 이 레슨에서는 단순한 컨테이너 오케스트레이션과 프로덕션급 인프라 사이의 간격을 메우는 고급 Kubernetes 리소스와 스케줄링 기법을 소개합니다.

## 목차

리소스 레퍼런스 전에 [**이론과 원리**](#이론과-원리) 섹션을 읽으세요. 스케줄러의 필터+점수 알고리즘, 모든 컨트롤러(커스텀 오퍼레이터 포함)를 떠받치는 컨트롤러 패턴(informer + work queue), 그리고 CRD/오퍼레이터 확장 모델을 다룹니다.

1. [Ingress](#1-ingress)
2. [Gateway API](#2-gateway-api)
3. [StatefulSet](#3-statefulset)
4. [영구 스토리지](#4-영구-스토리지)
5. [ConfigMap 고급](#5-configmap-고급)
6. [DaemonSet과 Job](#6-daemonset과-job)
7. [고급 스케줄링](#7-고급-스케줄링)
8. [연습 문제](#8-연습-문제)

---

## 이론과 원리

Advanced 레슨은 대체로 Kubernetes를 *확장*하는 것에 관한 것입니다 — 새 리소스(CRD), 새 컨트롤러(오퍼레이터), 그리고 기존 스케줄러에게 실제로 원하는 바를 알리는 새 방법. 레슨 나머지의 거의 모든 것이 다음 세 깊은 아이디어 위에 있습니다 — **스케줄러의 두 단계 결정**, **컨트롤러 패턴**(informer + work queue + reconcile), **CRD/오퍼레이터 확장 모델**.

### A. 스케줄러 상세: 필터 + 점수

스케줄러는 스케줄되지 않은 Pod 하나마다(`spec.nodeName == ""`) 한 번 돕니다. 결정은 두 단계 —

**필터 단계(predicates).** 모든 노드에서 이 Pod을 못 돌리는 노드를 떨굼. 고전 predicate —

- `PodFitsResources` — 노드에 `requests`를 만족할 만큼의 할당 가능 CPU/메모리가 남았는가?
- `PodFitsHostPorts` — 요청한 호스트 포트가 비어 있는가?
- `PodToleratesNodeTaints` — Pod이 노드의 `NoSchedule` taint를 toleration하는가?
- `MatchNodeSelector` — 노드의 라벨이 `spec.nodeSelector`와 `nodeAffinity`에 매칭되는가?
- `NoVolumeZoneConflict` — 상태 있는 Pod의 경우, 노드가 PVC 볼륨과 같은 클라우드 존에 있는가?
- `MatchInterPodAffinity` — 이 노드가 `podAffinity`와 `podAntiAffinity`를 만족하는가?

predicate 하나라도 실패한 노드는 후보에서 제외. 살아남은 노드가 0이면 Pod은 영원히 Pending(이벤트가 이유 설명).

**점수 단계(priorities).** 살아남은 각 노드에 priority 함수의 가중 합으로 0~100 점수를 계산.

- `BalancedResourceAllocation` — CPU와 메모리 사용률이 균형 잡힌 노드 선호(메모리는 10%인데 CPU만 90%로 채우지 않음).
- `LeastRequestedPriority` — 자유 자원이 많은 노드 선호(부하 분산).
- `NodeAffinityPriority` — 소프트 `nodeAffinity`가 *선호*하는 노드 선호.
- `InterPodAffinityPriority` — 소프트 Pod 간 affinity에 같은 식.
- `ImageLocalityPriority` — 이미지를 이미 캐시한 노드 선호(시작 빠름).
- `TaintTolerationPriority` — 모든 `PreferNoSchedule` taint가 toleration된 노드 선호.

가장 점수 높은 노드 선택, 동률은 무작위. 선택된 노드가 `spec.nodeName`에 쓰이고, 그 노드의 kubelet이 Pod을 가져가 런타임 시작.

스케줄러는 **플러그인 가능**합니다 — 자체 predicate와 priority를 작성하고 *두 번째 스케줄러*를 기본 옆에서 돌릴 수 있습니다. Pod은 `spec.schedulerName: my-scheduler`로 스케줄러를 선택. 이상한 하드웨어를 가진 팀(다양한 세대 GPU, FPGA, 특수 NIC)이 기본을 포크하지 않고 스케줄링을 확장하는 방법.

**Taint와 toleration**이 가장 유용한 수동 스케줄링 도구 — 노드를 `gpu=true:NoSchedule`로 taint하면 매칭 toleration 없는 것은 모두 못 들어옴. Pod의 toleration은 taint된 노드에 들어갈 *허락*이지 *요구사항*이 아님("여기서 반드시 돌아라"는 `nodeSelector`/`nodeAffinity` 사용). 조합 — 노드 전용으로 taint, 요구/선호로 affinity — 이 실세계 토폴로지 대부분을 커버.

### B. 컨트롤러 패턴: Informer + Work Queue + Reconcile

모든 K8s 컨트롤러(내장이든 커스텀이든)는 같은 내부 아키텍처를 따릅니다 — **컨트롤러 패턴(controller pattern)**. 호출하는 프레임워크가 아니라, API 서버 설계가 밀어 넣는 구조입니다.

**Informer.** API 서버에 대한 장수 watch + 로컬 인메모리 캐시. informer는 —

1. 시작 시 watch 대상 타입의 모든 객체를 list해 캐시에 저장.
2. watch(`?watch=true&resourceVersion=...`) 수립, 스트리밍 Add/Update/Delete 이벤트 수신.
3. 로컬 캐시 갱신, 등록된 이벤트 핸들러 발화.

캐시 덕분에 컨트롤러가 API 서버를 read로 두드리지 않고, watch 덕분에 변경을 밀리초 단위로 들음.

**Work queue.** 이벤트 핸들러는 직접 작업을 *하지 않습니다*. 영향받은 객체의 *키*(`namespace/name`)를 중복 제거, 레이트 제한 work queue에 넣습니다. 같은 키에 대한 여러 이벤트는 하나의 work item으로 합쳐집니다. 조정 실패 시 지수 백오프로 재큐.

**Reconcile 루프.** 워커 고루틴이 큐에서 키를 빼고, informer 캐시로 그 객체의 현재 상태를 조회하고, spec에서 원하는 상태를 계산하고, 차이를 좁히는 행동(자식 생성, status 갱신, 고아 삭제)을 취합니다. *무엇이 바뀌었는지 들은 게 아니라* 매번 새로 차이를 계산합니다. 이게 reconcile 함수를 **멱등**하고 **상태 기반(state-driven)**(이벤트 기반이 아닌)으로 만듭니다.

이 패턴 전체가 `controller-runtime`(오퍼레이터를 빌드하는 Go 라이브러리)에 담겨 있습니다. 같은 모양이 Deployment 컨트롤러, ReplicaSet 컨트롤러, kube-controller-manager 번들, 그리고 당신이 쓰는 모든 CRD 오퍼레이터의 바닥에 있습니다.

규율 — **이벤트를 절대 신뢰하지 말고, 항상 상태를 다시 읽어라.** 큐가 지연되면 세상은 이미 움직였고, 방금 받은 이벤트가 아니라 캐시에 대해 조정하는 것이 stale-event 버그를 방어합니다.

### C. CRD와 오퍼레이터: API 확장

Kubernetes는 처음부터 확장 가능하도록 설계되었습니다. 메커니즘은 **CRD(Custom Resource Definition)** — 새 리소스 타입을 API 서버에 등록하는 YAML 객체.

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: postgresqls.acid.zalan.do
spec:
  group: acid.zalan.do
  names: {kind: PostgreSQL, plural: postgresqls, shortNames: [pg]}
  scope: Namespaced
  versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema: {...}
```

이걸 적용하면 API 서버가 `kubectl get postgresqls`를 서빙할 줄 알게 됩니다. 사용자가 이제 `PostgreSQL` 객체를 만들 수 있고 다른 내장 리소스처럼 etcd에 영속됩니다 — 그러나 컨트롤러가 watch할 때까지 *아무 일도 안 일어납니다*.

그 컨트롤러가 **오퍼레이터(operator)**입니다. CRD를 watch하고 spec을 실제 리소스로 조정하는 컨트롤러 패턴(§B) 프로세스에 불과합니다 — Postgres pod용 StatefulSet, 클라이언트용 Service, 자격 증명용 Secret, 정기 백업, failover 로직. 오퍼레이터가 운영 지식 — "PostgreSQL을 고가용으로 만들려면 X, Y, Z를 해야 한다" — 을 계속 도는 코드로 인코딩합니다.

이 패턴이 동작하는 이유 — CRD가 도메인 전문가에게 고수준 리소스(`PostgreSQL`)를 정의하게 하고, 그것을 운영할 줄 아는 컨트롤러를 함께 배포할 수 있기 때문. 사용자는 `kubectl apply -f postgres.yaml`을 보고, 오퍼레이터가 그 아래에서 StatefulSet/PVC/복제를 다룹니다. 예시 — cert-manager(`Certificate`, `Issuer`), Prometheus 오퍼레이터(`Prometheus`, `ServiceMonitor`), Argo CD(`Application`), Strimzi(`Kafka`).

Operator Framework / OperatorSDK / kubebuilder는 Go(또는 controller-runtime 포트가 있는 다른 언어)로 오퍼레이터를 작성하기 위한 스캐폴딩 도구들. Helm 차트도 오퍼레이터 모델을 쓸 수 있습니다 — helm-operator가 차트를 감싸 CRD 인스턴스로 배포되게 함.

### D. StatefulSet과 "안정적 정체성" 문제

Deployment는 Pod을 교환 가능한 복제로 다룹니다 — 같은 이미지, 같은 설정, 정체성 없음. 데이터베이스와 합의 시스템(Postgres 복제, ZooKeeper ensemble, Cassandra ring)에는 틀린 모델입니다. 이들은 다음을 필요로 합니다.

- **안정적 네트워크 정체성.** Pod 0은 재시작 후에도 항상 `db-0.db-service.namespace.svc.cluster.local`이어야 함.
- **안정적 스토리지.** Pod 0의 PVC가 재스케줄에도 Pod 0을 따라가야 함 — 새 빈 볼륨이 아님.
- **순서 있는 롤아웃.** 업데이트 중 Pod N이 healthy가 된 후에만 Pod N-1 교체. 스케일 다운은 역순.

`StatefulSet`이 셋 다 제공 — Pod 이름이 `<name>-0`, `<name>-1`, ..., `headless Service`(clusterIP=None)가 각 Pod에 자체 DNS 이름 부여, `volumeClaimTemplates`가 pod 삭제에도 살아남는 복제당 PVC 하나씩 생성.

대가는 운영. StatefulSet 업데이트는 더 느림(한 번에 한 Pod, 순서대로). 스케일 다운은 PVC를 유지(다시 스케일 업하면 재부착 — 정말 없애려면 수동 정리). 그리고 애플리케이션 자체가 정체성을 알아야 함("나는 복제 0, 나는 primary").

### E. 영구 스토리지: PV, PVC, StorageClass

K8s의 스토리지는 개발자와 클러스터 관리자를 분리하기 위해 세 리소스로 쪼개집니다.

- **PersistentVolume(PV)** — *존재하는* 스토리지 덩어리(실제 EBS 볼륨, 실제 NFS export, 실제 Ceph 이미지)를 나타내는 클러스터 스코프 리소스. 관리자가 만들거나 동적 프로비저너가 만듦.
- **PersistentVolumeClaim(PVC)** — 네임스페이스 스코프 스토리지 요청 — "10Gi의 `fast-ssd` 스토리지가 ReadWriteOnce로 필요하다." 애플리케이션 개발자가 만듦.
- **StorageClass** — 동적 프로비저닝의 템플릿 — "PVC가 이 클래스를 참조하면, 이 매개변수로 PV를 할당하기 위해 *이* CSI 드라이버를 돌려라." 클러스터 관리자가 한 번 만듦.

매칭 로직 — 새 PVC가 적합한 기존 PV(크기 + 클래스 + access mode)에 바인딩되거나, 없으면 StorageClass의 프로비저너를 트리거해 생성. 바인딩되면 Pod의 `volumes` 필드가 PVC를 이름으로 참조, kubelet이 기저 볼륨을 컨테이너에 마운트.

`accessModes`가 중요 — `ReadWriteOnce`(단일 노드, EBS 같은 블록 디바이스에 적합), `ReadWriteMany`(다중 노드, NFS류 백엔드 필요), `ReadOnlyMany`. 대부분의 클라우드 블록 스토리지는 RWO이고, 상태 있는 워크로드를 단일 노드 배치로 강제.

CSI(Container Storage Interface)가 표준 플러그인 API — 모든 클라우드와 스토리지 벤더가 CSI 드라이버를 배포, K8s는 같은 인터페이스로 그들 모두와 통신. 스냅샷, 복제, 리사이징은 스토리지 백엔드가 지원할 수도 있고 아닐 수도 있는 CSI 기능.

### 이론에서 아래의 리소스로

- **Ingress / Gateway API** — `Ingress`/`HTTPRoute` 리소스(CRD 패턴 §C)를 watch하고 nginx/Envoy/Istio 설정으로 조정하는 컨트롤러.
- **StatefulSet** — §D의 YAML 형태. headless Service + volumeClaimTemplates를 엮어 안정적 정체성 + 스토리지.
- **PV / PVC / StorageClass** — §E. 기저 디스크를 명명하지 않고도 "영구 스토리지가 필요하다"를 의미 있게 만드는 스토리지 분리.
- **DaemonSet** — reconcile 규칙이 "이 셀렉터에 매칭되는 노드마다 복제 하나"인 컨트롤러(§B). 로그 시퍼, 노드 모니터링, CNI 에이전트에 사용.
- **Job / CronJob** — "이 Pod을 N번 완료까지 실행"(Job)이나 "이 스케줄에 Job 생성"(CronJob)을 조정하는 컨트롤러.
- **Affinity / antiAffinity / taint / toleration** — §A. 스케줄러 결정을 편향시키기 위해 쓰는 어휘.
- **커스텀 컨트롤러 / 오퍼레이터** — §B + §C 결합. 대부분의 프로덕션 K8s가 결국 여러 개를 돌리게 됨.

남은 섹션은 이 리소스들을 둘러봅니다. 리소스가 신비롭게 느껴질 때마다, 어느 컨트롤러가 그것을 소유하고 무엇을 watch하는지 식별하세요. 조정 패턴이 나머지를 설명합니다.

---

## 1. Ingress

### 1.1 Ingress 개념

```
┌─────────────────────────────────────────────────────────────┐
│                    Ingress Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Internet                                                  │
│      │                                                      │
│      ▼                                                      │
│  ┌───────────────────────────────────────────┐             │
│  │         Ingress Controller                 │             │
│  │    (nginx, traefik, haproxy, etc.)        │             │
│  └───────────────────┬───────────────────────┘             │
│                      │                                      │
│        ┌─────────────┼─────────────┐                       │
│        │             │             │                        │
│        ▼             ▼             ▼                        │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐                   │
│   │Ingress  │  │Ingress  │  │Ingress  │                   │
│   │Resource │  │Resource │  │Resource │                   │
│   └────┬────┘  └────┬────┘  └────┬────┘                   │
│        │             │             │                        │
│        ▼             ▼             ▼                        │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐                   │
│   │Service A│  │Service B│  │Service C│                   │
│   └─────────┘  └─────────┘  └─────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Ingress Controller 설치

```bash
# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.11.3/deploy/static/provider/cloud/deploy.yaml

# Verify installation
kubectl get pods -n ingress-nginx
kubectl get svc -n ingress-nginx

# Check IngressClass
kubectl get ingressclass
```

### 1.3 기본 Ingress

```yaml
# simple-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: simple-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  rules:
  - host: myapp.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 80

---
# Host-based routing
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: host-based-ingress
spec:
  ingressClassName: nginx
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 8080
  - host: admin.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: admin-service
            port:
              number: 3000
```

### 1.4 경로 기반 라우팅

```yaml
# path-based-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: path-based-ingress
  annotations:
    nginx.ingress.kubernetes.io/use-regex: "true"
spec:
  ingressClassName: nginx
  rules:
  - host: app.example.com
    http:
      paths:
      # /api/* → api-service
      - path: /api(/|$)(.*)
        pathType: ImplementationSpecific
        backend:
          service:
            name: api-service
            port:
              number: 8080
      # /static/* → static-service
      - path: /static
        pathType: Prefix
        backend:
          service:
            name: static-service
            port:
              number: 80
      # Default → frontend
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 80
```

### 1.5 TLS 설정

```yaml
# tls-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tls-ingress
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"  # Force HTTPS — prevents sensitive data from traveling over plaintext HTTP
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - secure.example.com
    secretName: tls-secret  # TLS Secret
  rules:
  - host: secure.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: secure-service
            port:
              number: 443

---
# Create TLS Secret
apiVersion: v1
kind: Secret
metadata:
  name: tls-secret
type: kubernetes.io/tls
data:
  tls.crt: <base64-encoded-cert>
  tls.key: <base64-encoded-key>
```

### 1.6 고급 Ingress 설정

```yaml
# advanced-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: advanced-ingress
  annotations:
    # Basic settings
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "60"

    # Rate Limiting — protects backend from DoS and abusive clients
    nginx.ingress.kubernetes.io/limit-rps: "100"
    nginx.ingress.kubernetes.io/limit-connections: "50"

    # CORS
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "https://frontend.example.com"
    nginx.ingress.kubernetes.io/cors-allow-methods: "GET, POST, PUT, DELETE, OPTIONS"

    # Authentication
    nginx.ingress.kubernetes.io/auth-type: basic
    nginx.ingress.kubernetes.io/auth-secret: basic-auth
    nginx.ingress.kubernetes.io/auth-realm: "Authentication Required"

    # Custom headers — security headers that defend against clickjacking and MIME-sniffing attacks
    nginx.ingress.kubernetes.io/configuration-snippet: |
      add_header X-Frame-Options "SAMEORIGIN";
      add_header X-Content-Type-Options "nosniff";

spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.example.com
    secretName: api-tls
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 8080
```

---

## 2. Gateway API

Gateway API는 Ingress의 공식 후속 기술로, Kubernetes 1.31부터 GA입니다. 역할 기반 리소스 모델을 통해 인프라 관심사와 애플리케이션 라우팅을 분리하여 Ingress의 한계를 해결합니다.

### 2.1 Ingress vs Gateway API

```
┌─────────────────────────────────────────────────────────────┐
│              Ingress vs Gateway API                          │
├──────────────────────┬──────────────────────────────────────┤
│      Ingress         │        Gateway API                   │
├──────────────────────┼──────────────────────────────────────┤
│  단일 리소스          │  3개 계층 리소스:                      │
│                      │  GatewayClass → Gateway → *Route     │
├──────────────────────┼──────────────────────────────────────┤
│  L7 HTTP만 지원       │  L4+L7: HTTP, gRPC, TCP, TLS, UDP   │
├──────────────────────┼──────────────────────────────────────┤
│  트래픽 분할 불가      │  네이티브 가중치 기반 트래픽 분할       │
├──────────────────────┼──────────────────────────────────────┤
│  어노테이션 의존       │  타입화된 구조적 설정                  │
├──────────────────────┼──────────────────────────────────────┤
│  단일 테넌트          │  멀티 테넌트 (역할 분리)               │
├──────────────────────┼──────────────────────────────────────┤
│  1.19부터 안정        │  1.31부터 GA                         │
└──────────────────────┴──────────────────────────────────────┘

역할 기반 설계:
  인프라 팀     → GatewayClass 배포 (클러스터 범위)
  플랫폼 팀     → Gateway 생성    (네임스페이스 범위)
  애플리케이션 팀 → HTTPRoute 연결  (네임스페이스 범위)
```

### 2.2 핵심 리소스

```yaml
# gateway-api-example.yaml

# 1. GatewayClass — 클러스터 범위, 구현체 정의
apiVersion: gateway.networking.k8s.io/v1
kind: GatewayClass
metadata:
  name: nginx
spec:
  controllerName: gateway.nginx.org/nginx-gateway-controller

---
# 2. Gateway — 네임스페이스 범위, 로드 밸런서 생성
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: main-gateway
  namespace: infra
spec:
  gatewayClassName: nginx
  listeners:
  - name: http
    port: 80
    protocol: HTTP
    allowedRoutes:
      namespaces:
        from: All  # 모든 네임스페이스의 라우트 허용 — 더 엄격한 멀티테넌시를 위해 "Same" 또는 "Selector"로 조정
  - name: https
    port: 443
    protocol: HTTPS
    tls:
      mode: Terminate
      certificateRefs:
      - name: tls-cert
    allowedRoutes:
      namespaces:
        from: All

---
# 3. HTTPRoute — 네임스페이스 범위, 라우팅 규칙 정의
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: app-routes
  namespace: production
spec:
  parentRefs:
  - name: main-gateway
    namespace: infra
  hostnames:
  - "app.example.com"
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /api
    backendRefs:
    - name: api-service
      port: 8080
  - matches:
    - path:
        type: PathPrefix
        value: /
    backendRefs:
    - name: frontend-service
      port: 80
```

### 2.3 트래픽 분할 (카나리 배포)

Gateway API는 가중치 기반 라우팅을 통해 카나리 배포를 일급 기능으로 지원합니다 — 별도의 어노테이션이 필요하지 않습니다.

```yaml
# canary-httproute.yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: canary-route
spec:
  parentRefs:
  - name: main-gateway
    namespace: infra
  hostnames:
  - "app.example.com"
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /
    backendRefs:
    - name: app-stable
      port: 80
      weight: 90   # 안정 버전에 트래픽 90%
    - name: app-canary
      port: 80
      weight: 10   # 카나리에 10% — 신뢰가 높아지면 점진적으로 증가
```

### 2.4 헤더 기반 라우팅

```yaml
# header-routing.yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: header-route
spec:
  parentRefs:
  - name: main-gateway
    namespace: infra
  rules:
  # 내부 사용자는 베타 버전으로 라우팅
  - matches:
    - headers:
      - name: X-User-Group
        value: internal
    backendRefs:
    - name: app-beta
      port: 80
  # 나머지는 안정 버전으로
  - backendRefs:
    - name: app-stable
      port: 80
```

---

## 3. StatefulSet

### 2.1 StatefulSet 개념

```
┌─────────────────────────────────────────────────────────────┐
│            StatefulSet vs Deployment                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Deployment (Stateless)                                     │
│  ┌───────┐ ┌───────┐ ┌───────┐                             │
│  │pod-xyz│ │pod-abc│ │pod-123│  Random names, replaceable  │
│  └───────┘ └───────┘ └───────┘                             │
│                                                             │
│  StatefulSet (Stateful)                                     │
│  ┌───────┐ ┌───────┐ ┌───────┐                             │
│  │web-0  │ │web-1  │ │web-2  │  Ordered, unique IDs        │
│  │  ↓    │ │  ↓    │ │  ↓    │                             │
│  │pvc-0  │ │pvc-1  │ │pvc-2  │  Dedicated storage each     │
│  └───────┘ └───────┘ └───────┘                             │
│                                                             │
│  Features:                                                  │
│  • Ordered creation/deletion (0 → 1 → 2)                   │
│  • Fixed network IDs (pod-name.service-name)               │
│  • Persistent storage attached                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 StatefulSet 정의

```yaml
# statefulset-example.yaml
apiVersion: v1
kind: Service
metadata:
  name: web-headless
  labels:
    app: web
spec:
  ports:
  - port: 80
    name: web
  clusterIP: None  # Headless Service — gives each pod a stable DNS name (pod-name.svc) instead of a single cluster IP
  selector:
    app: web

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: web
spec:
  serviceName: "web-headless"  # Connect to Headless Service — required for stable per-pod DNS names
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
      - name: nginx
        image: nginx:1.25
        ports:
        - containerPort: 80
          name: web
        volumeMounts:
        - name: www
          mountPath: /usr/share/nginx/html

  # Volume claim templates
  volumeClaimTemplates:
  - metadata:
      name: www
    spec:
      accessModes: ["ReadWriteOnce"]  # Each pod gets its own PVC — prevents data corruption from concurrent writes
      storageClassName: standard
      resources:
        requests:
          storage: 1Gi

  # Update strategy
  updateStrategy:
    type: RollingUpdate  # Zero-downtime deploy: new pods start before old ones terminate
    rollingUpdate:
      partition: 0  # Only Pods >= this number are updated — useful for canary-testing a subset of replicas

  # Pod management policy
  podManagementPolicy: OrderedReady  # Or Parallel — OrderedReady ensures each pod is Running before starting the next (safe for leader election)
```

### 2.3 데이터베이스 StatefulSet

```yaml
# mysql-statefulset.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mysql-config
data:
  my.cnf: |
    [mysqld]
    bind-address = 0.0.0.0
    default_authentication_plugin = mysql_native_password

---
apiVersion: v1
kind: Secret
metadata:
  name: mysql-secret
type: Opaque
stringData:
  root-password: "rootpass123"
  user-password: "userpass123"

---
apiVersion: v1
kind: Service
metadata:
  name: mysql-headless
spec:
  clusterIP: None
  selector:
    app: mysql
  ports:
  - port: 3306

---
apiVersion: v1
kind: Service
metadata:
  name: mysql
spec:
  selector:
    app: mysql
    statefulset.kubernetes.io/pod-name: mysql-0  # Primary only — routes all write traffic to the leader, preventing split-brain
  ports:
  - port: 3306

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mysql
spec:
  serviceName: mysql-headless
  replicas: 3
  selector:
    matchLabels:
      app: mysql
  template:
    metadata:
      labels:
        app: mysql
    spec:
      initContainers:
      - name: init-mysql
        image: mysql:8.0
        command:
        - bash
        - "-c"
        - |
          set -ex
          # Generate server-id from Pod index
          [[ `hostname` =~ -([0-9]+)$ ]] || exit 1
          ordinal=${BASH_REMATCH[1]}
          echo [mysqld] > /mnt/conf.d/server-id.cnf
          echo server-id=$((100 + $ordinal)) >> /mnt/conf.d/server-id.cnf
        volumeMounts:
        - name: conf
          mountPath: /mnt/conf.d

      containers:
      - name: mysql
        image: mysql:8.0
        env:
        - name: MYSQL_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mysql-secret
              key: root-password
        ports:
        - containerPort: 3306
        volumeMounts:
        - name: data
          mountPath: /var/lib/mysql
        - name: conf
          mountPath: /etc/mysql/conf.d
        - name: config
          mountPath: /etc/mysql/my.cnf
          subPath: my.cnf
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
        livenessProbe:  # liveness restarts the pod if MySQL hangs; separate from readiness to avoid cascading restarts
          exec:
            command: ["mysqladmin", "ping"]
          initialDelaySeconds: 30  # Give MySQL time to initialize before checking — avoids restart loops on slow starts
          periodSeconds: 10
        readinessProbe:  # readiness gates traffic; a failing probe removes the pod from the Service
          exec:
            command: ["mysql", "-h", "127.0.0.1", "-e", "SELECT 1"]  # Verifies the query engine is ready, not just the process
          initialDelaySeconds: 5
          periodSeconds: 2

      volumes:
      - name: conf
        emptyDir: {}
      - name: config
        configMap:
          name: mysql-config

  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast
      resources:
        requests:
          storage: 10Gi
```

### 2.4 StatefulSet 관리

```bash
# View StatefulSets
kubectl get statefulset
kubectl describe statefulset web

# Check Pods (ordered names)
kubectl get pods -l app=web
# NAME    READY   STATUS
# web-0   1/1     Running
# web-1   1/1     Running
# web-2   1/1     Running

# Check DNS
# Each Pod: web-0.web-headless.default.svc.cluster.local
kubectl run -it --rm debug --image=busybox -- nslookup web-0.web-headless

# Scaling
kubectl scale statefulset web --replicas=5

# Rolling update
kubectl set image statefulset/web nginx=nginx:1.26

# Update specific Pods only (using partition)
kubectl patch statefulset web -p '{"spec":{"updateStrategy":{"rollingUpdate":{"partition":2}}}}'

# Delete (PVCs are retained)
kubectl delete statefulset web
kubectl delete pvc -l app=web  # Delete PVCs
```

---

## 4. 영구 스토리지

### 3.1 스토리지 계층 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    Storage Hierarchy                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────┐               │
│  │              Pod                         │               │
│  │  ┌─────────────────────────────────┐   │               │
│  │  │     Volume Mount                 │   │               │
│  │  │     /data                        │   │               │
│  │  └─────────────┬───────────────────┘   │               │
│  └─────────────────┼───────────────────────┘               │
│                    │                                        │
│                    ▼                                        │
│  ┌─────────────────────────────────────────┐               │
│  │     PersistentVolumeClaim (PVC)         │               │
│  │     • Storage request                   │               │
│  │     • Namespace-scoped                  │               │
│  └─────────────────┬───────────────────────┘               │
│                    │ Binding                                │
│                    ▼                                        │
│  ┌─────────────────────────────────────────┐               │
│  │     PersistentVolume (PV)               │               │
│  │     • Actual storage                    │               │
│  │     • Cluster-scoped                    │               │
│  └─────────────────┬───────────────────────┘               │
│                    │                                        │
│                    ▼                                        │
│  ┌─────────────────────────────────────────┐               │
│  │     StorageClass                        │               │
│  │     • Dynamic provisioning              │               │
│  │     • Storage type definition           │               │
│  └─────────────────────────────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 StorageClass

```yaml
# storageclass.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: kubernetes.io/gce-pd  # Varies by cloud provider
parameters:
  type: pd-ssd
reclaimPolicy: Delete  # Delete or Retain — Delete auto-cleans cloud disks; use Retain for databases that need manual backup before deletion
allowVolumeExpansion: true  # Lets you grow PVCs in place — avoids data migration when storage needs increase
volumeBindingMode: WaitForFirstConsumer  # Delays provisioning until a pod is scheduled — ensures the disk is in the same zone as the pod

---
# AWS EBS StorageClass
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: aws-fast
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
reclaimPolicy: Delete
allowVolumeExpansion: true

---
# Local StorageClass
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: local-storage
provisioner: kubernetes.io/no-provisioner
volumeBindingMode: WaitForFirstConsumer
```

### 3.3 PersistentVolume (정적 프로비저닝)

```yaml
# pv-static.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-manual
  labels:
    type: local
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: manual
  hostPath:
    path: /mnt/data

---
# NFS PV
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-nfs
spec:
  capacity:
    storage: 100Gi
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: nfs
  nfs:
    server: nfs-server.example.com
    path: /exports/data

---
# AWS EBS PV
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-aws-ebs
spec:
  capacity:
    storage: 50Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: aws-fast
  awsElasticBlockStore:
    volumeID: vol-0123456789abcdef
    fsType: ext4
```

### 3.4 PersistentVolumeClaim

```yaml
# pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: app-data-pvc
  namespace: production
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: fast  # Dynamic provisioning
  # selector:             # Use for static binding
  #   matchLabels:
  #     type: local

---
# Using PVC in Pod
apiVersion: v1
kind: Pod
metadata:
  name: app-with-storage
spec:
  containers:
  - name: app
    image: myapp:latest
    volumeMounts:
    - name: data
      mountPath: /app/data
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: app-data-pvc
```

### 3.5 볼륨 확장 및 스냅샷

```yaml
# Expand PVC (requires allowVolumeExpansion: true)
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: expandable-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi  # Expanded from 5Gi
  storageClassName: fast

---
# VolumeSnapshot (requires CSI driver)
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: data-snapshot
spec:
  volumeSnapshotClassName: csi-snapclass
  source:
    persistentVolumeClaimName: app-data-pvc

---
# Restore PVC from snapshot
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: restored-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: fast
  dataSource:
    name: data-snapshot
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
```

---

## 5. ConfigMap 고급

### 4.1 ConfigMap 생성 방법

```bash
# Create from literals
kubectl create configmap app-config \
  --from-literal=LOG_LEVEL=info \
  --from-literal=MAX_CONNECTIONS=100

# Create from file
kubectl create configmap nginx-config \
  --from-file=nginx.conf

# Create from directory
kubectl create configmap app-configs \
  --from-file=config/
```

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  # Simple key-value
  LOG_LEVEL: "info"
  DATABASE_HOST: "db.example.com"

  # File format
  app.properties: |
    server.port=8080
    server.host=0.0.0.0
    logging.level=INFO

  nginx.conf: |
    server {
        listen 80;
        server_name localhost;

        location / {
            root /usr/share/nginx/html;
            index index.html;
        }

        location /api {
            proxy_pass http://backend:8080;
        }
    }
```

### 4.2 ConfigMap 사용 방법

```yaml
# configmap-usage.yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
  - name: app
    image: myapp:latest

    # Inject as environment variable
    env:
    - name: LOG_LEVEL
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: LOG_LEVEL

    # Entire ConfigMap as environment variables
    envFrom:
    - configMapRef:
        name: app-config
      prefix: APP_  # Optional prefix

    volumeMounts:
    # Mount as file
    - name: config-volume
      mountPath: /etc/app
    # Mount specific key only
    - name: nginx-volume
      mountPath: /etc/nginx/conf.d

  volumes:
  - name: config-volume
    configMap:
      name: app-config
      # All items
  - name: nginx-volume
    configMap:
      name: app-config
      items:
      - key: nginx.conf
        path: default.conf
        mode: 0644
```

### 4.3 ConfigMap 변경 감지

```yaml
# Auto-reload (using Reloader)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-deployment
  annotations:
    # stakater/Reloader annotation
    reloader.stakater.com/auto: "true"
    # Or specific ConfigMap only
    configmap.reloader.stakater.com/reload: "app-config"
spec:
  template:
    spec:
      containers:
      - name: app
        image: myapp:latest
        volumeMounts:
        - name: config
          mountPath: /etc/app
      volumes:
      - name: config
        configMap:
          name: app-config

---
# Sidecar for change detection
apiVersion: v1
kind: Pod
metadata:
  name: app-with-reloader
spec:
  containers:
  - name: app
    image: myapp:latest
    volumeMounts:
    - name: config
      mountPath: /etc/app

  - name: config-reloader
    image: jimmidyson/configmap-reload:v0.5.0
    args:
    - --volume-dir=/etc/app
    - --webhook-url=http://localhost:8080/-/reload
    volumeMounts:
    - name: config
      mountPath: /etc/app
      readOnly: true

  volumes:
  - name: config
    configMap:
      name: app-config
```

---

## 6. DaemonSet과 Job

### 5.1 DaemonSet

```yaml
# daemonset.yaml
# Deploy Pod on every node
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: node-exporter
  labels:
    app: node-exporter
spec:
  selector:
    matchLabels:
      app: node-exporter
  template:
    metadata:
      labels:
        app: node-exporter
    spec:
      # Deploy on specific nodes only
      nodeSelector:
        monitoring: "true"

      tolerations:
      # Deploy on master nodes too — monitoring must cover control-plane nodes for full cluster visibility
      - key: node-role.kubernetes.io/control-plane
        operator: Exists
        effect: NoSchedule

      containers:
      - name: node-exporter
        image: prom/node-exporter:v1.6.1
        ports:
        - containerPort: 9100
          hostPort: 9100
        volumeMounts:
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: sys
          mountPath: /host/sys
          readOnly: true
        resources:
          limits:
            cpu: 200m
            memory: 100Mi
          requests:
            cpu: 100m
            memory: 50Mi

      hostNetwork: true  # Needed so the exporter can see host-level network metrics and bind to the node's IP
      hostPID: true  # Required to enumerate host processes for per-process metrics

      volumes:
      - name: proc
        hostPath:
          path: /proc
      - name: sys
        hostPath:
          path: /sys

  updateStrategy:
    type: RollingUpdate  # Zero-downtime: updates one node at a time so monitoring coverage is never fully lost
    rollingUpdate:
      maxUnavailable: 1  # Only one node loses its exporter during rollout — balances speed vs observability gap
```

### 5.2 Job

```yaml
# job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: data-migration
spec:
  # Completion conditions
  completions: 1        # Number of successful Pods required
  parallelism: 1        # Number of concurrent Pods
  backoffLimit: 3       # Retry count on failure — prevents infinite retry loops on permanent errors
  activeDeadlineSeconds: 600  # Maximum execution time — hard kill prevents runaway jobs from consuming resources forever

  # Delete after completion — auto-cleanup keeps the cluster tidy and avoids accumulating finished pods
  ttlSecondsAfterFinished: 3600

  template:
    spec:
      restartPolicy: Never  # OnFailure or Never
      containers:
      - name: migrator
        image: myapp/migrator:latest
        command: ["python", "migrate.py"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url

---
# Parallel Job
apiVersion: batch/v1
kind: Job
metadata:
  name: parallel-processing
spec:
  completions: 10     # Total 10 completions
  parallelism: 3      # 3 at a time
  template:
    spec:
      restartPolicy: OnFailure
      containers:
      - name: worker
        image: myapp/worker:latest
```

### 5.3 CronJob

```yaml
# cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: db-backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  timeZone: "Asia/Seoul"

  # Concurrency policy — Forbid prevents overlapping backup runs that could corrupt data
  concurrencyPolicy: Forbid  # Allow, Forbid, or Replace

  # Starting deadline — if the scheduler misses the window by >300s, skip this run rather than starting a stale backup
  startingDeadlineSeconds: 300

  # Success/failure history — keep enough history for debugging but avoid cluttering etcd with old Job objects
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 1

  # Suspend
  suspend: false

  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: backup
            image: postgres:15
            command:
            - /bin/sh
            - -c
            - |
              pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME > /backup/db_$(date +%Y%m%d).sql
              aws s3 cp /backup/db_$(date +%Y%m%d).sql s3://backups/
            env:
            - name: DB_HOST
              value: "postgres"
            - name: DB_USER
              valueFrom:
                secretKeyRef:
                  name: db-secret
                  key: username
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: db-secret
                  key: password
            volumeMounts:
            - name: backup
              mountPath: /backup
          volumes:
          - name: backup
            emptyDir: {}
```

---

## 7. 고급 스케줄링

### 6.1 Node Affinity

```yaml
# node-affinity.yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  affinity:
    nodeAffinity:
      # Required conditions
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: gpu
            operator: In
            values:
            - "true"
          - key: kubernetes.io/arch
            operator: In
            values:
            - amd64

      # Preferred conditions
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        preference:
          matchExpressions:
          - key: gpu-type
            operator: In
            values:
            - nvidia-a100
      - weight: 50
        preference:
          matchExpressions:
          - key: gpu-type
            operator: In
            values:
            - nvidia-v100

  containers:
  - name: gpu-app
    image: nvidia/cuda:12.0-base
    resources:
      limits:
        nvidia.com/gpu: 1
```

### 6.2 Pod Affinity/Anti-Affinity

```yaml
# pod-affinity.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      affinity:
        # Prefer same node as cache Pod
        podAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - cache
              topologyKey: kubernetes.io/hostname

        # Deploy on different nodes than other Pods of same app — ensures a single node failure doesn't take down all replicas
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - web
            topologyKey: kubernetes.io/hostname

      containers:
      - name: web
        image: nginx:latest
```

### 6.3 Taints와 Tolerations

```bash
# Add Taint to node
kubectl taint nodes node1 dedicated=gpu:NoSchedule
kubectl taint nodes node2 special=true:PreferNoSchedule
kubectl taint nodes node3 critical=true:NoExecute

# Remove Taint
kubectl taint nodes node1 dedicated=gpu:NoSchedule-
```

```yaml
# tolerations.yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  tolerations:
  # Exact match
  - key: "dedicated"
    operator: "Equal"
    value: "gpu"
    effect: "NoSchedule"

  # Key exists
  - key: "special"
    operator: "Exists"
    effect: "PreferNoSchedule"

  # NoExecute + tolerationSeconds
  - key: "critical"
    operator: "Equal"
    value: "true"
    effect: "NoExecute"
    tolerationSeconds: 3600  # Evict after 1 hour

  containers:
  - name: app
    image: myapp:latest
```

### 6.4 Topology Spread Constraints

```yaml
# topology-spread.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: distributed-app
spec:
  replicas: 6
  selector:
    matchLabels:
      app: distributed
  template:
    metadata:
      labels:
        app: distributed
    spec:
      topologySpreadConstraints:
      # Even distribution across zones — survives an entire AZ outage with minimal capacity loss
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule  # Hard constraint: refuse to schedule rather than create an imbalanced deployment
        labelSelector:
          matchLabels:
            app: distributed

      # Even distribution across nodes
      - maxSkew: 1
        topologyKey: kubernetes.io/hostname
        whenUnsatisfiable: ScheduleAnyway
        labelSelector:
          matchLabels:
            app: distributed

      containers:
      - name: app
        image: myapp:latest
```

---

## 8. 연습 문제

### 연습 1: 마이크로서비스 Ingress
```yaml
# Requirements:
# - api.example.com/v1/* → api-v1-service
# - api.example.com/v2/* → api-v2-service
# - TLS enabled, HTTP→HTTPS redirect
# - Rate limiting applied

# Write Ingress
```

### 연습 2: Redis Cluster StatefulSet
```yaml
# Requirements:
# - 3-node Redis Cluster
# - 1Gi PVC for each node
# - Headless Service for inter-node communication
# - Appropriate resource limits

# Write StatefulSet
```

### 연습 3: 로그 수집 DaemonSet
```yaml
# Requirements:
# - Collect /var/log from all nodes
# - Send to Elasticsearch
# - Manage config with ConfigMap
# - Deploy on master nodes too

# Write DaemonSet
```

### 연습 4: 배치 데이터 처리 Job
```yaml
# Requirements:
# - Process 100 data items (completions: 100)
# - Process 10 at a time (parallelism: 10)
# - Retry 3 times on failure
# - Must complete within 2 hours
# - Delete 24 hours after completion

# Write Job
```

---

## 다음 단계

- [09_Helm_패키지관리](09_Helm_Package_Management.md) - Helm 차트
- [10_CI_CD_파이프라인](10_CI_CD_Pipelines.md) - 자동화 배포
- [07_Kubernetes_보안](07_Kubernetes_Security.md) - 보안 복습

## 참고 자료

- [Kubernetes Ingress](https://kubernetes.io/docs/concepts/services-networking/ingress/)
- [Gateway API](https://gateway-api.sigs.k8s.io/)
- [StatefulSets](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/)
- [Persistent Volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)
- [DaemonSet](https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/)

---

## 연습 문제

### 연습 1: StatefulSet(스테이트풀셋)으로 상태 유지 애플리케이션 배포

Deployment와 StatefulSet의 차이를 복제 데이터베이스 배포를 통해 직접 경험합니다.

1. `redis:7-alpine`을 사용하여 3개 복제본을 가진 StatefulSet 매니페스트를 작성합니다
2. 동일한 셀렉터(selector)를 가진 헤드리스 서비스(headless Service) (`clusterIP: None`)를 포함합니다
3. 두 매니페스트를 적용하고 각 Pod가 안정적이고 순서가 있는 이름(`redis-0`, `redis-1`, `redis-2`)을 갖는 것을 확인합니다
4. `redis-0`에 exec로 접속하여 키를 설정합니다: `redis-cli set mykey "hello"`
5. `redis-0`을 삭제하고 Kubernetes가 동일한 이름과 순서 번호(ordinal)로 재생성하는 것을 확인합니다
6. 재생성 후 키가 사라진 것을 확인합니다 (아직 영구 저장소 없음) — PVC 기반 StatefulSet과의 차이를 이해합니다

### 연습 2: PVC(퍼시스턴트볼륨클레임)로 영구 저장소 프로비저닝

각 복제본에 독립적인 저장소를 제공하기 위해 PersistentVolumeClaim을 StatefulSet에 연결합니다.

1. 연습 1의 StatefulSet에 1Gi 저장소를 요청하는 `volumeClaimTemplate`을 추가합니다
2. 적용 후 각 Pod가 자체 PVC를 갖는지 확인합니다: `kubectl get pvc`
3. `redis-0`에 exec로 접속하여 키를 설정한 후 Pod를 삭제합니다
4. `redis-0`이 재시작된 후 다시 접속하여 키가 여전히 존재하는지 확인합니다
5. `kubectl get pv`를 실행하여 동적으로 프로비저닝된 PersistentVolume을 확인합니다
6. StatefulSet을 삭제하고 PVC가 유지되는지 또는 삭제되는지 확인합니다

### 연습 3: Ingress(인그레스)로 Service 노출

Ingress 리소스를 구성하여 외부 HTTP 트래픽을 여러 Service로 라우팅합니다.

1. minikube에서 Nginx Ingress 컨트롤러를 활성화합니다: `minikube addons enable ingress`
2. 두 개의 Deployment와 ClusterIP Service를 생성합니다: 포트 8080의 `app-v1`과 포트 8081의 `app-v2`
3. 다음을 라우팅하는 Ingress 매니페스트를 작성합니다:
   - `/v1` 경로 → `app-v1` Service
   - `/v2` 경로 → `app-v2` Service
4. Ingress를 적용하고 IP를 확인합니다: `kubectl get ingress`
5. `/etc/hosts`에 IP를 추가합니다 (예: `192.168.x.x myapp.local`)
6. 라우팅을 테스트합니다: `curl http://myapp.local/v1` 및 `curl http://myapp.local/v2`

### 연습 4: Job(잡)과 CronJob(크론잡)으로 배치 워크로드 실행

Job과 CronJob 리소스를 사용하여 일회성 및 스케줄된 배치 작업을 실행합니다.

1. 컨테이너를 실행하고 "Batch job complete"을 출력한 후 종료 코드 0으로 종료하는 Job 매니페스트를 작성합니다
2. 적용 후 Pod가 완료될 때까지 실행되는 것을 확인합니다: `kubectl get jobs` 및 `kubectl get pods`
3. 잡 로그를 읽습니다: `kubectl logs job/my-job`
4. 동일한 작업을 매분 실행하는 CronJob을 스케줄 `*/1 * * * *`으로 작성합니다
5. 2분 후 여러 Job이 생성되었는지 확인합니다: `kubectl get jobs`
6. CronJob 스펙에 `successfulJobsHistoryLimit: 3`과 `failedJobsHistoryLimit: 1`을 설정하고 적용합니다

### 연습 5: 노드 어피니티(Node Affinity)로 Pod 배치 제어

스케줄링 규칙을 사용하여 워크로드가 실행될 노드에 영향을 줍니다.

1. minikube 노드(또는 컨트롤 플레인)에 레이블을 지정합니다: `kubectl label node minikube tier=frontend`
2. `tier=frontend` 레이블을 가진 노드에만 배치되도록 `nodeAffinity`를 사용하는 Deployment를 작성합니다
3. 적용 후 Pod가 레이블이 지정된 노드에 스케줄되었는지 확인합니다: `kubectl get pods -o wide`
4. 다른 노드에 `tier=backend` 레이블을 추가합니다 (minikube의 경우 동일 노드에서 레이블을 제거하고 재추가)
5. `tier=backend` 노드에 대한 `preferredDuringSchedulingIgnoredDuringExecution` 어피니티를 가진 두 번째 Deployment를 생성합니다
6. 스케줄링 동작을 확인합니다 — 선호(preferred) 어피니티는 일치하는 노드가 없어도 스케줄링을 차단하지 않습니다

---

[← 이전: Kubernetes 보안](07_Kubernetes_Security.md) | [다음: Helm 패키지 관리 →](09_Helm_Package_Management.md) | [목차](00_Overview.md)

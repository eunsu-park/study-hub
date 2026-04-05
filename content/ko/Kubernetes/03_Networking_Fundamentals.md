# 03. 네트워킹 기초

**이전**: [워크로드 리소스](./02_Workload_Resources.md) | **다음**: [스토리지와 영속성](./04_Storage_and_Persistence.md)

## 학습 목표
- 쿠버네티스 네트워킹 모델과 기본 요구사항을 이해한다
- 서비스 유형(ClusterIP, NodePort, LoadBalancer, ExternalName)을 구성하고 구별한다
- CoreDNS가 클러스터 내에서 서비스 디스커버리를 제공하는 방법을 설명한다
- kube-proxy 모드(iptables, IPVS, nftables)를 비교하고 그 트레이드오프를 이해한다
- 표준 도구와 기법을 사용하여 네트워킹 이슈를 디버깅한다

---

네트워킹은 쿠버네티스의 연결 조직입니다. 모든 파드(Pod)는 자체 IP 주소를 받고,
모든 서비스(Service)는 안정적인 가상 IP를 받으며, DNS가 이 모든 것을 연결합니다.
네트워킹이 종종 부차적인 고려사항이었던 전통적 인프라와 달리, 쿠버네티스는
네트워킹을 설계의 핵심에 둡니다. 이 레슨에서는 네트워킹 모델, 서비스 추상화,
DNS, 프록시 모드, 디버깅 기법을 다룹니다.

## 목차
1. [쿠버네티스 네트워킹 모델](#1-쿠버네티스-네트워킹-모델)
2. [서비스 유형](#2-서비스-유형)
3. [쿠버네티스의 DNS (CoreDNS)](#3-쿠버네티스의-dns-coredns)
4. [kube-proxy 모드](#4-kube-proxy-모드)
5. [엔드포인트와 엔드포인트슬라이스](#5-엔드포인트와-엔드포인트슬라이스)
6. [서비스 토폴로지와 트래픽 정책](#6-서비스-토폴로지와-트래픽-정책)
7. [헤드리스 서비스](#7-헤드리스-서비스)
8. [네트워크 디버깅](#8-네트워크-디버깅)
9. [연습문제](#연습문제)

---

## 1. 쿠버네티스 네트워킹 모델

쿠버네티스는 세 가지 기본 네트워킹 요구사항을 부과합니다:

1. **파드 간(Pod-to-Pod)**: 모든 파드가 NAT 없이 다른 모든 파드와 통신 가능
2. **파드-서비스 간(Pod-to-Service)**: 파드가 안정적인 가상 IP를 통해 서비스에 접근
3. **외부-서비스 간(External-to-Service)**: 외부 트래픽이 NodePort, LoadBalancer, 또는
   인그레스(Ingress)를 통해 서비스에 도달

### 1.1 파드 네트워킹

각 파드는 클러스터의 파드 CIDR 범위에서 고유한 IP 주소를 받습니다. 같은 파드 내의
컨테이너는 네트워크 네임스페이스를 공유하고 `localhost`를 통해 통신합니다.

```
┌──────────────────────────────────────┐
│  Node 1 (10.0.1.0/24)               │
│  ┌──────────┐  ┌──────────┐         │
│  │ Pod A     │  │ Pod B     │        │
│  │ 10.244.1.5│  │ 10.244.1.6│        │
│  │ ┌──┐ ┌──┐│  │ ┌──┐      │        │
│  │ │C1│ │C2││  │ │C1│      │        │
│  │ └──┘ └──┘│  │ └──┘      │        │
│  └──────────┘  └──────────┘         │
│        veth          veth            │
│         │             │              │
│     ┌───┴─────────────┴───┐         │
│     │       cbr0 / cni0   │         │
│     └──────────┬──────────┘         │
│                │                     │
└────────────────┼─────────────────────┘
                 │
          ┌──────┴──────┐
          │   Network   │
          │   Fabric    │
          └──────┬──────┘
                 │
┌────────────────┼─────────────────────┐
│  Node 2 (10.0.2.0/24)               │
│     ┌──────────┴──────────┐         │
│     │       cbr0 / cni0   │         │
│     └───┬─────────────┬───┘         │
│        veth          veth            │
│  ┌──────────┐  ┌──────────┐         │
│  │ Pod C     │  │ Pod D     │        │
│  │ 10.244.2.3│  │ 10.244.2.4│        │
│  └──────────┘  └──────────┘         │
└──────────────────────────────────────┘
```

### 1.2 CNI (컨테이너 네트워크 인터페이스)

쿠버네티스는 파드 네트워킹을 CNI 플러그인에 위임합니다. kubelet이 파드 생성 시
CNI 플러그인을 호출하여 네트워킹을 설정합니다.

주요 CNI 플러그인:

| 플러그인 | 접근 방식 | 핵심 기능 |
|--------|----------|-------------|
| Calico | L3 라우팅 (BGP) | 네트워크폴리시(NetworkPolicy) 적용, 고성능 |
| Cilium | eBPF 기반 | L7 가시성, iptables 의존성 없음 |
| Flannel | 오버레이 (VXLAN) | 간단한 설정, 제한된 기능 |
| Weave | 오버레이 (메시) | 내장 암호화 |
| AWS VPC CNI | 네이티브 VPC IP | 파드가 실제 VPC IP를 받음 |

```bash
# 설치된 CNI 확인
ls /etc/cni/net.d/

# minikube에서 CNI 설정 확인
minikube ssh -- cat /etc/cni/net.d/*.conf

# 파드 CIDR 할당 확인
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}: {.spec.podCIDR}{"\n"}{end}'
```

### 1.3 네트워크 네임스페이스

각 파드는 자체 Linux 네트워크 네임스페이스에서 실행됩니다. pause 컨테이너(인프라
컨테이너)가 네임스페이스를 유지하고, 다른 모든 컨테이너가 이에 참여합니다.

```bash
# 노드에서 네트워크 네임스페이스 나열
minikube ssh -- sudo ip netns list

# 파드의 네트워크 네임스페이스 검사
POD_ID=$(minikube ssh -- sudo crictl pods --name simple-pod -q)
minikube ssh -- sudo crictl inspectp $POD_ID | python3 -c "
import json, sys
data = json.load(sys.stdin)
print('Network namespace:', data['info']['runtimeSpec']['linux']['namespaces'])
"
```

---

## 2. 서비스 유형

서비스(Services)는 파드 집합에 대한 안정적인 네트워킹을 제공합니다. 고정된 가상
IP(ClusterIP)를 할당하여 파드 IP의 변동성을 추상화합니다.

### 2.1 ClusterIP (기본)

클러스터 내부 IP에서 서비스를 노출합니다. 클러스터 내에서만 접근 가능합니다.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: backend-svc
spec:
  type: ClusterIP          # 기본값; 생략 가능
  selector:
    app: backend
  ports:
    - name: http
      port: 80             # 서비스 포트 (클라이언트가 연결하는 포트)
      targetPort: 8080     # 컨테이너 포트 (트래픽이 전달되는 포트)
      protocol: TCP
    - name: grpc
      port: 9090
      targetPort: 9090
      protocol: TCP
```

```bash
# 생성 및 확인
kubectl apply -f backend-svc.yaml
kubectl get svc backend-svc

# 출력:
# NAME          TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)          AGE
# backend-svc   ClusterIP   10.96.142.15   <none>        80/TCP,9090/TCP  5s

# 클러스터 내에서 테스트
kubectl run curl-test --rm -it --image=curlimages/curl:8.5.0 --restart=Never -- \
  curl -s http://backend-svc.default.svc.cluster.local/health
```

### 2.2 NodePort

각 노드의 IP에서 정적 포트(30000-32767)로 서비스를 노출합니다.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: web-nodeport
spec:
  type: NodePort
  selector:
    app: web
  ports:
    - port: 80
      targetPort: 8080
      nodePort: 30080       # 선택 사항; 생략하면 자동 할당 (30000-32767)
      protocol: TCP
```

```bash
# 노드 IP를 통해 접근
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
curl http://$NODE_IP:30080

# minikube에서
minikube service web-nodeport --url
```

트래픽 흐름: `클라이언트 → NodeIP:NodePort → ClusterIP:Port → PodIP:TargetPort`

### 2.3 LoadBalancer

외부 로드 밸런서를 프로비저닝합니다 (클라우드 공급자 필요).

```yaml
apiVersion: v1
kind: Service
metadata:
  name: web-lb
  annotations:
    # 클라우드 특화 어노테이션
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-scheme: "internet-facing"
spec:
  type: LoadBalancer
  selector:
    app: web
  ports:
    - port: 443
      targetPort: 8443
      protocol: TCP
  # 선택 사항: 소스 IP 제한
  loadBalancerSourceRanges:
    - "203.0.113.0/24"
    - "198.51.100.0/24"
```

```bash
# 외부 IP 할당 대기
kubectl get svc web-lb -w

# 출력 (클라우드):
# NAME     TYPE           CLUSTER-IP     EXTERNAL-IP       PORT(S)         AGE
# web-lb   LoadBalancer   10.96.45.123   a1b2c3.elb.aws    443:31234/TCP   60s

# minikube에서는 터널 사용
minikube tunnel
# 할당된 외부 IP를 통해 접근
```

### 2.4 ExternalName

서비스를 외부 DNS 이름에 매핑합니다 (CNAME 레코드). 프록시가 관여하지 않습니다.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: external-db
spec:
  type: ExternalName
  externalName: db.example.com
  # 셀렉터, 포트 불필요
```

```bash
# 파드에서 external-db를 해석하면 db.example.com의 CNAME을 반환
kubectl run dns-test --rm -it --image=busybox:1.36 --restart=Never -- \
  nslookup external-db.default.svc.cluster.local

# 출력:
# external-db.default.svc.cluster.local  canonical name = db.example.com
```

사용 사례:
- 마이그레이션 중 외부 데이터베이스 참조
- 외부 서비스 엔드포인트를 쿠버네티스 이름 뒤에 추상화
- 외부에서 인클러스터 서비스로의 점진적 마이그레이션

### 2.5 서비스 비교

| 유형 | 접근 범위 | 외부 IP | 사용 사례 |
|------|-------------|-------------|----------|
| ClusterIP | 클러스터 내부만 | 없음 | 내부 마이크로서비스 |
| NodePort | 노드 IP 통해 외부 | 노드 IP | 개발, 온프레미스 |
| LoadBalancer | LB 통해 외부 | 클라우드 LB IP | 프로덕션 클라우드 |
| ExternalName | DNS 별칭 | N/A | 외부 서비스 참조 |

---

## 3. 쿠버네티스의 DNS (CoreDNS)

### 3.1 CoreDNS 아키텍처

CoreDNS는 쿠버네티스의 기본 DNS 서버입니다. `kube-system` 네임스페이스에서
디플로이먼트로 실행되며 서비스와 파드에 대한 DNS 레코드를 제공합니다.

```bash
# CoreDNS 디플로이먼트 확인
kubectl -n kube-system get deployment coredns

# CoreDNS 설정 확인
kubectl -n kube-system get configmap coredns -o yaml
```

기본 CoreDNS Corefile:

```
.:53 {
    errors
    health {
       lameduck 5s
    }
    ready
    kubernetes cluster.local in-addr.arpa ip6.arpa {
       pods insecure
       fallthrough in-addr.arpa ip6.arpa
       ttl 30
    }
    prometheus :9153
    forward . /etc/resolv.conf {
       max_concurrent 1000
    }
    cache 30
    loop
    reload
    loadbalance
}
```

### 3.2 DNS 레코드 형식

서비스는 다음 형식의 DNS 레코드를 받습니다:

```
<service-name>.<namespace>.svc.<cluster-domain>
```

예시:
```
# default 네임스페이스의 서비스
backend-svc.default.svc.cluster.local

# production 네임스페이스의 서비스
api-gateway.production.svc.cluster.local

# 축약형 (같은 네임스페이스 내)
backend-svc                              # 같은 네임스페이스
backend-svc.production                   # 다른 네임스페이스
backend-svc.production.svc               # 명시적 svc
backend-svc.production.svc.cluster.local # 정규화된 도메인 이름
```

### 3.3 서비스 유형별 DNS 레코드

**ClusterIP 서비스**:
```
# A 레코드
backend-svc.default.svc.cluster.local → 10.96.142.15

# SRV 레코드 (포트 검색용)
_http._tcp.backend-svc.default.svc.cluster.local → 0 100 80 backend-svc.default.svc.cluster.local
```

**헤드리스 서비스(Headless Service)** (ClusterIP: None):
```
# A 레코드가 파드 IP를 직접 반환
postgres-headless.default.svc.cluster.local → 10.244.1.5, 10.244.1.6, 10.244.2.3

# 개별 파드 DNS (스테이트풀셋만)
postgres-0.postgres-headless.default.svc.cluster.local → 10.244.1.5
postgres-1.postgres-headless.default.svc.cluster.local → 10.244.1.6
```

### 3.4 파드 DNS 설정

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: custom-dns
spec:
  dnsPolicy: None          # 커스텀 DNS 설정 사용
  dnsConfig:
    nameservers:
      - 8.8.8.8
      - 8.8.4.4
    searches:
      - default.svc.cluster.local
      - svc.cluster.local
      - cluster.local
    options:
      - name: ndots
        value: "5"
      - name: timeout
        value: "2"
  containers:
    - name: app
      image: nginx:1.25
```

DNS 정책:

| 정책 | 동작 |
|--------|----------|
| ClusterFirst | 클러스터 DNS 사용; 업스트림으로 폴스루 (기본) |
| Default | 노드의 DNS 설정 상속 |
| None | `dnsConfig` 설정만 사용 |
| ClusterFirstWithHostNet | hostNetwork를 사용하는 파드에 대한 ClusterFirst |

### 3.5 DNS 성능 튜닝

```bash
# 파드 내에서 ndots 설정 확인
kubectl exec my-pod -- cat /etc/resolv.conf

# 출력:
# nameserver 10.96.0.10
# search default.svc.cluster.local svc.cluster.local cluster.local
# options ndots:5

# ndots:5에서 점이 5개 미만인 이름은 검색 목록 확장을 트리거함
# "api.example.com" (2개 점 < 5)은 다음을 시도:
#   1. api.example.com.default.svc.cluster.local
#   2. api.example.com.svc.cluster.local
#   3. api.example.com.cluster.local
#   4. api.example.com.                       (절대)
```

외부 DNS 쿼리를 많이 하는 애플리케이션의 경우, `ndots`를 줄이거나
FQDN(후행 점)을 사용하면 성능이 향상됩니다:

```yaml
dnsConfig:
  options:
    - name: ndots
      value: "2"    # 외부 이름에 대한 불필요한 DNS 조회 감소
```

---

## 4. kube-proxy 모드

kube-proxy는 각 노드에서 데이터 플레인 규칙을 프로그래밍하여 서비스(Service)
추상화를 구현합니다. 서비스와 엔드포인트슬라이스(EndpointSlice) 오브젝트를 감시하고
그에 따라 커널 네트워킹을 설정합니다.

### 4.1 iptables 모드 (기본)

kube-proxy는 각 서비스/엔드포인트슬라이스에 대해 iptables 규칙을 생성합니다.

```
Client Pod → iptables DNAT → Backend Pod
                 │
    ┌────────────┴────────────┐
    │  KUBE-SERVICES chain    │
    │  Match: dest=ClusterIP  │
    │  Jump: KUBE-SVC-xxx     │
    └────────────┬────────────┘
                 │
    ┌────────────┴────────────┐
    │  KUBE-SVC-xxx chain     │
    │  Random: 33% → EP1     │
    │  Random: 50% → EP2     │
    │  Default:   → EP3      │
    └─────────────────────────┘
```

```bash
# 서비스에 대한 iptables 규칙 확인
minikube ssh -- sudo iptables -t nat -L KUBE-SERVICES -n | grep backend-svc

# 특정 서비스 체인 확인
minikube ssh -- sudo iptables -t nat -L KUBE-SVC-XXXXXXXX -n

# iptables 규칙 수 세기 (서비스 수에 따라 증가)
minikube ssh -- sudo iptables -t nat -L | wc -l
```

특성:
- **장점**: 안정적, 잘 테스트됨, 어디서나 동작
- **단점**: O(n) 규칙 평가; 수천 개의 서비스에서 느림
- **로드 밸런싱**: 동일 확률의 랜덤
- **연결 추적**: 설정된 연결에 conntrack 사용

### 4.2 IPVS 모드

L4 로드 밸런싱을 위한 Linux IPVS(IP Virtual Server) 커널 모듈을 사용합니다.

```bash
# kube-proxy에서 IPVS 모드 활성화
kubectl -n kube-system edit configmap kube-proxy
# mode: "ipvs"로 설정
# 그 다음 kube-proxy 파드 재시작

# 또는 minikube에서
minikube start --extra-config=kube-proxy.mode=ipvs
```

```bash
# IPVS 규칙 확인
minikube ssh -- sudo ipvsadm -Ln

# 출력 예시:
# TCP  10.96.142.15:80 rr
#   -> 10.244.1.5:8080    Masq    1      0       0
#   -> 10.244.1.6:8080    Masq    1      0       0
#   -> 10.244.2.3:8080    Masq    1      0       0
```

IPVS 스케줄링 알고리즘:

| 알고리즘 | 플래그 | 설명 |
|-----------|------|-------------|
| 라운드 로빈(Round Robin) | rr | 균등 분배 |
| 최소 연결(Least Connections) | lc | 부하가 적은 백엔드 선호 |
| 대상 해싱(Destination Hashing) | dh | 대상별 일관된 해싱 |
| 소스 해싱(Source Hashing) | sh | 같은 소스 → 같은 백엔드 |
| 최단 예상 지연(Shortest Expected Delay) | sed | 가중 최소 연결 |

특성:
- **장점**: O(1) 조회, 다양한 LB 알고리즘, 더 나은 확장성
- **단점**: IPVS 커널 모듈 필요, 디버깅이 약간 더 복잡
- **규모**: 10,000개 이상의 서비스를 효율적으로 처리

### 4.3 nftables 모드 (v1.29+)

iptables의 후속인 nftables를 사용합니다:

```bash
# nftables 모드 활성화
# kube-proxy 컨피그맵에서 mode: "nftables"로 설정

# nftables 규칙 확인
minikube ssh -- sudo nft list ruleset | grep kube
```

특성:
- **장점**: iptables보다 나은 성능, 원자적 규칙 업데이트
- **단점**: 더 새로움, 실전 검증이 적음
- **호환성**: Linux 커널 5.13+ 필요

### 4.4 모드 비교

| 기능 | iptables | IPVS | nftables |
|---------|----------|------|----------|
| 조회 복잡도 | O(n) | O(1) | O(1) |
| LB 알고리즘 | 랜덤 | 다양 | 랜덤 |
| 최대 서비스 수 | ~5,000 | 10,000+ | 10,000+ |
| 규칙 업데이트 | 전체 교체 | 점진적 | 원자적 |
| 세션 어피니티 | 예 | 예 | 예 |
| 커널 요구사항 | 아무거나 | IPVS 모듈 | 5.13+ |

---

## 5. 엔드포인트와 엔드포인트슬라이스

### 5.1 엔드포인트(Endpoints) (레거시)

엔드포인트 오브젝트는 서비스 셀렉터와 일치하는 파드의 IP 주소를 포함합니다.

```bash
# 서비스의 엔드포인트 확인
kubectl get endpoints backend-svc

# 출력:
# NAME          ENDPOINTS                                      AGE
# backend-svc   10.244.1.5:8080,10.244.1.6:8080,10.244.2.3:8080   5m

# 상세 확인
kubectl describe endpoints backend-svc
```

엔드포인트의 한계:
- 서비스당 하나의 엔드포인트 오브젝트, 모든 파드 IP 포함
- ~1,000개 이상의 엔드포인트에서 잘 확장되지 않음
- 변경 시 전체 오브젝트를 전송해야 함

### 5.2 엔드포인트슬라이스(EndpointSlices) (현대)

엔드포인트슬라이스는 엔드포인트를 더 작은 청크로 분할하여 확장성 문제를
해결합니다 (기본: 슬라이스당 100개 엔드포인트).

```bash
# 엔드포인트슬라이스 확인
kubectl get endpointslices -l kubernetes.io/service-name=backend-svc

# 출력:
# NAME                  ADDRESSTYPE   PORTS   ENDPOINTS                  AGE
# backend-svc-abc12     IPv4          8080    10.244.1.5,10.244.1.6,...  5m

# 상세 확인
kubectl describe endpointslice backend-svc-abc12
```

```yaml
# 엔드포인트슬라이스 구조 (엔드포인트 컨트롤러가 자동 관리)
apiVersion: discovery.k8s.io/v1
kind: EndpointSlice
metadata:
  name: backend-svc-abc12
  labels:
    kubernetes.io/service-name: backend-svc
addressType: IPv4
ports:
  - name: http
    protocol: TCP
    port: 8080
endpoints:
  - addresses:
      - "10.244.1.5"
    conditions:
      ready: true
      serving: true
      terminating: false
    nodeName: node-1
    zone: us-east-1a
  - addresses:
      - "10.244.1.6"
    conditions:
      ready: true
      serving: true
      terminating: false
    nodeName: node-1
    zone: us-east-1a
```

### 5.3 수동 엔드포인트

셀렉터가 없는 서비스(외부 리소스를 가리키는)의 경우:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: external-database
spec:
  # 셀렉터 없음
  ports:
    - port: 5432
      targetPort: 5432
---
apiVersion: discovery.k8s.io/v1
kind: EndpointSlice
metadata:
  name: external-database-1
  labels:
    kubernetes.io/service-name: external-database
addressType: IPv4
ports:
  - port: 5432
    protocol: TCP
endpoints:
  - addresses:
      - "192.168.1.100"   # 외부 데이터베이스 IP
  - addresses:
      - "192.168.1.101"   # 레플리카
```

---

## 6. 서비스 토폴로지와 트래픽 정책

### 6.1 내부 트래픽 정책(Internal Traffic Policy)

클러스터 내 파드의 트래픽이 서비스 엔드포인트에 도달하는 방식을 제어합니다.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: backend-svc
spec:
  selector:
    app: backend
  ports:
    - port: 80
      targetPort: 8080
  internalTrafficPolicy: Local    # 같은 노드의 파드로만 라우팅
  # internalTrafficPolicy: Cluster  # 어떤 파드로든 라우팅 (기본)
```

`Local` 정책:
- 같은 노드의 엔드포인트로만 라우팅
- 지연 시간과 노드 간 트래픽 감소
- 위험: 로컬 엔드포인트가 없으면 트래픽이 드롭됨 (503)

### 6.2 외부 트래픽 정책(External Traffic Policy)

외부 소스의 트래픽이 서비스 엔드포인트에 도달하는 방식을 제어합니다.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: web-lb
spec:
  type: LoadBalancer
  selector:
    app: web
  ports:
    - port: 80
      targetPort: 8080
  externalTrafficPolicy: Local    # 클라이언트 소스 IP 보존
  # externalTrafficPolicy: Cluster  # SNAT 가능, 소스 IP 손실 (기본)
```

| 정책 | 소스 IP 보존 | 부하 분배 | 실패 모드 |
|--------|--------------------|-------------------|--------------|
| Cluster | 아니오 (SNAT) | 모든 파드에 균등 | 항상 동작 |
| Local | 예 | 로컬 엔드포인트만 | 로컬 파드 없으면 503 |

### 6.3 토폴로지 인식 라우팅(Topology Aware Routing) (v1.27+)

같은 존의 엔드포인트로 라우팅을 선호합니다:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: backend-svc
  annotations:
    service.kubernetes.io/topology-mode: Auto
spec:
  selector:
    app: backend
  ports:
    - port: 80
      targetPort: 8080
```

활성화되면 kube-proxy가 엔드포인트슬라이스의 힌트를 프로그래밍하여 같은 존의
엔드포인트를 선호하고, 존 간 트래픽과 지연 시간을 줄입니다.

```bash
# 토폴로지 힌트가 설정되었는지 확인
kubectl get endpointslice -l kubernetes.io/service-name=backend-svc -o yaml | grep -A 5 hints
```

---

## 7. 헤드리스 서비스

헤드리스 서비스(Headless Service)는 `clusterIP: None`을 가집니다. 가상 IP 대신
DNS가 파드 IP를 직접 반환합니다.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: app-headless
spec:
  clusterIP: None
  selector:
    app: my-app
  ports:
    - port: 80
      targetPort: 8080
```

### 7.1 DNS 동작

```bash
# 일반 서비스: 단일 ClusterIP 반환
kubectl run dns-test --rm -it --image=busybox:1.36 --restart=Never -- \
  nslookup backend-svc
# Address: 10.96.142.15

# 헤드리스 서비스: 모든 파드 IP 반환
kubectl run dns-test --rm -it --image=busybox:1.36 --restart=Never -- \
  nslookup app-headless
# Address: 10.244.1.5
# Address: 10.244.1.6
# Address: 10.244.2.3
```

### 7.2 사용 사례

| 사용 사례 | 왜 헤드리스인가? |
|----------|--------------|
| 스테이트풀셋(StatefulSets) | 파드별 안정적 DNS (pod-0.svc, pod-1.svc) |
| 클라이언트 측 로드 밸런싱 | 애플리케이션이 연결할 파드를 선택 |
| 서비스 메시(Service Mesh) | 사이드카 프록시가 라우팅 처리, kube-proxy가 아님 |
| 데이터베이스 클러스터 | 특정 레플리카에 주소 지정 필요 |
| gRPC | 연결 간 클라이언트 측 로드 밸런싱 |

### 7.3 스테이트풀셋과 헤드리스 서비스

```yaml
apiVersion: v1
kind: Service
metadata:
  name: cassandra-headless
spec:
  clusterIP: None
  selector:
    app: cassandra
  ports:
    - port: 9042
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: cassandra
spec:
  serviceName: cassandra-headless
  replicas: 3
  selector:
    matchLabels:
      app: cassandra
  template:
    metadata:
      labels:
        app: cassandra
    spec:
      containers:
        - name: cassandra
          image: cassandra:4.1
          ports:
            - containerPort: 9042
          env:
            - name: CASSANDRA_SEEDS
              # 헤드리스 DNS를 사용하여 시드 노드 발견
              value: "cassandra-0.cassandra-headless.default.svc.cluster.local"
```

각 파드는 DNS 항목을 받습니다:
```
cassandra-0.cassandra-headless.default.svc.cluster.local
cassandra-1.cassandra-headless.default.svc.cluster.local
cassandra-2.cassandra-headless.default.svc.cluster.local
```

---

## 8. 네트워크 디버깅

### 8.1 디버깅 도구

```bash
# 네트워킹 도구가 포함된 디버그 파드 배포
kubectl run netshoot --rm -it --image=nicolaka/netshoot --restart=Never -- /bin/bash

# 디버그 파드 내에서:
# DNS 해석
nslookup backend-svc.default.svc.cluster.local
dig backend-svc.default.svc.cluster.local +short

# HTTP 연결
curl -v http://backend-svc/health

# TCP 연결
nc -zv backend-svc 80

# 경로 추적
traceroute backend-svc

# DNS 응답 시간
dig @10.96.0.10 backend-svc.default.svc.cluster.local +stats

# /etc/resolv.conf 확인
cat /etc/resolv.conf
```

### 8.2 일반적인 네트워킹 이슈

#### 서비스에 접근할 수 없는 경우

```bash
# 1. 서비스가 존재하고 엔드포인트가 있는지 확인
kubectl get svc backend-svc
kubectl get endpoints backend-svc

# 2. 엔드포인트가 비어있으면 셀렉터가 일치하는지 확인
kubectl get svc backend-svc -o jsonpath='{.spec.selector}'
kubectl get pods -l app=backend

# 3. 파드 준비 상태 확인 (준비되지 않은 파드는 엔드포인트에서 제거됨)
kubectl get pods -l app=backend -o wide

# 4. kube-proxy가 실행 중인지 확인
kubectl -n kube-system get pods -l k8s-app=kube-proxy
```

#### DNS 해석 실패

```bash
# 1. CoreDNS 파드가 실행 중인지 확인
kubectl -n kube-system get pods -l k8s-app=kube-dns

# 2. CoreDNS 로그 확인
kubectl -n kube-system logs -l k8s-app=kube-dns --tail=20

# 3. DNS 직접 테스트
kubectl run dns-debug --rm -it --image=busybox:1.36 --restart=Never -- \
  nslookup kubernetes.default.svc.cluster.local

# 4. CoreDNS 컨피그맵 확인
kubectl -n kube-system get configmap coredns -o yaml
```

#### 네임스페이스 간 통신

```bash
# 파드는 항상 FQDN을 사용하여 다른 네임스페이스의 서비스에 접근 가능
kubectl run curl-test --rm -it --image=curlimages/curl:8.5.0 --restart=Never -- \
  curl -s http://backend-svc.production.svc.cluster.local/health

# 네트워크폴리시가 네임스페이스 간 트래픽을 차단할 수 있음
kubectl get networkpolicies --all-namespaces
```

### 8.3 네트워크폴리시(NetworkPolicy)

네트워크폴리시는 네트워크 수준에서 파드 간 트래픽을 제어합니다 (Calico나
Cilium 같은 네트워크폴리시를 지원하는 CNI 필요).

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: backend-policy
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
    - Ingress
    - Egress

  ingress:
    # 같은 네임스페이스의 프론트엔드 파드에서의 트래픽 허용
    - from:
        - podSelector:
            matchLabels:
              app: frontend
      ports:
        - protocol: TCP
          port: 8080

    # 모니터링 네임스페이스에서의 트래픽 허용
    - from:
        - namespaceSelector:
            matchLabels:
              name: monitoring
      ports:
        - protocol: TCP
          port: 9090

  egress:
    # DNS 허용
    - to:
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53
        - protocol: TCP
          port: 53

    # 데이터베이스 연결 허용
    - to:
        - podSelector:
            matchLabels:
              app: postgres
      ports:
        - protocol: TCP
          port: 5432
```

```bash
# 네트워크폴리시 나열
kubectl get networkpolicies -n production

# 정책 적용 후 연결 테스트
kubectl -n production exec frontend-pod -- curl -s http://backend-svc:8080/health
# 성공해야 함 (인그레스 규칙에 의해 허용됨)

kubectl -n production exec other-pod -- curl -s http://backend-svc:8080/health
# 실패해야 함 (허용된 인그레스에 없음)
```

### 8.4 패킷 캡처

```bash
# 특정 파드의 네트워크 인터페이스에서 패킷 캡처
# 먼저 파드의 노드와 컨테이너 ID 찾기
NODE=$(kubectl get pod my-pod -o jsonpath='{.spec.nodeName}')
CONTAINER_ID=$(kubectl get pod my-pod -o jsonpath='{.status.containerStatuses[0].containerID}' | cut -d/ -f3)

# nsenter를 사용하여 파드의 네트워크 네임스페이스에 진입 (노드에서)
# 그런 다음 tcpdump 사용
kubectl debug node/$NODE -it --image=nicolaka/netshoot -- \
  nsenter -t $(crictl inspect $CONTAINER_ID | jq .info.pid) -n \
  tcpdump -i eth0 -c 20 port 8080
```

### 8.5 서비스 연결 매트릭스

```bash
# 빠른 연결 테스트 스크립트
cat <<'SCRIPT' > /tmp/test-connectivity.sh
#!/bin/bash
SERVICES=("frontend" "backend" "database")
for src in "${SERVICES[@]}"; do
  for dst in "${SERVICES[@]}"; do
    if [ "$src" != "$dst" ]; then
      result=$(kubectl exec deploy/$src -- curl -s -o /dev/null -w "%{http_code}" http://$dst/ 2>/dev/null)
      echo "$src -> $dst: $result"
    fi
  done
done
SCRIPT
bash /tmp/test-connectivity.sh
```

---

## 연습문제

### 연습문제 1: 서비스 디스커버리

3개의 레플리카를 가진 디플로이먼트와 ClusterIP 서비스를 생성합니다. 디버그 파드에서
DNS 해석이 ClusterIP를 반환하는지, HTTP 요청이 모든 파드에 로드 밸런싱되는지 확인하세요.

<details>
<summary>정답 보기</summary>

```yaml
# /tmp/svc-discovery.yaml로 저장
apiVersion: apps/v1
kind: Deployment
metadata:
  name: whoami
spec:
  replicas: 3
  selector:
    matchLabels:
      app: whoami
  template:
    metadata:
      labels:
        app: whoami
    spec:
      containers:
        - name: whoami
          image: traefik/whoami:v1.10
          ports:
            - containerPort: 80
          resources:
            requests:
              cpu: "50m"
              memory: "32Mi"
---
apiVersion: v1
kind: Service
metadata:
  name: whoami-svc
spec:
  selector:
    app: whoami
  ports:
    - port: 80
      targetPort: 80
```

```bash
kubectl apply -f /tmp/svc-discovery.yaml
kubectl wait --for=condition=Available deployment/whoami --timeout=60s

# DNS 해석 확인
kubectl run dns-test --rm -it --image=busybox:1.36 --restart=Never -- \
  nslookup whoami-svc.default.svc.cluster.local
# ClusterIP를 반환해야 함 (예: 10.96.xxx.xxx)

# 로드 밸런싱 확인 (10번 요청, 다른 파드 호스트네임 확인)
kubectl run curl-test --rm -it --image=curlimages/curl:8.5.0 --restart=Never -- \
  sh -c 'for i in $(seq 1 10); do curl -s http://whoami-svc/ | grep Hostname; done'
# 다른 호스트네임(파드 이름)이 표시되어 로드 밸런싱을 보여줘야 함

# 엔드포인트 확인
kubectl get endpoints whoami-svc
# 3개의 파드 IP가 표시되어야 함

# 정리
kubectl delete -f /tmp/svc-discovery.yaml
```

</details>

### 연습문제 2: NodePort 서비스

웹 애플리케이션을 포트 30080의 NodePort 서비스로 노출합니다. 노드의 IP 주소를
사용하여 클러스터 외부에서 접근하세요.

<details>
<summary>정답 보기</summary>

```yaml
# /tmp/nodeport-exercise.yaml로 저장
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
    spec:
      containers:
        - name: nginx
          image: nginx:1.25
          ports:
            - containerPort: 80
          resources:
            requests:
              cpu: "50m"
              memory: "64Mi"
---
apiVersion: v1
kind: Service
metadata:
  name: web-app-nodeport
spec:
  type: NodePort
  selector:
    app: web-app
  ports:
    - port: 80
      targetPort: 80
      nodePort: 30080
```

```bash
kubectl apply -f /tmp/nodeport-exercise.yaml
kubectl wait --for=condition=Available deployment/web-app --timeout=60s

# 노드 IP 가져오기
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
echo "Node IP: $NODE_IP"

# NodePort를 통해 접근 (minikube에서)
minikube service web-app-nodeport --url
# 또는 직접:
curl http://$NODE_IP:30080

# 서비스 확인
kubectl get svc web-app-nodeport
# NAME               TYPE       CLUSTER-IP     EXTERNAL-IP   PORT(S)        AGE
# web-app-nodeport   NodePort   10.96.x.x      <none>        80:30080/TCP   30s

# 엔드포인트 확인
kubectl get endpoints web-app-nodeport
# 2개의 파드 IP가 표시되어야 함

# 정리
kubectl delete -f /tmp/nodeport-exercise.yaml
```

</details>

### 연습문제 3: 네트워크폴리시

두 개의 네임스페이스(`frontend`와 `backend`)를 생성합니다. 각각에 앱을 배포합니다.
frontend 네임스페이스만 포트 8080에서 backend 서비스에 접근할 수 있도록 하는
네트워크폴리시를 생성하세요.

<details>
<summary>정답 보기</summary>

```bash
# 레이블이 있는 네임스페이스 생성
kubectl create namespace frontend
kubectl label namespace frontend name=frontend
kubectl create namespace backend
kubectl label namespace backend name=backend
```

```yaml
# /tmp/netpol-exercise.yaml로 저장
# 백엔드 디플로이먼트와 서비스
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
  namespace: backend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: api-server
  template:
    metadata:
      labels:
        app: api-server
    spec:
      containers:
        - name: whoami
          image: traefik/whoami:v1.10
          ports:
            - containerPort: 80
          resources:
            requests:
              cpu: "50m"
              memory: "32Mi"
---
apiVersion: v1
kind: Service
metadata:
  name: api-server
  namespace: backend
spec:
  selector:
    app: api-server
  ports:
    - port: 8080
      targetPort: 80
---
# 네트워크폴리시: frontend 네임스페이스에서만 허용
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend-only
  namespace: backend
spec:
  podSelector:
    matchLabels:
      app: api-server
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: frontend
      ports:
        - protocol: TCP
          port: 80
```

```bash
kubectl apply -f /tmp/netpol-exercise.yaml

# frontend 네임스페이스에서 테스트 (성공해야 함)
kubectl -n frontend run curl-test --rm -it --image=curlimages/curl:8.5.0 --restart=Never -- \
  curl -s --max-time 5 http://api-server.backend.svc.cluster.local:8080/
# whoami 응답을 반환해야 함

# default 네임스페이스에서 테스트 (실패/타임아웃이어야 함)
kubectl run curl-test --rm -it --image=curlimages/curl:8.5.0 --restart=Never -- \
  curl -s --max-time 5 http://api-server.backend.svc.cluster.local:8080/
# 타임아웃되어야 함 (네트워크폴리시에 의해 차단됨)

# 네트워크폴리시 확인
kubectl -n backend get networkpolicy allow-frontend-only

# 정리
kubectl delete namespace frontend backend
```

</details>

### 연습문제 4: DNS를 가진 헤드리스 서비스

헤드리스 서비스와 스테이트풀셋을 생성합니다. DNS가 개별 파드 IP를 반환하고
각 파드가 안정적인 DNS 이름을 가지는 것을 확인하세요.

<details>
<summary>정답 보기</summary>

```yaml
# /tmp/headless-exercise.yaml로 저장
apiVersion: v1
kind: Service
metadata:
  name: web-headless
spec:
  clusterIP: None
  selector:
    app: web-sts
  ports:
    - port: 80
      targetPort: 80
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: web-sts
spec:
  serviceName: web-headless
  replicas: 3
  selector:
    matchLabels:
      app: web-sts
  template:
    metadata:
      labels:
        app: web-sts
    spec:
      containers:
        - name: nginx
          image: nginx:1.25
          ports:
            - containerPort: 80
          resources:
            requests:
              cpu: "50m"
              memory: "64Mi"
```

```bash
kubectl apply -f /tmp/headless-exercise.yaml
kubectl rollout status statefulset/web-sts

# 헤드리스 DNS가 모든 파드 IP를 반환하는지 확인
kubectl run dns-test --rm -it --image=busybox:1.36 --restart=Never -- \
  nslookup web-headless.default.svc.cluster.local
# 3개의 파드 IP 주소를 반환해야 함 (ClusterIP 없음)

# 개별 파드 DNS 이름 확인
for i in 0 1 2; do
  echo "=== web-sts-$i ==="
  kubectl run dns-test-$i --rm -it --image=busybox:1.36 --restart=Never -- \
    nslookup web-sts-$i.web-headless.default.svc.cluster.local
done
# 각각 특정 파드의 IP로 해석되어야 함

# 파드 IP와 비교
kubectl get pods -l app=web-sts -o wide
# DNS IP가 파드 IP와 일치해야 함

# ClusterIP가 할당되지 않았는지 확인
kubectl get svc web-headless
# CLUSTER-IP가 "None"으로 표시되어야 함

# 정리
kubectl delete -f /tmp/headless-exercise.yaml
```

</details>

### 연습문제 5: DNS 디버깅

파드가 `billing` 네임스페이스의 `payment-api` 서비스에 접근할 수 없습니다.
체계적인 디버깅 과정을 통해 문제를 식별하고 해결하세요.

<details>
<summary>정답 보기</summary>

```bash
# 1단계: 서비스가 존재하는지 확인
kubectl get svc payment-api -n billing
# 없으면 서비스가 존재하지 않음

# 2단계: 네임스페이스가 존재하는지 확인
kubectl get namespace billing
# 없으면 생성

# 3단계: 테스트 환경 생성
kubectl create namespace billing

cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: payment-api
  namespace: billing
spec:
  replicas: 2
  selector:
    matchLabels:
      app: payment-api
  template:
    metadata:
      labels:
        app: payment-api
    spec:
      containers:
        - name: api
          image: traefik/whoami:v1.10
          ports:
            - containerPort: 80
          resources:
            requests:
              cpu: "50m"
              memory: "32Mi"
---
apiVersion: v1
kind: Service
metadata:
  name: payment-api
  namespace: billing
spec:
  selector:
    app: payment-api
  ports:
    - port: 80
      targetPort: 80
EOF

# 4단계: 체계적 디버깅
# 4a. CoreDNS가 실행 중인지 확인
kubectl -n kube-system get pods -l k8s-app=kube-dns

# 4b. 디버그 파드에서 DNS 해석 테스트
kubectl run debug --rm -it --image=nicolaka/netshoot --restart=Never -- \
  bash -c '
    echo "=== DNS 테스트 ==="
    nslookup payment-api.billing.svc.cluster.local

    echo "=== 연결 테스트 ==="
    curl -s --max-time 5 http://payment-api.billing.svc.cluster.local/

    echo "=== resolv.conf ==="
    cat /etc/resolv.conf

    echo "=== DNS로의 경로 ==="
    traceroute -m 3 10.96.0.10
  '

# 4c. 엔드포인트 확인
kubectl get endpoints payment-api -n billing
# 비어있으면: 셀렉터 불일치 또는 파드 미준비

# 4d. 파드 준비 상태 확인
kubectl get pods -n billing -l app=payment-api

# 4e. 트래픽을 차단하는 네트워크폴리시가 있는지 확인
kubectl get networkpolicies -n billing

# 정리
kubectl delete namespace billing
```

</details>

---

**이전**: [워크로드 리소스](./02_Workload_Resources.md) | **다음**: [스토리지와 영속성](./04_Storage_and_Persistence.md)

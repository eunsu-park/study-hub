# 08. CNI와 고급 네트워킹(CNI and Advanced Networking)

**이전**: [인그레스와 Gateway API](./07_Ingress_and_Gateway_API.md) | **다음**: [Helm과 Kustomize](./09_Helm_and_Kustomize.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 컨테이너 네트워크 인터페이스(CNI, Container Network Interface) 명세와 플러그인 수명주기를 설명할 수 있다
2. 데이터 플레인 기술(iptables, eBPF, IPVS) 전반에 걸쳐 Calico, Cilium 및 기타 CNI 플러그인을 비교할 수 있다
3. 이그레스 규칙, CIDR 블록, DNS 인식 필터링을 포함하는 고급 NetworkPolicy를 작성할 수 있다
4. eBPF 기본 원리와 Cilium이 네트워킹 및 관측 가능성(observability)에 이를 활용하는 방법을 설명할 수 있다
5. 표준 진단 도구를 사용하여 Kubernetes 네트워킹 문제를 해결할 수 있다

---

Kubernetes 네트워킹은 표면적으로는 간단해 보입니다 -- 모든 파드가 IP를 받고, 모든 서비스가 DNS 이름을 받습니다 -- 하지만 기반 구현은 CNI 플러그인, iptables/eBPF 규칙, 라우팅 테이블, 오버레이 네트워크 간의 복잡한 상호작용을 수반합니다. 이 레슨에서는 CNI 명세를 깊이 다루고, 두 가지 주요 플러그인(Calico와 Cilium)을 탐구하며, 네트워킹 프리미티브로서의 eBPF를 소개하고, 고급 NetworkPolicy 패턴, 서비스 메시(service mesh) 기본, 네트워크 트러블슈팅을 다룹니다.

> **Kubernetes 네트워크 모델:** Kubernetes는 세 가지 기본 요구사항을 부과합니다: (1) 모든 파드가 고유한 IP를 받고, (2) 파드가 NAT 없이 다른 모든 파드와 통신할 수 있으며, (3) 노드의 에이전트가 해당 노드의 모든 파드와 통신할 수 있어야 합니다. 이것이 어떻게 달성되는지는 전적으로 CNI 플러그인에 달려 있습니다.

구성에 들어가기 전에, [**이론과 원리**](#이론과-원리) 섹션을 먼저 읽어보세요. 모든 쿠버네티스 네트워크가 구현해야 하는 CNI 플러그인 계약, 데이터 플레인 기법의 네 가지 계열(오버레이, 언더레이/BGP, eBPF, IPVS), eBPF가 네트워킹에서 iptables를 대체하는 이유, 그리고 NetworkPolicy 시맨틱이 커널 규칙으로 어떻게 컴파일되는지를 다룹니다.

## 목차

- [이론과 원리](#이론과-원리)
- [1. CNI 명세](#1-cni-명세)
  - [1.1 CNI 작동 방식](#11-cni-작동-방식)
  - [1.2 CNI 플러그인 수명주기](#12-cni-플러그인-수명주기)
  - [1.3 CNI 구성](#13-cni-구성)
- [2. Calico](#2-calico)
  - [2.1 아키텍처](#21-아키텍처)
  - [2.2 데이터 플레인 모드](#22-데이터-플레인-모드)
  - [2.3 Calico NetworkPolicy 확장](#23-calico-networkpolicy-확장)
- [3. Cilium](#3-cilium)
  - [3.1 eBPF 기반 아키텍처](#31-ebpf-기반-아키텍처)
  - [3.2 설치 및 구성](#32-설치-및-구성)
  - [3.3 Cilium 네트워크 정책](#33-cilium-네트워크-정책)
  - [3.4 Hubble 관측 가능성](#34-hubble-관측-가능성)
- [4. eBPF 기본](#4-ebpf-기본)
  - [4.1 eBPF란?](#41-ebpf란)
  - [4.2 네트워킹을 위한 eBPF](#42-네트워킹을-위한-ebpf)
  - [4.3 eBPF vs iptables](#43-ebpf-vs-iptables)
- [5. 고급 NetworkPolicy](#5-고급-networkpolicy)
  - [5.1 이그레스 정책](#51-이그레스-정책)
  - [5.2 CIDR 기반 규칙](#52-cidr-기반-규칙)
  - [5.3 포트 범위 정책](#53-포트-범위-정책)
  - [5.4 DNS 인식 정책 (Cilium)](#54-dns-인식-정책-cilium)
- [6. 서비스 메시 개요](#6-서비스-메시-개요)
  - [6.1 서비스 메시란?](#61-서비스-메시란)
  - [6.2 Istio](#62-istio)
  - [6.3 Linkerd](#63-linkerd)
  - [6.4 Cilium 서비스 메시](#64-cilium-서비스-메시)
- [7. 대역폭과 QoS](#7-대역폭과-qos)
- [8. IPv4/IPv6 듀얼 스택(Dual-Stack)](#8-ipv4ipv6-듀얼-스택dual-stack)
- [9. 네트워크 트러블슈팅](#9-네트워크-트러블슈팅)
  - [9.1 진단 도구](#91-진단-도구)
  - [9.2 일반적인 문제](#92-일반적인-문제)
- [연습문제](#연습문제)

---

## 이론과 원리

3강은 쿠버네티스 네트워킹의 네 가지 불변 조건을 진술했지만 공리로 다뤘습니다. 이 레슨은 상자를 엽니다 — *누가* 그 불변 조건을 구현하고, *어떻게*, 그리고 트레이드오프는 무엇인지. 답은 **CNI 플러그인**입니다 — kubelet은 Pod에 IP를 부여하거나, 네트워크에 attach하거나, 정책을 강제하는 방법을 모릅니다. 설치된 플러그인에게 그 일을 넘깁니다. Calico, Cilium, Flannel, AWS VPC CNI, Azure CNI — 각각 동일한 CNI 계약을 다르게 구현하며, 성능·확장성·정책 표현력에 결과를 가져옵니다. 이 섹션은 계약, 네 가지 구현 계열, 그리고 데이터 플레인을 재편하고 있는 eBPF 혁명을 설명합니다.

### A. CNI 계약 — 최소한의 플러그인 인터페이스

CNI는 쿠버네티스 전용 스펙이 아닙니다 — 쿠버네티스, Mesos, podman 등이 사용하는 CNCF 프로젝트입니다. 인터페이스는 의도적으로 작습니다 — CNI 플러그인은 컨테이너 런타임이 세 명령과 JSON config로 호출하는 실행 파일일 뿐입니다:

```
ADD <network> <container-id> <netns>     # 새 컨테이너에 네트워킹 설정
DEL <network> <container-id> <netns>     # 해체
CHECK <network> <container-id> <netns>   # 설정 확인
```

kubelet이 Pod를 만들면, 런타임(containerd/CRI-O)이 구성된 CNI 플러그인의 `ADD`를 호출합니다. 플러그인의 일은 정확히:

1. 클러스터 pod CIDR(또는 구성된 IPAM 스킴)에서 IP 할당.
2. Pod의 네트워크 네임스페이스 안에 네트워크 인터페이스 생성.
3. 클러스터 나머지에 트래픽이 도달할 수 있도록 배선(라우트 테이블, 캡슐화 터널, BGP 광고, eBPF 프로그램 — 구현 선택).
4. IP와 모든 라우트를 JSON으로 런타임에 반환.

그게 전부입니다. kubelet, kube-proxy, 그리고 쿠버네티스 나머지는 네트워크가 *어떻게* 동작하는지에서 완전히 격리됩니다. 이 최소 계약이 풍부한 생태계를 가능하게 했습니다 — 새 네트워킹 모델 추가는 쿠버네티스 코어를 패치하는 것이 아니라 `ADD`/`DEL`/`CHECK`를 다루는 바이너리를 작성하는 것입니다.

트레이드오프는 일부 고급 기능(NetworkPolicy 강제, service 로드 밸런싱, 관측 가능성)이 CNI 스펙의 일부가 아니라는 것 — 플러그인 확장입니다. 따라서 실무에서 "CNI 플러그인"은 "CNI를 하는 바이너리 + 플러그인 작성자가 추가하고 싶었던 모든 것을 하는 데몬"을 의미합니다.

### B. 데이터 플레인의 네 가지 계열

노드 간 Pod-Pod 트래픽은 물리적으로 노드 A의 네트워크에서 노드 B의 네트워크로 가야 합니다. 네 가지 주류 접근:

**1. 오버레이(캡슐화) — VXLAN, Geneve.** Pod 패킷은 destination이 *노드의* IP인 UDP 패킷으로 감싸집니다. 받는 노드가 풀어 정확한 Pod에 전달합니다. 특별한 구성 없이 어떤 L3 네트워크에서도 동작 — 비용은 패킷당 약 50바이트 오버헤드와 두 배의 커널 작업. Flannel(기본 모드), Calico(VXLAN 모드), Weave가 이를 사용. **적합 대상: 시작하는 경우, 멀티 클라우드, 제한된 네트워크.**

**2. 언더레이 / BGP — Calico (BGP 모드), Cilium (BGP).** 각 노드가 자신의 Pod CIDR을 BGP로 인접 라우터(또는 직접 ToR 스위치)에 광고합니다. 패킷은 캡슐화 없이 native하게 이동합니다. 와이어 라인 성능, 그러나 L3 라우팅 패브릭의 통제가 필요 — 보통 온프레미스이거나 명시적으로 지원하는 클라우드 설정(예: AWS의 VPC CNI)에서만 가능. **적합 대상: 베어메탈, 고처리량 워크로드, 네트워크를 통제할 때.**

**3. eBPF — Cilium.** 커널 netfilter 체인의 iptables 규칙 대신, Cilium은 eBPF 프로그램을 네트워크 인터페이스에 부착합니다. 패킷은 iptables 스택에 닿기 *전에* 이 프로그램에 의해 검사·전달되며, 보통 TC ingress 훅 하나만으로. 이는 리눅스 네트워크 스택의 많은 부분을 우회하고 큰 클러스터에서 극적으로 빠릅니다 — L7 인식(HTTP/gRPC 파싱)과 풍부한 관측 가능성(Hubble)도 가능하게 합니다. **적합 대상: 큰 클러스터, 성능에 민감한 워크로드, 현대 배포판.**

**4. 클라우드 네이티브 — AWS VPC CNI, GCP Netd, Azure CNI.** 각 Pod가 실제 클라우드 VPC IP를 받습니다(AWS의 ENI 보조 IP, GCP의 alias IP). 오버레이 없음, BGP 없음 — 클라우드의 기저 SDN이 전달을 처리합니다. 한계 — 실제 IP를 더 빨리 소진하고, 클러스터 간 라우팅에 VPC 피어링이 필요. **적합 대상: 클러스터 크기가 IP 예산에 맞는 클라우드 네이티브 배포.**

선택은 사소하지 않은 결과를 가집니다 — 클러스터 메시 프라이버시를 위한 암호화 오버레이(Calico/Cilium의 WireGuard 모드), kube-proxy 대체를 위한 eBPF(iptables 전혀 없음), sub-millisecond pod-pod 지연을 위한 BGP. CNI 선택은 나중에 되돌리기 어려운 몇 안 되는 클러스터 결정 중 하나입니다.

### C. eBPF — 커널 안의 프로그램, 커널 패치가 아님

eBPF(extended Berkeley Packet Filter)는 수년간 리눅스 네트워킹에서 가장 파괴적인 기술이며, Cilium은 그 대표 쿠버네티스 구현입니다. 기본 아이디어 — 새 네트워킹 동작을 추가하기 위해 **커널을 수정**하는 대신, 커널이 잘 정의된 훅 지점(들어오는 패킷, 나가는 패킷, 시스템 호출, tracepoint)에서 실행하는 **검증된 작은 바이트코드 프로그램**을 컴파일합니다.

핵심 속성:

- **커널 내, 컨텍스트 스위치 없음.** 사용자 공간 프록시는 각 패킷이 커널↔사용자 경계를 두 번 건너야 합니다. eBPF는 패킷 데이터가 이미 있는 커널에서 실행됩니다. 이는 사용자 공간 데이터 플레인 프록시의 가장 큰 비용을 제거합니다.
- **안전성 검증.** 커널 측 verifier가 영원히 루프하거나, 잘못된 포인터를 역참조하거나, 한정된 스택 사용을 초과할 수 있는 프로그램을 거부합니다. 이것이 "당신 커널에서 임의 코드 실행"을 미친 짓이 아니게 만드는 것입니다.
- **핫 로드 가능.** 재부팅 없음, 재컴파일 없음. Cilium은 새 정책을 새 eBPF 프로그램으로 푸시합니다 — 진행 중인 옛 패킷은 완료될 때까지 옛 프로그램을 받습니다.

쿠버네티스 네트워킹에서 eBPF는 한 번에 세 가지를 대체합니다:

- **kube-proxy의 iptables**(Service VIP 재작성) — O(N) 선형 체인 대 O(1) 해시 조회.
- **NetworkPolicy의 iptables**(허용/거부 규칙) — 긴 iptables 체인이 아니라 커널 내 테이블로 컴파일되는 정책.
- iptables가 근본적으로 할 수 없는 **새 능력 추가** — L7 HTTP 인식 정책, service-to-service 식별, 투명 암호화, 깊은 관측 가능성(Hubble이 pod-pod 흐름을 실시간으로 보여줌).

이것이 "kube-proxy 대체와 함께한 Cilium"이 성능과 관측 가능성을 우선시하는 클러스터에 현대적 기본인 이유입니다. iptables는 내일 사라지지 않지만 호환 모드로 강등되고 있습니다.

### D. NetworkPolicy — 선언적 허용 목록, CNI마다 다르게 컴파일

`NetworkPolicy`는 허용된 pod-pod 및 pod-외부 트래픽의 선언적 진술입니다:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
spec:
  podSelector: { matchLabels: { app: db } }
  policyTypes: [Ingress, Egress]
  ingress:
    - from:
        - podSelector: { matchLabels: { app: api } }
      ports:
        - port: 5432
          protocol: TCP
  egress:
    - to:
        - namespaceSelector: { matchLabels: { name: kube-system } }
          podSelector: { matchLabels: { k8s-app: kube-dns } }
      ports:
        - port: 53
          protocol: UDP
```

처음에 모두가 틀리는 두 가지 시맨틱 규칙:

- **기본은 모두 허용.** *첫* NetworkPolicy가 파드를 선택할 때까지 모든 트래픽이 허용됩니다. 어떤 정책이든 파드를 선택하면, 나열된 policyTypes에 대해 기본 거부가 되고, 명시적으로 허용된 트래픽만 통과합니다.
- **같은 방향의 정책은 가산형.** 두 ingress 정책이 같은 파드를 선택하면, 그들의 `from` 목록의 합집합이 허용됩니다. RBAC처럼 "거부" 규칙은 없습니다.

CNI 플러그인은 이 YAML 규칙을 자신의 네이티브 강제 메커니즘으로 컴파일합니다:

- iptables 기반 CNI(Calico iptables 모드)는 정책당 iptables 체인을 생성.
- IPVS 기반 모드는 IPVS 규칙을 생성.
- eBPF 기반(Cilium)은 인터페이스에 부착된 eBPF 프로그램으로 컴파일.

이것이 동일한 NetworkPolicy YAML을 가진 두 클러스터가 매우 다른 성능과 동작을 가질 수 있는 이유입니다 — 규칙 시맨틱은 표준이지만 강제 구현은 CNI별입니다. 일부 CNI는 L7 규칙, FQDN 기반 egress, 또는 스펙이 다루지 않는 클러스터 전역 정책을 위한 독자 CRD(Cilium의 `CiliumNetworkPolicy`, Calico의 `GlobalNetworkPolicy`)를 추가합니다.

### 이론에서 아래의 구성으로

이제 레슨은 이 추상을 적용합니다:

- **섹션 1 (CNI 명세)**는 §A입니다 — 실제 `ADD`/`DEL`/`CHECK` 인터페이스와 config 형식.
- **섹션 2 (Calico)와 3 (Cilium)**은 §B/§C 구현 선택을 구체적 플러그인 형태로 보여줍니다.
- **섹션 4 (eBPF 기본)**은 프로그램과 훅의 예제로 §C를 풀어냅니다.
- **섹션 5 (고급 NetworkPolicy)**는 §D 규칙을 비자명한 시나리오 — egress, CIDR, port range, DNS-aware (Cilium) — 에 사용합니다.
- **섹션 6 (Service Mesh 개요)**는 CNI 위의 L7 계층입니다 — Istio, Linkerd, Cilium Service Mesh — §A–§C의 pod-pod 연결 위에 구축.
- **섹션 7–9 (대역폭, IPv6, 트러블슈팅)**은 데이터 플레인 선택 위의 운영 오버레이입니다.

CNI를 계약으로, eBPF를 현대적 데이터 플레인으로, NetworkPolicy를 컴파일 시간 규칙으로 보고 나면, "왜 내 파드의 트래픽이 떨어지는가?" 질문은 특정 계층으로 매핑됩니다.

---

## 1. CNI 명세

컨테이너 네트워크 인터페이스(CNI)는 컨테이너 런타임이 컨테이너의 네트워킹을 구성하는 방법을 정의하는 명세입니다. Kubernetes는 CNI 플러그인을 사용하여 파드에 IP 주소를 할당하고, 라우트를 구성하며, 네트워크 네임스페이스를 설정합니다.

### 1.1 CNI 작동 방식

```
┌──────────────────────────────────────────────────────────────┐
│  Node                                                        │
│                                                              │
│  ┌──────────┐    1. CreatePod    ┌──────────┐                │
│  │ kubelet  │───────────────────▶│ CRI      │                │
│  │          │                    │(containerd)                │
│  └──────────┘                    └────┬─────┘                │
│                                       │                      │
│                              2. CNI ADD                      │
│                                       ▼                      │
│                                ┌──────────────┐              │
│                                │  CNI Plugin   │              │
│                                │ (calico/cilium)│             │
│                                └──────┬───────┘              │
│                                       │                      │
│                      3. veth 쌍 생성, IP 할당                  │
│                         라우트 구성                            │
│                                       │                      │
│  ┌──────────────────┐                 │                      │
│  │  Pod Network NS  │◀───────────────┘                       │
│  │  eth0: 10.0.1.5  │                                       │
│  └──────────────────┘                                        │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 CNI 플러그인 수명주기

CNI 플러그인은 다음 작업을 구현합니다:

| 작업 | 설명 | 호출 시점 |
|------|------|----------|
| `ADD` | 컨테이너의 네트워킹 설정 | 파드 생성 |
| `DEL` | 컨테이너의 네트워킹 정리 | 파드 삭제 |
| `CHECK` | 네트워킹이 여전히 올바른지 확인 | 헬스 체크 |
| `VERSION` | 지원하는 CNI 버전 보고 | 검색 |

간단한 CNI 플러그인 상호작용 (이해를 위한 것이며 매일 작성하는 코드가 아님):

```go
// Go로 작성된 CNI 플러그인 스켈레톤
package main

import (
	"encoding/json"
	"fmt"
	"net"
	"runtime"

	"github.com/containernetworking/cni/pkg/skel"
	"github.com/containernetworking/cni/pkg/types"
	current "github.com/containernetworking/cni/pkg/types/100"
	"github.com/containernetworking/cni/pkg/version"
)

// NetConf는 CNI 네트워크 구성을 나타냄
type NetConf struct {
	types.NetConf
	Subnet string `json:"subnet"`
}

func cmdAdd(args *skel.CmdArgs) error {
	conf := &NetConf{}
	if err := json.Unmarshal(args.StdinData, conf); err != nil {
		return fmt.Errorf("failed to parse config: %v", err)
	}

	// 1. 서브넷에서 IP 주소 할당
	_, subnet, _ := net.ParseCIDR(conf.Subnet)
	ip := allocateIP(subnet) // 구현 생략

	// 2. veth 쌍 생성 (한쪽은 파드, 한쪽은 호스트)
	// 3. 한쪽을 컨테이너 네트워크 네임스페이스로 이동
	// 4. 컨테이너 인터페이스에 IP 할당
	// 5. 라우트 설정

	result := &current.Result{
		CNIVersion: conf.CNIVersion,
		IPs: []*current.IPConfig{
			{
				Address: net.IPNet{
					IP:   ip,
					Mask: subnet.Mask,
				},
				Gateway: gatewayIP(subnet),
			},
		},
	}
	return types.PrintResult(result, conf.CNIVersion)
}

func cmdDel(args *skel.CmdArgs) error {
	// 정리: veth 쌍 제거, IP 해제
	return nil
}

func cmdCheck(args *skel.CmdArgs) error {
	// 네트워킹이 여전히 올바르게 구성되어 있는지 확인
	return nil
}

func main() {
	runtime.LockOSThread()
	skel.PluginMainFuncs(skel.CNIFuncs{
		Add:   cmdAdd,
		Del:   cmdDel,
		Check: cmdCheck,
	}, version.All, "example-cni-plugin")
}
```

### 1.3 CNI 구성

CNI 구성은 각 노드에 저장되며, 일반적으로 `/etc/cni/net.d/`에 위치합니다.

```json
{
  "cniVersion": "1.0.0",
  "name": "k8s-pod-network",
  "type": "calico",
  "datastore_type": "kubernetes",
  "mtu": 1440,
  "nodename_file_optional": false,
  "log_level": "Info",
  "log_file_path": "/var/log/calico/cni/cni.log",
  "ipam": {
    "type": "calico-ipam",
    "assign_ipv4": "true",
    "assign_ipv6": "false"
  },
  "container_settings": {
    "allow_ip_forwarding": false
  },
  "policy": {
    "type": "k8s"
  },
  "kubernetes": {
    "kubeconfig": "/etc/cni/net.d/calico-kubeconfig"
  }
}
```

```bash
# 노드의 CNI 구성 검사
ssh node01 "ls /etc/cni/net.d/"
# 10-calico.conflist

ssh node01 "cat /etc/cni/net.d/10-calico.conflist"

# CNI 플러그인 바이너리 위치
ssh node01 "ls /opt/cni/bin/"
# bandwidth  bridge  calico  calico-ipam  dhcp  flannel  host-local  ...
```

---

## 2. Calico

Calico는 가장 널리 배포된 CNI 플러그인 중 하나입니다. BGP 라우팅, IP-in-IP 터널링, 또는 VXLAN 오버레이를 사용하여 네트워킹과 네트워크 정책 적용을 제공합니다.

### 2.1 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│  Calico 아키텍처                                              │
│                                                              │
│  ┌──────────────────┐     ┌──────────────────┐              │
│  │    calico-node   │     │    calico-node   │              │
│  │   (DaemonSet)    │     │   (DaemonSet)    │              │
│  │                  │     │                  │              │
│  │ ┌──────────────┐ │     │ ┌──────────────┐ │              │
│  │ │ Felix         │ │     │ │ Felix         │ │              │
│  │ │ (정책        │ │     │ │ (정책        │ │              │
│  │ │  적용)       │ │     │ │  적용)       │ │              │
│  │ └──────────────┘ │     │ └──────────────┘ │              │
│  │ ┌──────────────┐ │     │ ┌──────────────┐ │              │
│  │ │ BIRD         │ │◀───▶│ │ BIRD         │ │              │
│  │ │ (BGP 데몬)   │ │ BGP │ │ (BGP 데몬)   │ │              │
│  │ └──────────────┘ │     │ └──────────────┘ │              │
│  └──────────────────┘     └──────────────────┘              │
│                                                              │
│  ┌───────────────────────────────────────────┐              │
│  │ calico-kube-controllers (Deployment)      │              │
│  │ Calico 데이터스토어를 K8s API와 동기화       │              │
│  └───────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

```bash
# 클러스터에 Calico 설치
kubectl apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.28.0/manifests/calico.yaml

# 또는 Tigera 오퍼레이터를 통해 (프로덕션 권장)
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.28.0/manifests/tigera-operator.yaml
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.28.0/manifests/custom-resources.yaml

# Calico 실행 확인
kubectl get pods -n calico-system
# NAME                                      READY   STATUS
# calico-node-xxxxx                         1/1     Running
# calico-kube-controllers-xxxxx             1/1     Running

# calicoctl CLI 설치
kubectl apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.28.0/manifests/calicoctl.yaml
```

### 2.2 데이터 플레인 모드

Calico는 세 가지 캡슐화 모드를 지원합니다:

| 모드 | 작동 방식 | 사용 시기 |
|------|----------|----------|
| **BGP (오버레이 없음)** | BGP를 통해 라우트 배포 | BGP 지원 라우터가 있는 온프레미스 |
| **IP-in-IP** | IP 헤더로 패킷 캡슐화 | 서브넷 간 통신 |
| **VXLAN** | UDP 기반 오버레이 | 클라우드 환경, BGP 미지원 |

```yaml
# Calico IPPool 구성
apiVersion: crd.projectcalico.org/v1
kind: IPPool
metadata:
  name: default-ipv4-ippool
spec:
  cidr: 192.168.0.0/16
  ipipMode: CrossSubnet      # 서브넷 간에만 IP-in-IP
  vxlanMode: Never
  natOutgoing: true
  nodeSelector: all()
```

```yaml
# VXLAN 모드 (클라우드 환경용)
apiVersion: crd.projectcalico.org/v1
kind: IPPool
metadata:
  name: vxlan-pool
spec:
  cidr: 10.244.0.0/16
  ipipMode: Never
  vxlanMode: Always
  natOutgoing: true
  blockSize: 26              # /26 = 노드 블록당 64개 IP
```

```bash
# 현재 Calico 구성 확인
kubectl exec -n calico-system calico-node-xxxxx -- calico-node -show-status

# IP 풀 조회
kubectl get ippools -o yaml

# 노드 간 BGP 피어링 조회
kubectl exec -n calico-system calico-node-xxxxx -- birdcl show protocols
```

### 2.3 Calico NetworkPolicy 확장

Calico는 표준 Kubernetes NetworkPolicy를 추가 기능으로 확장합니다.

```yaml
# Calico GlobalNetworkPolicy: 모든 네임스페이스에 적용
apiVersion: crd.projectcalico.org/v1
kind: GlobalNetworkPolicy
metadata:
  name: deny-external-egress
spec:
  selector: role == "internal"
  types:
    - Egress
  egress:
    # 클러스터 내 트래픽 허용
    - action: Allow
      destination:
        nets:
          - 10.0.0.0/8
          - 172.16.0.0/12
          - 192.168.0.0/16
    # 나머지 모두 차단
    - action: Deny
```

```yaml
# 애플리케이션 계층 정책이 있는 Calico NetworkPolicy (HTTP 메서드)
apiVersion: crd.projectcalico.org/v1
kind: NetworkPolicy
metadata:
  name: allow-get-only
  namespace: production
spec:
  selector: app == "api"
  types:
    - Ingress
  ingress:
    - action: Allow
      protocol: TCP
      source:
        selector: role == "frontend"
      destination:
        ports: [8080]
      http:
        methods: ["GET", "HEAD"]    # L7 정책 (Envoy 프록시 필요)
```

---

## 3. Cilium

Cilium은 eBPF를 사용하여 네트워킹, 보안, 관측 가능성을 제공하는 CNI 플러그인입니다. iptables 없이 커널 수준에서 작동하여 더 나은 성능과 풍부한 기능을 제공합니다.

### 3.1 eBPF 기반 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│  Cilium 아키텍처                                              │
│                                                              │
│  ┌──────────────────────────────────────────┐               │
│  │  cilium-agent (노드당 DaemonSet)         │               │
│  │                                          │               │
│  │  ┌────────────────┐  ┌────────────────┐  │               │
│  │  │ eBPF 프로그램   │  │ eBPF 프로그램   │  │               │
│  │  │ (TC 인그레스)   │  │ (TC 이그레스)   │  │               │
│  │  └───────┬────────┘  └───────┬────────┘  │               │
│  │          │                    │           │               │
│  │  ┌───────▼────────────────────▼────────┐  │               │
│  │  │     eBPF Maps (커널 데이터 플레인)    │  │               │
│  │  │  - 연결 추적                        │  │               │
│  │  │  - 정책 맵                          │  │               │
│  │  │  - 서비스 맵 (kube-proxy 대체)       │  │               │
│  │  │  - NAT 맵                           │  │               │
│  │  └────────────────────────────────────┘  │               │
│  └──────────────────────────────────────────┘               │
│                                                              │
│  ┌───────────────────────────────┐                          │
│  │  Hubble (관측 가능성 계층)      │                          │
│  │  - 플로우 로그                  │                          │
│  │  - 서비스 맵                    │                          │
│  │  - 메트릭 (Prometheus)         │                          │
│  └───────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 설치 및 구성

```bash
# Cilium CLI 설치
curl -L --remote-name https://github.com/cilium/cilium-cli/releases/latest/download/cilium-linux-amd64.tar.gz
tar xzvf cilium-linux-amd64.tar.gz
sudo mv cilium /usr/local/bin/

# 클러스터에 Cilium 설치 (kube-proxy 대체)
cilium install --version 1.16.0 \
  --set kubeProxyReplacement=true \
  --set k8sServiceHost=API_SERVER_IP \
  --set k8sServicePort=6443

# 또는 Helm으로
helm repo add cilium https://helm.cilium.io/
helm install cilium cilium/cilium \
  --namespace kube-system \
  --set kubeProxyReplacement=true \
  --set hubble.enabled=true \
  --set hubble.relay.enabled=true \
  --set hubble.ui.enabled=true

# 설치 확인
cilium status
# Output:
#     /\
#  /\  Warning: Unable to detect...
# /  \
# \  /  Cilium:       OK
#  \/   Operator:     OK
#       Hubble:       OK

# 연결 테스트 실행
cilium connectivity test
```

```bash
# Cilium으로 minikube 시작 (kube-proxy 없이)
minikube start --network-plugin=cni --cni=false
cilium install
cilium status --wait
```

### 3.3 Cilium 네트워크 정책

Cilium은 L7(애플리케이션 계층) 인식과 DNS 기반 규칙으로 Kubernetes NetworkPolicy를 확장합니다.

```yaml
# CiliumNetworkPolicy: L7 HTTP 필터링
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: api-l7-policy
  namespace: production
spec:
  endpointSelector:
    matchLabels:
      app: api-server
  ingress:
    - fromEndpoints:
        - matchLabels:
            app: frontend
      toPorts:
        - ports:
            - port: "8080"
              protocol: TCP
          rules:
            http:
              - method: "GET"
                path: "/api/v1/.*"
              - method: "POST"
                path: "/api/v1/orders"
                headers:
                  - 'Content-Type: application/json'
```

```yaml
# CiliumClusterWideNetworkPolicy: 클러스터 전체 기본 거부
apiVersion: cilium.io/v2
kind: CiliumClusterwideNetworkPolicy
metadata:
  name: default-deny
spec:
  endpointSelector: {}
  ingress:
    - fromEndpoints:
        - {}                    # 클러스터 내 트래픽만 허용
  egress:
    - toEndpoints:
        - {}
    - toEntities:
        - kube-dns              # 항상 DNS 허용
```

### 3.4 Hubble 관측 가능성

Hubble은 eBPF 플로우 데이터를 사용하여 심층 네트워크 관측 가능성을 제공합니다.

```bash
# Hubble 활성화
cilium hubble enable --ui

# Hubble UI 포트 포워딩
cilium hubble ui
# 브라우저에서 http://localhost:12000 열림

# 플로우 관측을 위한 Hubble CLI 사용
hubble observe --namespace production
# Timestamp  Source                Destination           Type    Verdict
# 10:30:01   production/frontend   production/api        L7/HTTP  FORWARDED
# 10:30:01   production/api        production/database   L4/TCP   FORWARDED

# 특정 플로우 필터링
hubble observe --namespace production \
  --from-pod production/frontend \
  --to-pod production/api \
  --protocol http

# 플로우 요약
hubble observe --namespace production -o json | \
  jq '.flow | {src: .source.labels, dst: .destination.labels, verdict: .verdict}'
```

---

## 4. eBPF 기본

### 4.1 eBPF란?

eBPF(extended Berkeley Packet Filter)는 커널 소스 코드를 수정하거나 커널 모듈을 로드하지 않고도 Linux 커널에서 프로그램을 실행할 수 있게 하는 기술입니다. 커널 내부의 프로그래밍 가능한 데이터 플레인입니다.

```
┌─────────────────────────────────────────────────────────────┐
│  사용자 공간                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ cilium-agent │  │ Hubble       │  │ bpftool      │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
├─────────┼──────────────────┼──────────────────┼──────────────┤
│  커널   │                  │                  │              │
│         ▼                  ▼                  ▼              │
│  ┌──────────────────────────────────────────────────┐       │
│  │              eBPF 가상 머신                        │       │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐          │       │
│  │  │  TC     │  │ XDP     │  │ Socket  │          │       │
│  │  │ ingress │  │(express │  │ filter  │  ...     │       │
│  │  │/egress  │  │ data    │  │         │          │       │
│  │  │         │  │ path)   │  │         │          │       │
│  │  └─────────┘  └─────────┘  └─────────┘          │       │
│  │                                                   │       │
│  │  eBPF Maps (커널/사용자 공유 상태):                  │       │
│  │  - 해시 맵, 배열, LRU, 링 버퍼                      │       │
│  └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 네트워킹을 위한 eBPF

Kubernetes 네트워킹과 관련된 eBPF 훅 포인트:

| 훅 포인트 | 위치 | 사용 사례 |
|----------|------|----------|
| **XDP** (eXpress Data Path) | NIC 드라이버, 커널 스택 이전 | DDoS 완화, 로드 밸런싱 |
| **TC** (Traffic Control) | 네트워크 스택 이후 | 파드 네트워킹, 정책 적용 |
| **Socket** | 소켓 작업 | 연결 추적, 로드 밸런싱 |
| **cgroup** | cgroup 수준 | 파드별 대역폭 제한 |

### 4.3 eBPF vs iptables

| 측면 | iptables | eBPF (Cilium) |
|------|----------|---------------|
| 규칙 평가 | 선형 체인 순회 | 해시 맵 조회 (O(1)) |
| 서비스 성능 | 서비스 수에 따라 저하 | 수에 관계없이 일정 |
| 업데이트 지연 | 전체 체인 재구성 | 점진적 맵 업데이트 |
| L7 가시성 | 없음 | HTTP, gRPC, Kafka, DNS |
| 연결 추적 | conntrack 모듈 | BPF 맵 |
| 리소스 사용 | 규칙이 많을수록 높음 | 낮음, 더 잘 확장 |

```bash
# iptables vs eBPF의 서비스 라우팅 비교
# iptables: n개 서비스에 대해 O(n) 규칙
sudo iptables -t nat -L KUBE-SERVICES | wc -l
# 중간 규모 클러스터에서 500+ 규칙

# eBPF: 서비스 수에 관계없이 O(1) 맵 조회
sudo bpftool map show
# Cilium이 서비스 라우팅에 사용하는 eBPF 맵 표시
```

---

## 5. 고급 NetworkPolicy

### 5.1 이그레스 정책

파드가 도달할 수 있는 외부 서비스를 제어합니다.

```yaml
# 파드가 특정 외부 서비스에만 도달하도록 허용
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: restricted-egress
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: payment-service
  policyTypes:
    - Egress
  egress:
    # DNS 허용
    - to:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: kube-system
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53
    # 결제 처리업체로의 HTTPS 허용
    - to:
        - ipBlock:
            cidr: 203.0.113.0/24     # 결제 처리업체 IP 범위
      ports:
        - protocol: TCP
          port: 443
    # 내부 데이터베이스 허용
    - to:
        - podSelector:
            matchLabels:
              app: postgres
      ports:
        - protocol: TCP
          port: 5432
```

### 5.2 CIDR 기반 규칙

```yaml
# 클라우드 메타데이터 서비스에 대한 접근 차단 (보안 모범 사례)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: block-metadata
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Egress
  egress:
    # 클라우드 메타데이터 엔드포인트를 제외한 모든 것 허용
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
            except:
              - 169.254.169.254/32   # AWS/GCP 메타데이터
              - 100.100.100.200/32   # Azure 메타데이터
```

### 5.3 포트 범위 정책

```yaml
# 포트 범위 허용 (Kubernetes 1.25+)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-ephemeral-ports
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: data-processor
  policyTypes:
    - Egress
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: data-store
      ports:
        - protocol: TCP
          port: 9000
          endPort: 9100           # 포트 범위 9000-9100
```

### 5.4 DNS 인식 정책 (Cilium)

```yaml
# CiliumNetworkPolicy: DNS 기반 이그레스 필터링
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: dns-egress-policy
  namespace: production
spec:
  endpointSelector:
    matchLabels:
      app: web-scraper
  egress:
    # DNS 해석 허용
    - toEndpoints:
        - matchLabels:
            io.kubernetes.pod.namespace: kube-system
            k8s-app: kube-dns
      toPorts:
        - ports:
            - port: "53"
              protocol: UDP
          rules:
            dns:
              - matchPattern: "*.example.com"
              - matchPattern: "api.github.com"
    # 해석된 FQDN에만 HTTPS 허용
    - toFQDNs:
        - matchPattern: "*.example.com"
        - matchName: "api.github.com"
      toPorts:
        - ports:
            - port: "443"
              protocol: TCP
```

---

## 6. 서비스 메시 개요

### 6.1 서비스 메시란?

서비스 메시(service mesh)는 서비스 간 통신을 위한 투명한 인프라 계층을 추가하여, 애플리케이션 코드 변경 없이 mTLS 암호화, 트래픽 관리, 관측 가능성을 제공합니다.

```
┌────────────────────────────────────────────────────────────┐
│  서비스 메시 없음               서비스 메시 사용              │
│                                                            │
│  ┌─────┐    ┌─────┐          ┌─────┐    ┌─────┐          │
│  │App A│───▶│App B│          │App A│    │App B│          │
│  └─────┘    └─────┘          │     │    │     │          │
│                               │proxy│───▶│proxy│          │
│  직접 연결                    │(sidecar) │(sidecar)       │
│  암호화 없음                  └─────┘    └─────┘          │
│  관측 불가                    mTLS, 재시도, 추적           │
└────────────────────────────────────────────────────────────┘
```

### 6.2 Istio

Istio는 Envoy 사이드카를 사용하는 가장 기능이 풍부한 서비스 메시입니다.

```bash
# Istio 설치
curl -L https://istio.io/downloadIstio | sh -
cd istio-1.23.0
export PATH=$PWD/bin:$PATH

istioctl install --set profile=demo
kubectl label namespace production istio-injection=enabled
```

```yaml
# 트래픽 분할을 위한 Istio VirtualService
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: app-routing
  namespace: production
spec:
  hosts:
    - app-service
  http:
    - match:
        - headers:
            x-user-type:
              exact: beta
      route:
        - destination:
            host: app-service
            subset: v2
    - route:
        - destination:
            host: app-service
            subset: v1
          weight: 90
        - destination:
            host: app-service
            subset: v2
          weight: 10
```

### 6.3 Linkerd

Linkerd는 단순성과 성능에 중점을 둔 경량 서비스 메시입니다.

```bash
# Linkerd 설치
curl --proto '=https' --tlsv1.2 -sSfL https://run.linkerd.io/install | sh
linkerd install --crds | kubectl apply -f -
linkerd install | kubectl apply -f -
linkerd check

# 네임스페이스에 사이드카 주입
kubectl annotate namespace production linkerd.io/inject=enabled
```

### 6.4 Cilium 서비스 메시

Cilium은 사이드카 없이 커널의 eBPF를 사용하여 서비스 메시 기능을 제공합니다.

```bash
# Cilium 서비스 메시 기능 활성화
helm upgrade cilium cilium/cilium \
  --namespace kube-system \
  --set envoy.enabled=true \
  --set loadBalancer.l7.backend=envoy
```

**서비스 메시 비교:**

| 기능 | Istio | Linkerd | Cilium |
|------|-------|---------|--------|
| 프록시 | Envoy 사이드카 | Rust 마이크로 프록시 사이드카 | eBPF (L4는 사이드카 없음) |
| mTLS | 예 | 예 | 예 (WireGuard 또는 IPsec) |
| L7 정책 | 전체 (HTTP, gRPC) | HTTP 헤더 | HTTP, gRPC, Kafka, DNS |
| 리소스 오버헤드 | 높음 (파드당 Envoy) | 낮음 | 최저 |
| 복잡도 | 높음 | 낮음 | 중간 |

---

## 7. 대역폭과 QoS

Kubernetes는 어노테이션을 통해 대역폭 제한을 지원합니다 (bandwidth CNI 플러그인 사용).

```yaml
# 대역폭 제한이 있는 파드
apiVersion: v1
kind: Pod
metadata:
  name: bandwidth-limited
  annotations:
    kubernetes.io/ingress-bandwidth: "10M"    # 10 Mbps 인그레스
    kubernetes.io/egress-bandwidth: "5M"      # 5 Mbps 이그레스
spec:
  containers:
    - name: app
      image: my-app:v1
```

```yaml
# Cilium 대역폭 매니저 (CNI bandwidth 플러그인보다 효율적)
# Helm values에서 활성화:
# bandwidthManager:
#   enabled: true

# 그런 다음 파드에 어노테이션 사용
apiVersion: v1
kind: Pod
metadata:
  name: rate-limited-pod
  annotations:
    kubernetes.io/egress-bandwidth: "50M"
spec:
  containers:
    - name: app
      image: my-app:v1
```

---

## 8. IPv4/IPv6 듀얼 스택(Dual-Stack)

Kubernetes는 파드와 서비스가 IPv4와 IPv6 주소를 모두 받는 듀얼 스택 네트워킹을 지원합니다.

```yaml
# 듀얼 스택 구성의 서비스
apiVersion: v1
kind: Service
metadata:
  name: dual-stack-service
spec:
  type: ClusterIP
  ipFamilyPolicy: PreferDualStack    # 또는 RequireDualStack
  ipFamilies:
    - IPv4
    - IPv6
  selector:
    app: my-app
  ports:
    - port: 80
      targetPort: 8080
```

```bash
# 듀얼 스택 서비스 확인
kubectl get svc dual-stack-service -o yaml | grep -A5 clusterIPs
# clusterIPs:
# - 10.96.0.100
# - fd00::1:100

# 파드에 IPv4와 IPv6 모두 있는지 확인
kubectl exec my-pod -- ip addr show eth0
# inet 10.244.1.5/24
# inet6 fd00::1:5/128
```

---

## 9. 네트워크 트러블슈팅

### 9.1 진단 도구

```bash
# 네트워크 디버그 파드 배포
kubectl run netdebug --image=nicolaka/netshoot --rm -it --restart=Never -- bash

# 디버그 파드 내부에서:
# DNS 해석
nslookup kubernetes.default.svc.cluster.local
dig +short my-service.production.svc.cluster.local

# 연결 테스트
curl -v http://my-service.production.svc.cluster.local:8080/health
wget -qO- --timeout=2 http://10.96.0.1:443

# TCP 연결 테스트
nc -zv my-service.production 8080

# 파드로의 경로 추적
traceroute 10.244.1.5

# 패킷 캡처
tcpdump -i eth0 -n port 8080 -c 100

# MTU 확인
ip link show eth0
ping -M do -s 1472 10.244.2.5
```

```bash
# 노드 수준 진단
# CNI 플러그인 로그 확인
kubectl logs -n kube-system -l k8s-app=calico-node --tail=50

# kube-proxy 로그 확인 (Cilium kube-proxy 대체를 사용하지 않는 경우)
kubectl logs -n kube-system -l k8s-app=kube-proxy --tail=50

# 노드의 iptables 규칙 검사
ssh node01 "sudo iptables -t nat -L KUBE-SERVICES -n | head -20"

# conntrack 테이블 확인
ssh node01 "sudo conntrack -L | wc -l"
```

### 9.2 일반적인 문제

| 증상 | 가능한 원인 | 진단 명령 |
|------|-----------|----------|
| 파드가 Service에 도달 못함 | kube-proxy/eBPF 구성 오류 | `kubectl get endpoints <svc>` |
| DNS가 해석되지 않음 | CoreDNS 파드 다운 또는 구성 오류 | `kubectl get pods -n kube-system -l k8s-app=kube-dns` |
| 노드 간 파드 통신 실패 | CNI 오버레이/라우팅 구성 오류 | `kubectl exec pod -- ping <cross-node-pod-ip>` |
| 파드가 IP를 받지 못함 | IPAM 고갈 | `kubectl describe pod <pod>` (이벤트) |
| NetworkPolicy가 작동하지 않음 | CNI가 NetworkPolicy를 지원하지 않음 | CNI 문서 확인 (예: flannel은 NetworkPolicy 미지원) |
| 간헐적 타임아웃 | 오버레이에서 MTU 불일치 | 노드에서 `ping -M do -s 1400 <pod-ip>` |

---

## 연습문제

### 연습문제 1: Minikube에 Cilium 설치 및 확인

Cilium을 CNI 플러그인으로 사용하여 minikube 클러스터를 시작하세요 (kube-proxy 대체). Cilium 연결 테스트를 실행하고 Hubble을 활성화하여 파드 간 플로우를 관측하세요.

<details><summary>정답 보기</summary>

```bash
# 기본 CNI 없이 minikube 시작
minikube start --network-plugin=cni --cni=false --cpus=4 --memory=4096

# kube-proxy 대체와 함께 Cilium 설치
cilium install --set kubeProxyReplacement=true

# Cilium 준비 대기
cilium status --wait

# 연결 테스트 실행
cilium connectivity test
# 모든 테스트가 통과해야 함 (몇 분 소요)

# Hubble 활성화
cilium hubble enable --ui

# 플로우 관측
hubble observe --all
# 연결 테스트 파드의 플로우가 보임

# Hubble UI 포트 포워딩
cilium hubble ui &

# kube-proxy 대체 확인
cilium status | grep KubeProxyReplacement
# KubeProxyReplacement: True
```

</details>

### 연습문제 2: Calico vs Cilium NetworkPolicy

두 개의 파드(`client`와 `server`)가 있는 네임스페이스를 생성하세요. 클라이언트가 포트 8080에서 서버에 HTTP GET 요청만 할 수 있도록 하는 NetworkPolicy를 작성하세요. 표준 Kubernetes NetworkPolicy와 Cilium CiliumNetworkPolicy 모두를 사용하여 구현하세요 (L7 기능의 차이를 보여줌).

<details><summary>정답 보기</summary>

```yaml
# setup.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: netpol-test
---
apiVersion: v1
kind: Pod
metadata:
  name: server
  namespace: netpol-test
  labels:
    role: server
spec:
  containers:
    - name: server
      image: hashicorp/http-echo
      args: ["-text=hello", "-listen=:8080"]
      ports:
        - containerPort: 8080
---
apiVersion: v1
kind: Pod
metadata:
  name: client
  namespace: netpol-test
  labels:
    role: client
spec:
  containers:
    - name: client
      image: nicolaka/netshoot
      command: ["sleep", "3600"]
```

```yaml
# 표준 Kubernetes NetworkPolicy (L3/L4만, HTTP 메서드 필터링 불가)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-client-to-server
  namespace: netpol-test
spec:
  podSelector:
    matchLabels:
      role: server
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              role: client
      ports:
        - protocol: TCP
          port: 8080
```

```yaml
# Cilium CiliumNetworkPolicy (L7 필터링 -- HTTP 메서드 제한 가능)
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: allow-client-get-only
  namespace: netpol-test
spec:
  endpointSelector:
    matchLabels:
      role: server
  ingress:
    - fromEndpoints:
        - matchLabels:
            role: client
      toPorts:
        - ports:
            - port: "8080"
              protocol: TCP
          rules:
            http:
              - method: "GET"    # GET만 허용; POST, PUT, DELETE는 차단
```

```bash
kubectl apply -f setup.yaml

# 표준 NetworkPolicy로 테스트 (포트 8080의 모든 HTTP 메서드 허용)
kubectl apply -f k8s-netpol.yaml
kubectl exec -n netpol-test client -- curl -s http://server:8080
# hello (작동)
kubectl exec -n netpol-test client -- curl -s -X POST http://server:8080
# hello (역시 작동 -- L4는 HTTP 메서드를 구분하지 못함)

# 표준 정책 제거, Cilium 정책 적용
kubectl delete networkpolicy allow-client-to-server -n netpol-test
kubectl apply -f cilium-netpol.yaml
kubectl exec -n netpol-test client -- curl -s http://server:8080
# hello (GET 작동)
kubectl exec -n netpol-test client -- curl -s -X POST http://server:8080
# Access denied (POST는 L7 정책에 의해 차단)
```

</details>

### 연습문제 3: DNS 인식 이그레스 정책

CiliumNetworkPolicy를 사용하여 `app=scraper` 레이블이 있는 파드가 다음을 할 수 있도록 하는 정책을 생성하세요:
- `api.github.com`과 `*.amazonaws.com`에 대해서만 DNS 해석
- 해석된 FQDN에만 HTTPS 연결
- 다른 모든 이그레스 트래픽 차단

<details><summary>정답 보기</summary>

```yaml
# dns-egress.yaml
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: scraper-dns-egress
  namespace: production
spec:
  endpointSelector:
    matchLabels:
      app: scraper
  egress:
    # DNS 해석 허용 (허용된 도메인에 대해서만)
    - toEndpoints:
        - matchLabels:
            io.kubernetes.pod.namespace: kube-system
            k8s-app: kube-dns
      toPorts:
        - ports:
            - port: "53"
              protocol: UDP
          rules:
            dns:
              - matchName: "api.github.com"
              - matchPattern: "*.amazonaws.com"
    # 해석된 FQDN에 HTTPS 허용
    - toFQDNs:
        - matchName: "api.github.com"
        - matchPattern: "*.amazonaws.com"
      toPorts:
        - ports:
            - port: "443"
              protocol: TCP
```

```bash
kubectl apply -f dns-egress.yaml

# 테스트 파드 배포
kubectl run scraper --image=nicolaka/netshoot --rm -it \
  --restart=Never --labels="app=scraper" -n production -- bash

# 파드 내부에서:
# 이것은 작동해야 함
curl -s https://api.github.com/rate_limit
# {"resources": ...}

# 이것은 작동해야 함
curl -s https://s3.amazonaws.com
# S3의 응답

# 이것은 차단되어야 함
curl -s --connect-timeout 3 https://google.com
# Connection timeout (정책에 의해 차단)

# 승인되지 않은 도메인의 DNS도 실패해야 함
nslookup google.com
# NXDOMAIN 또는 타임아웃
```

</details>

### 연습문제 4: 네트워크 트러블슈팅

`debug` 네임스페이스의 파드가 `app` 네임스페이스의 `backend` 서비스에 도달할 수 없습니다. 서비스는 존재하고 백엔드 파드는 실행 중입니다. 체계적인 트러블슈팅 프로세스를 설명하고 진단 명령을 작성하세요.

<details><summary>정답 보기</summary>

```bash
# 1단계: 서비스와 엔드포인트 존재 확인
kubectl get svc backend -n app
kubectl get endpoints backend -n app
# 엔드포인트가 비어있으면 파드 레이블이 서비스 셀렉터와 일치하는지 확인
kubectl get pods -n app --show-labels | grep backend

# 2단계: 소스 파드에서 DNS 해석 확인
kubectl exec -n debug debug-pod -- nslookup backend.app.svc.cluster.local
# DNS가 실패하면 CoreDNS 확인
kubectl get pods -n kube-system -l k8s-app=kube-dns
kubectl logs -n kube-system -l k8s-app=kube-dns --tail=20

# 3단계: 직접 파드 간 연결 확인 (서비스 우회)
BACKEND_POD_IP=$(kubectl get pod -n app -l app=backend -o jsonpath='{.items[0].status.podIP}')
kubectl exec -n debug debug-pod -- curl -v --connect-timeout 3 http://${BACKEND_POD_IP}:8080

# 4단계: NetworkPolicy 확인
kubectl get networkpolicies -n app
kubectl get networkpolicies -n debug
# 기본 거부 정책이나 누락된 허용 규칙 확인

kubectl describe networkpolicy -n app
# 정책이 debug 네임스페이스에서의 인그레스를 허용하는지 확인

# 5단계: Cilium 사용 시 정책 판정 확인
hubble observe --namespace app --to-pod app/backend --verdict DROPPED

# 6단계: 노드 간 문제인지 확인
kubectl get pod debug-pod -n debug -o wide
kubectl get pod -n app -l app=backend -o wide
# 파드가 다른 노드에 있으면 오버레이 네트워킹 확인

# 7단계: 노드 수준 네트워킹 확인
# 소스 노드에 SSH 접속하여 연결 테스트
ssh node01 "curl --connect-timeout 3 http://${BACKEND_POD_IP}:8080"

# 8단계: 수정 -- NetworkPolicy가 문제인 경우 허용 규칙 추가
cat <<'EOF' | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-debug-to-backend
  namespace: app
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: debug
      ports:
        - protocol: TCP
          port: 8080
EOF
```

</details>

### 연습문제 5: 듀얼 스택 서비스 구성

IPv4/IPv6 듀얼 스택으로 Kubernetes 서비스를 구성하세요. `RequireDualStack` 정책으로 서비스를 생성하고 두 주소가 모두 할당되어 접근 가능한지 확인하세요.

<details><summary>정답 보기</summary>

```yaml
# dual-stack-app.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dual-stack-app
  namespace: default
spec:
  replicas: 2
  selector:
    matchLabels:
      app: dual-stack
  template:
    metadata:
      labels:
        app: dual-stack
    spec:
      containers:
        - name: app
          image: hashicorp/http-echo
          args: ["-text=dual-stack-works", "-listen=:8080"]
          ports:
            - containerPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: dual-stack-svc
  namespace: default
spec:
  type: ClusterIP
  ipFamilyPolicy: RequireDualStack
  ipFamilies:
    - IPv4
    - IPv6
  selector:
    app: dual-stack
  ports:
    - port: 80
      targetPort: 8080
```

```bash
kubectl apply -f dual-stack-app.yaml

# 듀얼 스택 IP 할당 확인
kubectl get svc dual-stack-svc -o jsonpath='{.spec.clusterIPs}'
# ["10.96.0.150","fd00::96:150"]

# 두 IP 패밀리 확인
kubectl get svc dual-stack-svc -o jsonpath='{.spec.ipFamilies}'
# ["IPv4","IPv6"]

# 테스트 파드에서 확인
kubectl run test --image=nicolaka/netshoot --rm -it --restart=Never -- bash

# 파드 내부에서:
# IPv4 접근
curl -4 http://dual-stack-svc/
# dual-stack-works

# IPv6 접근
curl -6 http://dual-stack-svc/
# dual-stack-works

# DNS가 A와 AAAA 레코드 모두 반환
dig dual-stack-svc.default.svc.cluster.local A +short
# 10.96.0.150
dig dual-stack-svc.default.svc.cluster.local AAAA +short
# fd00::96:150

# 파드에 두 주소가 있는지 확인
kubectl exec dual-stack-app-xxxxx -- ip addr show eth0
# inet 10.244.1.50/24
# inet6 fd00:244:1::50/128
```

</details>

---

**이전**: [인그레스와 Gateway API](./07_Ingress_and_Gateway_API.md) | **다음**: [Helm과 Kustomize](./09_Helm_and_Kustomize.md)

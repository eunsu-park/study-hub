[이전: 컨테이너 네트워킹](./19_Container_Networking.md) | [다음: 서비스 품질](./21_Quality_of_Service.md)

---

# 20. 소프트웨어 정의 네트워킹(Software-Defined Networking, SDN)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 제어 평면(control plane)과 데이터 평면(data plane) 분리 개념 및 SDN이 등장한 이유를 설명한다
2. OpenFlow 프로토콜과 플로우 테이블 매칭 파이프라인을 설명한다
3. SDN 컨트롤러 아키텍처(중앙집중형 vs 분산형)를 비교한다
4. 포워딩 규칙을 프로그래밍하는 기본 SDN 컨트롤러를 구현한다
5. P4 프로그래머블 데이터 평면과 현대 네트워킹에서의 역할을 논의한다

---

## 목차

1. [전통적 아키텍처 vs SDN 아키텍처](#1-전통적-아키텍처-vs-sdn-아키텍처)
2. [OpenFlow 프로토콜](#2-openflow-프로토콜)
3. [SDN 컨트롤러](#3-sdn-컨트롤러)
4. [네트워크 애플리케이션](#4-네트워크-애플리케이션)
5. [P4: 프로그래머블 데이터 평면](#5-p4-프로그래머블-데이터-평면)
6. [실제 SDN 적용](#6-실제-sdn-적용)
7. [연습문제](#7-연습문제)

---

## 1. 전통적 아키텍처 vs SDN 아키텍처

### 1.1 전통적 네트워킹의 문제점

전통적인 네트워크에서는 각 장치(라우터, 스위치)가 다음 두 가지를 모두 포함합니다:
- **제어 평면(control plane)**: 라우팅 결정 (OSPF, BGP, 스패닝 트리)
- **데이터 평면(data plane)**: 제어 평면의 결정에 따른 패킷 포워딩

```
전통적 네트워크:

  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Router A │    │ Router B │    │ Router C │
  │┌────────┐│    │┌────────┐│    │┌────────┐│
  ││Control ││    ││Control ││    ││Control ││
  ││ Plane  ││    ││ Plane  ││    ││ Plane  ││
  │├────────┤│    │├────────┤│    │├────────┤│
  ││  Data  ││    ││  Data  ││    ││  Data  ││
  ││ Plane  ││    ││ Plane  ││    ││ Plane  ││
  │└────────┘│    │└────────┘│    │└────────┘│
  └──────────┘    └──────────┘    └──────────┘
  각 장치가 독립적으로 포워딩 방법을 결정합니다.
  설정: 장치별 CLI (오류 발생 가능, 느림).
```

문제점:
- **분산 복잡성**: 각 장치가 자체 프로토콜을 실행
- **벤더 종속(vendor lock-in)**: 벤더마다 독점적인 CLI와 펌웨어
- **느린 혁신**: 새 기능 추가 시 모든 장치의 펌웨어 업데이트 필요
- **일관성 부족**: 네트워크 전체 정책 적용이 어려움

### 1.2 SDN 아키텍처

SDN은 제어 평면을 중앙 집중형 컨트롤러로 분리합니다:

```
SDN 아키텍처:

              ┌────────────────────────┐
              │     SDN 컨트롤러        │  ← 중앙집중형 두뇌
              │  (네트워크 전체 뷰)      │
              │                        │
              │  • 라우팅 결정          │
              │  • 정책 적용            │
              │  • 토폴로지 탐색        │
              └───────┬────┬────┬──────┘
                      │    │    │   사우스바운드 API (OpenFlow)
              ┌───────┘    │    └───────┐
              ▼            ▼            ▼
         ┌─────────┐ ┌─────────┐ ┌─────────┐
         │Switch A │ │Switch B │ │Switch C │
         │┌───────┐│ │┌───────┐│ │┌───────┐│
         ││ Data  ││ ││ Data  ││ ││ Data  ││
         ││ Plane ││ ││ Plane ││ ││ Plane ││
         │└───────┘│ │└───────┘│ │└───────┘│
         └─────────┘ └─────────┘ └─────────┘
         "단순" 스위치 — 지시대로 포워딩만 합니다.
```

### 1.3 SDN 계층

| 계층 | 기능 | 예시 |
|------|------|------|
| 애플리케이션 계층 | 네트워크 앱 (방화벽, 로드 밸런서, 모니터) | 커스텀 앱, 노스바운드 API 소비자 |
| 제어 계층 | 중앙집중형 로직, 토폴로지, 상태 | OpenDaylight, ONOS, Ryu, Floodlight |
| 인프라 계층 | 패킷 포워딩 하드웨어 | OpenFlow 스위치, P4 스위치 |

API:
- **노스바운드 API(Northbound API)**: 컨트롤러 ↔ 애플리케이션 (REST, gRPC)
- **사우스바운드 API(Southbound API)**: 컨트롤러 ↔ 스위치 (OpenFlow, P4Runtime, NETCONF)

---

## 2. OpenFlow 프로토콜

### 2.1 플로우 테이블(Flow Table)

각 OpenFlow 스위치는 하나 이상의 **플로우 테이블**을 포함합니다. 각 테이블에는 플로우 엔트리가 있습니다:

```
플로우 엔트리:
┌──────────────┬───────────┬──────────────┬─────────┬──────────┐
│ 매치 필드     │ 우선순위   │ 명령어        │ 카운터  │ 타임아웃  │
│              │           │              │         │          │
│ src_ip       │ 높을수록   │ 포트 3으로    │ 패킷수  │ 유휴: 60s│
│ dst_ip       │ 먼저 검사  │ 포워딩        │ 바이트  │ 하드: 0  │
│ src_port     │           │ VLAN 설정     │         │          │
│ dst_port     │           │ 드롭          │         │          │
│ protocol     │           │ 컨트롤러로    │         │          │
│ in_port      │           │ 전송          │         │          │
│ VLAN ID      │           │              │         │          │
└──────────────┴───────────┴──────────────┴─────────┴──────────┘
```

### 2.2 패킷 처리 파이프라인

```
스위치에 패킷 도착:

  ┌─────────┐    ┌──────────┐    ┌──────────┐
  │ Table 0 │───►│ Table 1  │───►│ Table 2  │───► ...
  └────┬────┘    └────┬─────┘    └────┬─────┘
       │              │               │
   매치 발견?     매치 발견?      매치 발견?
       │              │               │
   예: 명령어    예: 명령어      예: 명령어
     실행         실행            실행
       │              │               │
   아니오:       아니오:         아니오: 테이블 미스
   다음 테이블로  다음 테이블로   → 컨트롤러로 전송
                                   또는 드롭
```

### 2.3 OpenFlow 메시지

```python
# 개념적 OpenFlow 메시지 타입

class OpenFlowMessages:
    """OpenFlow 주요 메시지 카테고리."""

    # 컨트롤러 → 스위치
    FLOW_MOD = "flow_mod"           # 플로우 엔트리 추가/수정/삭제
    PACKET_OUT = "packet_out"       # 특정 포트로 패킷 전송
    BARRIER = "barrier"             # 이전 메시지 모두 처리 보장

    # 스위치 → 컨트롤러
    PACKET_IN = "packet_in"         # 패킷이 플로우와 불일치 → 컨트롤러에 문의
    FLOW_REMOVED = "flow_removed"   # 플로우 엔트리 만료 또는 삭제
    PORT_STATUS = "port_status"     # 포트 상태 변경 (업/다운)

    # 양방향
    HELLO = "hello"                 # 연결 설정
    ECHO = "echo"                   # 킵얼라이브
    FEATURES_REQUEST = "features"   # 컨트롤러가 스위치 기능 요청
```

### 2.4 플로우 매칭 예시

```python
def match_packet(flow_tables, packet):
    """OpenFlow 패킷 매칭 시뮬레이션.

    패킷은 우선순위 순서로 플로우 엔트리와 매칭됩니다.
    각 테이블 내에서 첫 번째 매칭이 적용됩니다.
    """
    for table in flow_tables:
        matched_entry = None
        best_priority = -1

        for entry in table:
            if matches(packet, entry['match']) and entry['priority'] > best_priority:
                matched_entry = entry
                best_priority = entry['priority']

        if matched_entry:
            # 명령어 실행
            for instruction in matched_entry['instructions']:
                if instruction['type'] == 'output':
                    return {'action': 'forward', 'port': instruction['port']}
                elif instruction['type'] == 'goto_table':
                    break  # 지정된 테이블로 계속
                elif instruction['type'] == 'drop':
                    return {'action': 'drop'}

    # 어떤 테이블에도 매칭 없음 → 테이블 미스
    return {'action': 'send_to_controller'}
```

---

## 3. SDN 컨트롤러

### 3.1 컨트롤러 아키텍처

```
┌───────────────────────────────────────────────────┐
│                  SDN 컨트롤러                       │
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │         노스바운드 API (REST/gRPC)            │  │
│  └─────────────────────────────────────────────┘  │
│                                                   │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐   │
│  │ 토폴로지  │  │  장치    │  │  플로우 규칙   │   │
│  │ 매니저   │  │  매니저  │  │  매니저       │   │
│  └──────────┘  └──────────┘  └───────────────┘   │
│                                                   │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐   │
│  │  통계    │  │  경로    │  │  호스트       │   │
│  │ 수집기   │  │  계산    │  │  추적기       │   │
│  └──────────┘  └──────────┘  └───────────────┘   │
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │     사우스바운드 API (OpenFlow/P4Runtime)    │  │
│  └─────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────┘
```

### 3.2 주요 SDN 컨트롤러

| 컨트롤러 | 언어 | 주요 특징 | 사용 사례 |
|---------|------|----------|---------|
| OpenDaylight (ODL) | Java | 모듈형, YANG 모델, NETCONF/RESTCONF | 엔터프라이즈, 서비스 제공자 |
| ONOS | Java | 분산형, 인텐트 기반, 고가용성 | 캐리어급, WAN |
| Ryu | Python | 경량, 학습 용이, 컴포넌트 기반 | 연구, 프로토타이핑 |
| Floodlight | Java | REST API, OpenStack 연동 | 클라우드 네트워킹 |
| FAUCET | Python | 프로덕션 준비, OpenFlow 1.3, YAML 설정 | 캠퍼스, 엔터프라이즈 |

### 3.3 간단한 컨트롤러 로직

```python
class SimpleSDNController:
    """반응형 포워딩을 보여주는 최소한의 SDN 컨트롤러.

    스위치가 매칭 플로우 없이 패킷을 수신하면
    컨트롤러로 패킷을 보냅니다(PACKET_IN).
    컨트롤러는 처리 방법을 결정하고 플로우 규칙을 설치합니다.
    """

    def __init__(self):
        self.mac_table = {}  # switch_id → {mac → port}
        self.topology = {}   # switch_id → {port → neighbor}

    def handle_packet_in(self, switch_id, in_port, packet):
        """플로우 규칙에 매칭되지 않은 패킷 처리.

        반응형 포워딩의 핵심:
        1. 소스 MAC → 포트 매핑 학습
        2. 목적지 MAC 조회
        3. 알려진 포트로 포워딩하거나 플러딩
        """
        src_mac = packet['src_mac']
        dst_mac = packet['dst_mac']

        # 소스 MAC 주소 학습
        if switch_id not in self.mac_table:
            self.mac_table[switch_id] = {}
        self.mac_table[switch_id][src_mac] = in_port

        # 목적지 조회
        if dst_mac in self.mac_table.get(switch_id, {}):
            out_port = self.mac_table[switch_id][dst_mac]
            # 이후 패킷이 직접 전달되도록 플로우 규칙 설치
            self.install_flow(switch_id, dst_mac, out_port)
            return {'action': 'forward', 'port': out_port}
        else:
            # 목적지 미지 → 소스 포트 제외 전체 포트에 플러딩
            return {'action': 'flood', 'exclude_port': in_port}

    def install_flow(self, switch_id, dst_mac, out_port):
        """스위치에 포워딩 규칙 설치."""
        flow_rule = {
            'match': {'dst_mac': dst_mac},
            'priority': 100,
            'instructions': [{'type': 'output', 'port': out_port}],
            'idle_timeout': 300,  # 5분 유휴 후 제거
        }
        print(f"  스위치 {switch_id}에 플로우 설치: "
              f"{dst_mac} → 포트 {out_port}")
        return flow_rule
```

---

## 4. 네트워크 애플리케이션

### 4.1 SDN 기반 방화벽

```python
class SDNFirewall:
    """SDN 애플리케이션으로 구현된 상태 비저장(stateless) 방화벽.

    IP 주소, 포트, 프로토콜에 따라 트래픽을 차단하거나
    허용하는 플로우 규칙을 설치합니다.
    """

    def __init__(self, controller):
        self.controller = controller
        self.rules = []

    def add_rule(self, src_ip=None, dst_ip=None, protocol=None,
                 dst_port=None, action='deny'):
        """방화벽 규칙 추가."""
        self.rules.append({
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'protocol': protocol,
            'dst_port': dst_port,
            'action': action,
        })

    def evaluate(self, packet):
        """방화벽 규칙에 패킷 확인 (첫 번째 매칭 적용)."""
        for rule in self.rules:
            if self._matches(packet, rule):
                return rule['action']
        return 'allow'  # 기본 허용

    @staticmethod
    def _matches(packet, rule):
        for field in ['src_ip', 'dst_ip', 'protocol', 'dst_port']:
            if rule[field] is not None and packet.get(field) != rule[field]:
                return False
        return True
```

### 4.2 SDN 로드 밸런서

```python
class SDNLoadBalancer:
    """SDN 애플리케이션으로 구현된 라운드 로빈(round-robin) 로드 밸런서.

    플로우 규칙의 목적지 IP/포트를 재작성하여
    백엔드 서버 간 들어오는 연결을 분산시킵니다.
    """

    def __init__(self, vip, backends):
        """
        vip: 가상 IP 주소 (클라이언트가 연결하는 주소)
        backends: {'ip': ..., 'port': ..., 'weight': ...} 목록
        """
        self.vip = vip
        self.backends = backends
        self.current_idx = 0

    def select_backend(self):
        """라운드 로빈 백엔드 선택."""
        backend = self.backends[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.backends)
        return backend

    def create_flow_rules(self, client_ip, client_port):
        """새 연결에 대한 양방향 플로우 규칙 생성."""
        backend = self.select_backend()

        # 순방향: 클라이언트 → VIP를 클라이언트 → 백엔드로 변경
        forward_rule = {
            'match': {
                'src_ip': client_ip,
                'dst_ip': self.vip,
                'src_port': client_port,
            },
            'instructions': [
                {'type': 'set_field', 'field': 'dst_ip',
                 'value': backend['ip']},
                {'type': 'set_field', 'field': 'dst_port',
                 'value': backend['port']},
                {'type': 'output', 'port': 'computed'},
            ],
        }

        # 역방향: 백엔드 → 클라이언트를 VIP → 클라이언트로 변경
        reverse_rule = {
            'match': {
                'src_ip': backend['ip'],
                'dst_ip': client_ip,
                'dst_port': client_port,
            },
            'instructions': [
                {'type': 'set_field', 'field': 'src_ip',
                 'value': self.vip},
                {'type': 'output', 'port': 'computed'},
            ],
        }

        return forward_rule, reverse_rule
```

---

## 5. P4: 프로그래머블 데이터 평면

### 5.1 OpenFlow를 넘어서

OpenFlow는 고정된 매치 필드와 액션 집합을 가집니다. **P4**(Protocol-independent Packet Processors 프로그래밍)는 커스텀 패킷 형식과 처리 로직을 정의할 수 있게 합니다.

```
OpenFlow:                           P4:
  고정된 헤더 필드                    커스텀 헤더 정의
  고정된 매치/액션 파이프라인           프로그래머블 파서
  새 프로토콜 → 새 OF 버전            새 프로토콜 → 새 P4 프로그램
```

### 5.2 P4 프로그램 구조

```
P4 프로그램:

  ┌─────────────────────────────────────┐
  │  1. 헤더 정의                        │
  │     패킷 헤더 형식 정의              │
  │     (Ethernet, IPv4, 커스텀, ...)   │
  │                                     │
  │  2. 파서(Parser)                    │
  │     패킷에서 헤더 추출              │
  │     (상태 머신)                     │
  │                                     │
  │  3. 매치-액션 테이블                 │
  │     테이블 + 액션 정의              │
  │     (OpenFlow와 유사하나 커스텀)    │
  │                                     │
  │  4. 제어 흐름                       │
  │     순서에 따라 테이블 적용         │
  │     (if/else, 테이블 체인)          │
  │                                     │
  │  5. 디파서(Deparser)                │
  │     수정된 헤더로 패킷              │
  │     재조립                          │
  └─────────────────────────────────────┘
```

### 5.3 P4 예시 (개념적)

```
// 인-네트워크 텔레메트리를 위한 커스텀 헤더 정의
header telemetry_t {
    bit<32> switch_id;
    bit<32> ingress_port;
    bit<48> ingress_timestamp;
    bit<32> queue_depth;
}

// 각 스위치는 패킷에 텔레메트리 데이터를 추가합니다
// 패킷이 목적지에 도달할 때쯤이면
// 경로의 모든 스위치에서 수집한 홉별 텔레메트리를 포함합니다
```

---

## 6. 실제 SDN 적용

### 6.1 배포 모델

| 모델 | 설명 | 예시 |
|------|------|------|
| 캠퍼스 SDN | 중앙집중형 스위치 관리 | Cisco DNA Center, Aruba Central |
| 데이터센터 SDN | 가상 네트워킹 오버레이 | VMware NSX, Cisco ACI |
| WAN SDN (SD-WAN) | 소프트웨어 정의 광역 네트워크 | Cisco Viptela, VMware VeloCloud |
| 캐리어 SDN | 서비스 제공자 네트워크 제어 | ONOS, OpenDaylight |

### 6.2 SDN vs 전통 방식: 트레이드오프

| 항목 | 전통 방식 | SDN |
|------|----------|-----|
| 제어 | 분산형, 자율적 | 중앙집중형, 프로그래밍 방식 |
| 확장성 | 각 장치가 독립적으로 확장 | 컨트롤러가 잠재적 병목 |
| 장애 모드 | 점진적 성능 저하 | 컨트롤러 장애 = 네트워크 맹목화 |
| 혁신 속도 | 느림 (벤더 펌웨어) | 빠름 (소프트웨어 정의) |
| 운영 비용 | 높음 (장치별 CLI) | 낮음 (자동화, API) |
| 벤더 유연성 | 벤더 생태계에 종속 | 개방형 표준 가능 |

### 6.3 고가용성

```
컨트롤러 고가용성(HA) 방식:

  1. 액티브-스탠바이(Active-Standby):
     ┌────────────┐     ┌────────────┐
     │ 컨트롤러   │────►│ 컨트롤러   │
     │  (액티브)  │     │ (스탠바이) │
     └────────────┘     └────────────┘
     단순하지만 스탠바이 자원이 낭비됩니다.

  2. 클러스터형 (ONOS 모델):
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │ 컨트롤러   │  │ 컨트롤러   │  │ 컨트롤러   │
     │  노드 1    │──│  노드 2    │──│  노드 3    │
     └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
           │               │               │
     ┌─────┴──┐      ┌─────┴──┐      ┌─────┴──┐
     │스위치들│      │스위치들│      │스위치들│
     │ 존 1  │      │ 존 2  │      │ 존 3  │
     └────────┘      └────────┘      └────────┘
     각 컨트롤러가 자신의 존을 관리하며, Raft를 통해 상태를 공유합니다.
```

---

## 7. 연습문제

### 연습문제 1: 반응형 L2 학습 스위치

반응형 L2 학습 스위치 컨트롤러를 구현하세요:
1. 스위치별 MAC 주소 테이블 유지
2. PACKET_IN 발생 시: 소스 MAC 학습, 목적지 조회
3. 목적지 알려진 경우 → 플로우 규칙 설치, 포워딩
4. 목적지 미지의 경우 → 수신 포트 제외 전체 포트 플러딩
5. 플로우 규칙에 유휴 타임아웃(60초) 추가
6. 4개 스위치 선형 토폴로지로 테스트

### 연습문제 2: SDN 방화벽

SDN 방화벽 애플리케이션을 구축하세요:
1. ACL 규칙 정의 (src_ip, dst_ip, protocol, port로 허용/거부)
2. 허용 트래픽에 대한 사전적(proactive) 플로우 규칙 설치
3. 기본 거부: 허용 규칙에 매칭되지 않는 패킷은 드롭
4. 로깅 추가: 규칙별 차단 패킷 수 카운트
5. 규칙 우선순위 정렬 구현

### 연습문제 3: 최단 경로 라우팅

최단 경로 포워딩을 구현하세요:
1. 스위치 연결에서 토폴로지 그래프 구성 (LLDP 탐색)
2. 모든 호스트 쌍 간 최단 경로 계산 (다익스트라 또는 BFS)
3. 계산된 경로를 따라 사전적 플로우 규칙 설치
4. 토폴로지 변경 처리: 링크 다운 시 경로 재계산
5. 반응형 vs 사전적 설치 지연 시간 비교

### 연습문제 4: SDN 로드 밸런서

라운드 로빈 로드 밸런서를 구현하세요:
1. 가상 IP(VIP) → 백엔드 서버 3개
2. 새 연결: 백엔드 선택, 재작성 규칙 설치
3. 기존 연결: 설치된 규칙 사용 (컨트롤러 개입 없음)
4. 헬스 체크 추가: 장애 백엔드를 순환에서 제거
5. 플로우 규칙 설치 오버헤드 측정

### 연습문제 5: 컨트롤러 확장성

컨트롤러 확장성을 분석하세요:
1. 증가하는 스위치 수 시뮬레이션 (10, 50, 100, 500)
2. 다양한 속도로 PACKET_IN 이벤트 생성
3. 부하에 따른 컨트롤러 응답 시간 측정
4. 병목 지점 파악: CPU, 메모리, 또는 사우스바운드 대역폭
5. 단일 컨트롤러 이상으로 확장하기 위한 전략 논의

---

*레슨 20 끝*

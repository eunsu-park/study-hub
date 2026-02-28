[이전: 소프트웨어 정의 네트워킹](./20_Software_Defined_Networking.md) | [다음: 멀티캐스트](./22_Multicast.md)

---

# 21. 서비스 품질(Quality of Service, QoS)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. QoS가 필요한 이유를 설명하고 IntServ와 DiffServ 아키텍처를 구분한다
2. 네트워크 트래픽을 분류하고 적절한 QoS 클래스에 매핑한다
3. 트래픽 폴리싱(토큰 버킷)과 쉐이핑 알고리즘을 구현한다
4. 큐잉 규칙(FIFO, WFQ, CBWFQ, LLQ)과 그 트레이드오프를 설명한다
5. DSCP 마킹을 적용하고 음성, 비디오, 데이터에 대한 QoS 정책을 설정한다

---

## 목차

1. [QoS가 필요한 이유](#1-qos가-필요한-이유)
2. [트래픽 분류](#2-트래픽-분류)
3. [IntServ vs DiffServ](#3-intserv-vs-diffserv)
4. [트래픽 폴리싱과 쉐이핑](#4-트래픽-폴리싱과-쉐이핑)
5. [큐잉 규칙](#5-큐잉-규칙)
6. [DSCP와 홉별 동작](#6-dscp와-홉별-동작)
7. [QoS 설계 패턴](#7-qos-설계-패턴)
8. [연습문제](#8-연습문제)

---

## 1. QoS가 필요한 이유

### 1.1 최선형(Best-Effort)의 한계

IP 네트워크는 기본적으로 최선형(best-effort) 방식입니다 — 대역폭, 지연, 지터(jitter), 패킷 손실에 대한 보장이 없습니다. 이는 이메일이나 웹 브라우징에는 충분하지만 다음 경우에는 부적합합니다:

```
애플리케이션 요구사항:

  음성 (VoIP):
    지연     < 150ms (단방향)
    지터     < 30ms
    손실     < 1%
    대역폭: 통화당 ~100 Kbps

  화상 회의:
    지연     < 200ms
    지터     < 50ms
    손실     < 0.1% (I-프레임이 중요)
    대역폭: 1-5 Mbps

  파일 전송:
    지연     무관
    지터     무관
    손실     0% (TCP 재전송)
    대역폭: 최대한 많이

QoS 없이: 대용량 파일 전송이 음성 통화를 굶길 수 있습니다.
```

### 1.2 QoS 메커니즘 개요

```
종단간(end-to-end) QoS 파이프라인:

  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ 분류 &   │───►│  폴리싱  │───►│   큐잉   │───►│ 스케줄링 │
  │  마킹    │    │  & 쉐이핑 │    │          │    │  & 전송  │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘

  1. 분류: 트래픽 유형 식별 (음성, 비디오, 데이터)
  2. 마킹: IP 헤더에 DSCP/ToS 비트 설정
  3. 폴리싱/쉐이핑: 합의된 속도로 트래픽 제한
  4. 큐잉: 트래픽 클래스별 별도 큐
  5. 스케줄링: 우선순위/가중치로 큐 서비스
```

---

## 2. 트래픽 분류

### 2.1 분류 방법

| 방법 | 방식 | 장점 | 단점 |
|------|------|------|------|
| 포트 기반 | TCP/UDP 포트 매칭 (80=HTTP, 5060=SIP) | 단순, 빠름 | 신뢰 불가 (포트 변경 가능) |
| 프로토콜 기반 | IP 프로토콜 필드 매칭 | 표준 | 세분화 부족 |
| DSCP 기반 | IP 헤더의 DSCP 마킹 읽기 | 효율적 | 수신 지점의 신뢰 필요 |
| NBAR/DPI | 심층 패킷 검사(Deep Packet Inspection) | 정확 | CPU 집약적, 개인정보 우려 |
| 플로우 기반 | 5-튜플 (src/dst IP, src/dst port, protocol) | 정밀 | 대용량 상태 테이블 |

### 2.2 일반적인 트래픽 클래스

```python
# 일반적인 기업 QoS 분류

QOS_CLASSES = {
    'voice': {
        'dscp': 46,          # EF (신속 포워딩, Expedited Forwarding)
        'bandwidth': '10%',
        'priority': True,     # 엄격한 우선순위 (LLQ)
        'description': 'VoIP RTP 스트림',
    },
    'video': {
        'dscp': 34,          # AF41
        'bandwidth': '30%',
        'priority': False,
        'description': '화상 회의',
    },
    'critical_data': {
        'dscp': 26,          # AF31
        'bandwidth': '25%',
        'priority': False,
        'description': '비즈니스 애플리케이션 (ERP, DB)',
    },
    'best_effort': {
        'dscp': 0,           # BE (기본값)
        'bandwidth': '25%',
        'priority': False,
        'description': '웹, 이메일, 일반 트래픽',
    },
    'scavenger': {
        'dscp': 8,           # CS1
        'bandwidth': '10%',
        'priority': False,
        'description': '백업, 업데이트, 비중요 트래픽',
    },
}
```

---

## 3. IntServ vs DiffServ

### 3.1 통합 서비스(IntServ, Integrated Services)

IntServ는 RSVP(Resource Reservation Protocol, 자원 예약 프로토콜)를 사용하여 플로우별 보장을 제공합니다:

```
RSVP를 사용한 IntServ:

  송신자 ─────────────────────────────────► 수신자
          ←─── RSVP PATH 메시지 ────────
          ────── RSVP RESV 메시지 ──────►

  경로상 각 라우터:
    1. 예약 요청 수신
    2. 자원 가용 여부 확인
    3. 이 플로우를 위한 대역폭/버퍼 예약
    4. 플로우별 상태 유지

  보장: 이 특정 플로우에 대한 대역폭 및 지연 한계.
```

**문제점**: IntServ는 확장성이 없습니다 — 라우터가 모든 플로우의 상태를 유지해야 합니다.

### 3.2 차별화 서비스(DiffServ, Differentiated Services)

DiffServ는 DSCP 마킹을 사용하여 플로우별이 아닌 클래스별 처리를 제공합니다:

```
DiffServ:

  에지 라우터:                    코어 라우터:
  ┌──────────────────────┐      ┌──────────────────────┐
  │ 트래픽 분류          │      │ DSCP 마킹 읽기       │
  │ 헤더에 DSCP 마킹     │─────►│ 홉별 동작 적용       │
  │ 수신 시 폴리싱       │      │ (플로우별 상태 없음)  │
  └──────────────────────┘      └──────────────────────┘

  확장성: 코어 라우터는 6비트 DSCP 필드만 봅니다.
  플로우별 상태 불필요.
```

### 3.3 비교

| 특징 | IntServ | DiffServ |
|------|---------|---------|
| 세분화 | 플로우별 | 클래스별 |
| 보장 | 강성 (대역폭, 지연) | 연성 (상대적 우선순위) |
| 확장성 | 낮음 (코어의 플로우별 상태) | 우수 (상태 없는 코어) |
| 시그널링 | RSVP (복잡) | DSCP 마킹 (단순) |
| 배포 | 드묾 (너무 복잡) | 표준 (광범위 배포) |

---

## 4. 트래픽 폴리싱과 쉐이핑

### 4.1 토큰 버킷(Token Bucket) 알고리즘

폴리싱과 쉐이핑 모두 토큰 버킷을 사용합니다:

```
토큰 버킷:

  토큰이 속도 r (토큰/초)로 도착
  버킷은 최대 b개의 토큰 수용 (버스트 크기)

  ┌──────────────┐
  │ 토큰 버킷    │ ← 속도 r로 토큰 도착
  │  ┌────────┐  │
  │  │████████│  │ 최대 용량 = b 토큰
  │  │████████│  │
  │  │████    │  │ 현재 토큰
  │  └────────┘  │
  └──────┬───────┘
         │
    ┌────┴────┐
    │  패킷   │  각 패킷은 자신의 크기만큼 토큰을 소비합니다.
    │  도착   │  토큰 충분 → 적합(conform) (전송/녹색 마킹)
    └─────────┘  토큰 부족 → 초과(exceed) (드롭/적색 마킹)
```

### 4.2 구현

```python
import time


class TokenBucket:
    """트래픽 폴리싱/쉐이핑을 위한 토큰 버킷.

    CIR(Committed Information Rate): 지속 속도
    CBS(Committed Burst Size): 최대 버스트 크기 (바이트)
    """

    def __init__(self, cir_bps, cbs_bytes):
        self.cir = cir_bps          # 초당 바이트
        self.cbs = cbs_bytes        # 버킷 용량
        self.tokens = cbs_bytes     # 가득 찬 상태로 시작
        self.last_time = time.time()

    def consume(self, packet_size):
        """패킷에 대한 토큰 소비 시도.

        토큰 가용 시 'conform' 반환, 그렇지 않으면 'exceed' 반환.
        """
        now = time.time()
        elapsed = now - self.last_time
        self.last_time = now

        # 경과 시간에 따라 토큰 추가
        self.tokens = min(
            self.cbs,
            self.tokens + elapsed * self.cir
        )

        if self.tokens >= packet_size:
            self.tokens -= packet_size
            return 'conform'
        else:
            return 'exceed'


class TrafficShaper:
    """트래픽 쉐이퍼: 초과 패킷을 드롭하지 않고 지연시킵니다.

    폴리싱(드롭)과 달리, 쉐이핑은 패킷을 버퍼에 저장했다가
    토큰이 가용해지면 전송합니다.
    """

    def __init__(self, cir_bps, cbs_bytes, buffer_size=100):
        self.bucket = TokenBucket(cir_bps, cbs_bytes)
        self.buffer = []
        self.buffer_size = buffer_size

    def enqueue(self, packet):
        result = self.bucket.consume(packet['size'])
        if result == 'conform':
            return 'send', packet
        else:
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(packet)
                return 'buffered', None
            else:
                return 'dropped', None  # 버퍼 가득 참

    def dequeue(self):
        """토큰이 가용해지면 버퍼링된 패킷 전송."""
        sent = []
        remaining = []
        for pkt in self.buffer:
            if self.bucket.consume(pkt['size']) == 'conform':
                sent.append(pkt)
            else:
                remaining.append(pkt)
        self.buffer = remaining
        return sent
```

### 4.3 폴리싱 vs 쉐이핑

| 항목 | 폴리싱 | 쉐이핑 |
|------|--------|--------|
| 한계 초과 시 동작 | 드롭 또는 재마킹 | 버퍼링 및 지연 |
| 지연 영향 | 없음 (버퍼링 없음) | 지연 추가 |
| 버퍼 필요 | 없음 | 있음 |
| 패킷 손실 | 높음 | 낮음 (버퍼 넘침 제외) |
| 사용 사례 | 수신 측 강제 적용 | 송신 측 평활화 |

---

## 5. 큐잉 규칙

### 5.1 FIFO (선입선출, First In, First Out)

기본값: 모든 패킷이 단일 큐에서 도착 순서대로 서비스됩니다. 차별화 없음.

### 5.2 우선순위 큐잉(Priority Queuing, PQ)

```
우선순위 큐잉:

  높은 우선순위: ████████ → 항상 먼저 서비스
  중간 우선순위: ██████████████
  낮은 우선순위: ████████████████████████

  문제: 기아(starvation) — 높은 우선순위 트래픽이
  지속적이면 낮은 우선순위는 서비스되지 않을 수 있습니다.
```

### 5.3 가중 공정 큐잉(WFQ, Weighted Fair Queuing)

```python
class WeightedFairQueue:
    """가중 공정 큐잉: 비례적 대역폭 할당.

    각 큐는 가중치에 비례한 대역폭을 받습니다.
    차별화를 제공하면서 기아 현상을 방지합니다.
    """

    def __init__(self, weights):
        """weights: queue_name → weight (높을수록 더 많은 대역폭) 딕셔너리"""
        self.queues = {name: [] for name in weights}
        self.weights = weights
        self.total_weight = sum(weights.values())

    def enqueue(self, queue_name, packet):
        self.queues[queue_name].append(packet)

    def schedule(self, num_slots):
        """다음 스케줄링 라운드에서 서비스할 큐를 결정합니다.

        전송할 (queue_name, packet) 목록을 반환합니다.
        """
        result = []
        for name, weight in self.weights.items():
            # 각 큐는 가중치에 비례한 슬롯을 받습니다
            slots = max(1, round(num_slots * weight / self.total_weight))
            for _ in range(slots):
                if self.queues[name]:
                    result.append((name, self.queues[name].pop(0)))
        return result
```

### 5.4 저지연 큐잉(LLQ)을 포함한 클래스 기반 WFQ (CBWFQ)

```
LLQ/CBWFQ (가장 일반적인 기업 QoS):

  ┌────────────────────────────────────────────┐
  │               스케줄러                      │
  │                                            │
  │  ┌──────────────────┐  엄격한 우선순위      │
  │  │ 음성 (LLQ)       │──────────────────►   │
  │  │ DSCP EF          │  항상 먼저 서비스     │
  │  │ 10%로 폴리싱     │                      │
  │  └──────────────────┘                      │
  │                                            │
  │  ┌──────────────────┐  가중 공정            │
  │  │ 비디오 (CBWFQ)   │──────────────────►   │
  │  │ DSCP AF41        │  30% 대역폭           │
  │  └──────────────────┘                      │
  │                                            │
  │  ┌──────────────────┐                      │
  │  │ 중요 데이터      │──────────────────►   │
  │  │ (CBWFQ)          │  25% 대역폭           │
  │  │ DSCP AF31        │                      │
  │  └──────────────────┘                      │
  │                                            │
  │  ┌──────────────────┐                      │
  │  │ 최선형           │──────────────────►   │
  │  │ DSCP 0           │  나머지 대역폭        │
  │  └──────────────────┘                      │
  └────────────────────────────────────────────┘

  LLQ = CBWFQ + 음성을 위한 엄격한 우선순위.
  음성은 항상 먼저 전송되지만, 기아 방지를 위해 폴리싱됩니다.
```

---

## 6. DSCP와 홉별 동작

### 6.1 DSCP 필드

DSCP(Differentiated Services Code Point, 차별화 서비스 코드 포인트)는 IP ToS 바이트의 상위 6비트를 사용합니다:

```
IP 헤더 ToS 바이트:
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ D │ S │ C │ P │ 3 │ 2 │ 1 │ 0 │
│   │   │   │   │   │   │   │   │
│←──── DSCP (6비트) ────→│ECN│
└───┴───┴───┴───┴───┴───┴───┴───┘
```

### 6.2 표준 홉별 동작(PHB, Per-Hop Behavior)

| PHB | DSCP 값 | 바이너리 | 목적 |
|-----|---------|---------|------|
| 기본값 (BE) | 0 | 000000 | 최선형 트래픽 |
| CS1 (스캐빈저) | 8 | 001000 | 최선형 이하 (대량) |
| AF11 | 10 | 001010 | 보증 포워딩 클래스 1, 낮은 드롭 |
| AF21 | 18 | 010010 | 보증 포워딩 클래스 2, 낮은 드롭 |
| AF31 | 26 | 011010 | 보증 포워딩 클래스 3, 낮은 드롭 |
| AF41 | 34 | 100010 | 보증 포워딩 클래스 4, 낮은 드롭 |
| EF | 46 | 101110 | 신속 포워딩 (음성) |
| CS6 | 48 | 110000 | 네트워크 제어 |

### 6.3 보증 포워딩(AF) 매트릭스

```
AF 클래스는 각각 3가지 드롭 우선순위를 가집니다:

         │ 낮은 드롭 (1) │ 중간 드롭 (2) │ 높은 드롭 (3)
─────────┼───────────────┼───────────────┼──────────────
클래스 1  │  AF11 (10)    │   AF12 (12)   │  AF13 (14)
클래스 2  │  AF21 (18)    │   AF22 (20)   │  AF23 (22)
클래스 3  │  AF31 (26)    │   AF32 (28)   │  AF33 (30)
클래스 4  │  AF41 (34)    │   AF42 (36)   │  AF43 (38)

혼잡 시 높은 드롭 패킷이 먼저 폐기됩니다.
이를 통해 WRED(가중 무작위 조기 감지)가 클래스 내에서
낮은 우선순위 트래픽을 선택적으로 드롭할 수 있습니다.
```

---

## 7. QoS 설계 패턴

### 7.1 기업 캠퍼스 QoS

```
신뢰 경계(Trust Boundary):
  전화기 → 신뢰됨 (EF 마킹)
  PC     → 신뢰 안 됨 (액세스 스위치에서 재마킹)

  액세스 스위치:  분류 + 마킹 + 폴리싱
  배포 계층:      필요 시 재마킹
  코어:           마킹 준수, 우선순위 큐
  WAN 에지:       링크 속도로 쉐이핑, LLQ/CBWFQ
```

### 7.2 VoIP QoS 체크리스트

| 요구사항 | 설정 |
|---------|------|
| 마킹 | 음성 RTP에 DSCP EF (46) |
| 시그널링 | SIP/SCCP에 DSCP CS3 (24) |
| 우선순위 큐 | 링크의 10-20%로 폴리싱된 LLQ |
| 지터 버퍼 | 엔드포인트에서 30-50ms |
| 최대 지연 | 종단간 150ms |
| 대역폭 | G.711 통화당 ~100 Kbps |

---

## 8. 연습문제

### 연습문제 1: 토큰 버킷 시뮬레이터

단일 속도 토큰 버킷을 구현하고 테스트하세요:
1. CIR = 1 Mbps, CBS = 10 KB
2. 다양한 속도로 다양한 크기의 패킷 입력
3. 시간에 따른 적합(conform) vs 초과(exceed) 결정 그래프
4. CBS 변경에 따른 버스트 허용량 변화 표시

### 연습문제 2: WFQ 스케줄러

가중 공정 큐잉을 구현하세요:
1. 가중치가 있는 4개 큐 생성: 음성=40, 비디오=30, 데이터=20, 벌크=10
2. 각 큐에 패킷 생성
3. 100개 시간 슬롯 스케줄링 후 큐별 서비스된 패킷 수 카운트
4. 대역폭 분배가 가중치와 일치하는지 확인

### 연습문제 3: DiffServ 분류기

트래픽 분류기를 구축하세요:
1. 5-튜플(src/dst IP, 포트, 프로토콜)로 패킷 분류
2. 분류 규칙에 따라 DSCP 값 할당
3. 신뢰 경계 구현: 신뢰되지 않는 트래픽 재마킹
4. DSCP 클래스별 패킷 수 카운트

### 연습문제 4: WRED 시뮬레이션

가중 무작위 조기 감지(WRED, Weighted Random Early Detection)를 시뮬레이션하세요:
1. 최소 임계값과 최대 임계값이 있는 큐 구현
2. 큐 깊이가 최소와 최대 사이일 때 증가하는 확률로 무작위 드롭
3. 높은 드롭 우선순위(AF13)는 낮은 드롭(AF11)보다 낮은 최소 임계값 설정
4. WRED가 테일 드롭 동기화를 줄이는 것을 보여주기

### 연습문제 5: 종단간 QoS 분석

QoS가 적용된 3홉 네트워크를 모델링하세요:
1. 각 링크: 10 Mbps, LLQ/CBWFQ 설정
2. 혼합 트래픽 생성: 음성(EF), 비디오(AF41), 데이터(BE)
3. 각 클래스에 대한 홉별 지연, 지터, 손실 측정
4. 혼잡 상황에서도 음성이 종단간 150ms 미만을 유지함을 증명

---

*레슨 21 끝*

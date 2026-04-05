# 레슨 13: 장애 감지와 그룹 멤버십

[개요](./00_Overview.md) | [이전: 분산 스토리지 사례 연구](./12_Distributed_Storage_Case_Studies.md) | [다음: 분산 조정 프리미티브](./14_Distributed_Coordination_Primitives.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 장애 감지기에서 완전성과 정확성 사이의 근본적인 트레이드오프를 분석한다
2. Jacobson/Karels 알고리즘을 사용하여 적응형 하트비트 기반 장애 감지를 구현한다
3. 조정 가능한 의심 임계값을 가진 phi accrual 장애 감지기를 설계하고 구현한다
4. SWIM 프로토콜의 3단계 감지 메커니즘과 감염 방식 전파를 설명한다
5. 가십 기반 멤버십 프로토콜과 그 수렴 보장을 비교한다

---

## 목차

1. [장애 감지 문제](#1-장애-감지-문제)
2. [하트비트 기반 감지](#2-하트비트-기반-감지)
3. [Phi Accrual 장애 감지기](#3-phi-accrual-장애-감지기)
4. [SWIM 프로토콜](#4-swim-프로토콜)
5. [가십 프로토콜](#5-가십-프로토콜)
6. [그룹 멤버십](#6-그룹-멤버십)
7. [구현: Phi Accrual 감지기](#7-구현-phi-accrual-감지기)
8. [구현: SWIM 프로토콜 시뮬레이터](#8-구현-swim-프로토콜-시뮬레이터)
9. [요약 및 추가 자료](#9-요약-및-추가-자료)

---

## 1. 장애 감지 문제

### 1.1 장애 감지가 어려운 이유

분산 시스템에서는 느린 프로세스와 충돌한 프로세스를 신뢰성 있게 구별할 수 없습니다. 이 관찰은 장애 감지의 모든 도전 과제의 핵심입니다. 메시지가 지연되거나, 손실되거나, 수신자가 실제로 장애가 발생했을 수 있으며 — 비동기 네트워크에서 발신자는 그 차이를 알 방법이 없습니다.

```
┌─────────┐     message     ┌─────────┐
│ Node A  │ ───────────────▶│ Node B  │
│         │                 │ (slow?) │
│         │  no response    │(crashed?)│
│         │◀── timeout ──── │(network?)│
└─────────┘                 └─────────┘

Three explanations for silence:
  1. Node B crashed
  2. Node B is alive but slow
  3. Network partitioned between A and B
```

### 1.2 장애 감지기의 형식적 속성

Chandra와 Toueg(1996)는 두 가지 핵심 속성으로 장애 감지기를 형식화했습니다:

| 속성 | 정의 |
|------|------|
| **강한 완전성 (Strong Completeness)** | 결국 충돌한 모든 프로세스는 모든 올바른 프로세스에 의해 영구적으로 의심됨 |
| **약한 완전성 (Weak Completeness)** | 결국 충돌한 모든 프로세스는 일부 올바른 프로세스에 의해 영구적으로 의심됨 |
| **강한 정확성 (Strong Accuracy)** | 올바른 프로세스는 절대 의심되지 않음 |
| **약한 정확성 (Weak Accuracy)** | 일부 올바른 프로세스는 절대 의심되지 않음 |
| **최종적 강한 정확성 (Eventual Strong Accuracy)** | 알 수 없는 시간 이후, 올바른 프로세스는 의심되지 않음 |
| **최종적 약한 정확성 (Eventual Weak Accuracy)** | 알 수 없는 시간 이후, 일부 올바른 프로세스는 절대 의심되지 않음 |

### 1.3 장애 감지기 클래스

완전성과 정확성 속성의 조합은 8가지 클래스의 장애 감지기를 정의합니다:

```
                    Strong Accuracy    Weak Accuracy    Eventual Strong    Eventual Weak
                                                        Accuracy           Accuracy
Strong Completeness      P (Perfect)      S (Strong)      ◇P (Ev.Perfect)   ◇S (Ev.Strong)
Weak Completeness        Q                W               ◇Q                 ◇W
```

**핵심 통찰**: 약한 완전성은 강한 완전성으로 변환될 수 있습니다. 한 프로세스가 충돌한 프로세스를 의심하면, 이 의심을 다른 모든 프로세스에 가십할 수 있습니다. 따라서 실질적으로 중요한 클래스는 다음과 같습니다:

- **P (Perfect)**: 최고 기준 — 절대 틀리지 않고, 항상 감지합니다. 동기 시스템에서만 달성 가능합니다.
- **◇P (Eventually Perfect)**: 처음에는 실수할 수 있지만, 결국 안정화됩니다. 합의에 충분합니다.
- **◇S (Eventually Strong)**: 어느 시점 이후, 적어도 하나의 올바른 프로세스가 의심되지 않습니다. **이것이 합의를 풀기에 충분한 가장 약한 장애 감지기입니다** (Chandra-Toueg 결과).

### 1.4 FLP과의 연결

레슨 3(FLP 불가능성)을 상기하세요: 단 하나의 충돌 장애가 있는 순수 비동기 시스템에서 합의는 불가능합니다. 장애 감지기는 FLP를 우회하는 데 필요한 추가적인 능력을 제공합니다:

```
Synchronous system  ──▶  Perfect failure detector (P)  ──▶  Consensus trivially solvable
Asynchronous system ──▶  No failure detector possible   ──▶  FLP impossibility
Partially synchronous  ──▶  ◇S failure detector          ──▶  Consensus solvable
```

이것이 실제 시스템이 타임아웃을 사용하는 이유입니다 — 불완전한 장애 감지기(◇S 클래스)를 구현하며, 이는 Paxos와 Raft 같은 합의 프로토콜에 충분합니다.

### 1.5 근본적인 트레이드오프

모든 장애 감지기는 본질적인 긴장에 직면합니다:

```
Detection Speed ◀──────────────────▶ False Positive Rate

Short timeout:                        Long timeout:
  + Fast detection of real failures     + Fewer false positives
  - Many false positives                - Slow detection
  - Unnecessary failovers               + Stable system
  - Wasted resources on recovery        - Delayed recovery start
```

**수학적 공식화**: 다음을 정의합니다:
- T_d = 감지 시간 (충돌부터 감지까지의 시간)
- P_fp = 오탐 확률
- λ = 메시지 손실률
- σ = 네트워크 지연 지터

고정 타임아웃 τ에 대해:
- T_d ≈ τ (감지 시간은 타임아웃에 의해 제한됨)
- P_fp ≈ P(delay > τ)이며, τ가 증가하면 감소함

최적 타임아웃은 비용 모델에 따라 달라집니다: C_total = C_miss × T_d + C_fp × P_fp.

---

## 2. 하트비트 기반 감지

### 2.1 고정 타임아웃 감지

가장 단순한 접근 방식: 주기적으로 하트비트를 보내고, 고정 타임아웃 내에 하트비트가 도착하지 않으면 노드를 의심합니다.

```python
import time
import threading
from typing import Dict, Optional

class FixedTimeoutDetector:
    """Simple fixed-timeout failure detector."""

    def __init__(self, timeout: float = 5.0, heartbeat_interval: float = 1.0):
        self.timeout = timeout
        self.heartbeat_interval = heartbeat_interval
        # Last heartbeat received from each node
        self.last_heartbeat: Dict[str, float] = {}
        self.suspected: set = set()
        self.lock = threading.Lock()

    def receive_heartbeat(self, node_id: str) -> None:
        """Record reception of heartbeat from a node."""
        with self.lock:
            self.last_heartbeat[node_id] = time.monotonic()
            # If we previously suspected this node, clear the suspicion
            self.suspected.discard(node_id)

    def check_nodes(self) -> set:
        """Check all known nodes and return the set of suspected nodes."""
        now = time.monotonic()
        with self.lock:
            for node_id, last_time in self.last_heartbeat.items():
                if now - last_time > self.timeout:
                    self.suspected.add(node_id)
                else:
                    self.suspected.discard(node_id)
            return set(self.suspected)

    def is_alive(self, node_id: str) -> bool:
        """Check if a specific node is considered alive."""
        with self.lock:
            if node_id not in self.last_heartbeat:
                return False
            return (time.monotonic() - self.last_heartbeat[node_id]) <= self.timeout
```

**고정 타임아웃의 한계**:
- 변화하는 네트워크 조건에 적응할 수 없음
- LAN에 맞춘 타임아웃은 WAN에서 오탐을 발생시킴
- WAN에 맞춘 타임아웃은 LAN에서 너무 느림
- 네트워크 혼잡이 일시적으로 지연 시간을 증가시켜 거짓 의심을 유발함

### 2.2 적응형 타임아웃: Jacobson/Karels 알고리즘

원래 TCP 재전송 타임아웃(RFC 6298)을 위해 설계된 이 알고리즘은 관측된 왕복 시간에 기반하여 타임아웃을 조정합니다. 핵심 아이디어는 하트비트 간격의 평균과 분산 모두에 대해 지수 가중 이동 평균(EWMA)을 유지하는 것입니다.

**알고리즘**:

새로운 샘플 RTT `R`이 주어졌을 때:

```
SRTT ← (1 - α) × SRTT + α × R          (smoothed RTT)
RTTVAR ← (1 - β) × RTTVAR + β × |SRTT - R|   (RTT variation)
RTO ← SRTT + K × RTTVAR                 (retransmission timeout)
```

여기서:
- α = 1/8 (평균에 대한 평활 계수)
- β = 1/4 (분산에 대한 평활 계수)
- K = 4 (안전 마진 승수)

```python
class AdaptiveTimeoutDetector:
    """Failure detector using TCP-style adaptive timeout (Jacobson/Karels)."""

    def __init__(self, alpha: float = 0.125, beta: float = 0.25, k: float = 4.0):
        self.alpha = alpha  # Smoothing factor for SRTT
        self.beta = beta    # Smoothing factor for RTTVAR
        self.k = k          # Safety margin multiplier

        # Per-node state
        self.srtt: Dict[str, float] = {}        # Smoothed RTT
        self.rttvar: Dict[str, float] = {}      # RTT variance
        self.timeout: Dict[str, float] = {}     # Computed timeout
        self.last_heartbeat: Dict[str, float] = {}
        self.suspected: set = set()
        self.lock = threading.Lock()

    def receive_heartbeat(self, node_id: str) -> None:
        """Process a heartbeat and update adaptive timeout."""
        now = time.monotonic()
        with self.lock:
            if node_id in self.last_heartbeat:
                # Compute observed interval
                interval = now - self.last_heartbeat[node_id]

                if node_id not in self.srtt:
                    # First measurement: initialize
                    self.srtt[node_id] = interval
                    self.rttvar[node_id] = interval / 2.0
                else:
                    # Jacobson/Karels update
                    self.rttvar[node_id] = (
                        (1 - self.beta) * self.rttvar[node_id]
                        + self.beta * abs(self.srtt[node_id] - interval)
                    )
                    self.srtt[node_id] = (
                        (1 - self.alpha) * self.srtt[node_id]
                        + self.alpha * interval
                    )

                # Compute new timeout
                self.timeout[node_id] = (
                    self.srtt[node_id] + self.k * self.rttvar[node_id]
                )

            self.last_heartbeat[node_id] = now
            self.suspected.discard(node_id)

    def get_timeout(self, node_id: str) -> float:
        """Get the current adaptive timeout for a node."""
        with self.lock:
            return self.timeout.get(node_id, 5.0)  # Default 5s if unknown

    def check_nodes(self) -> set:
        """Check all nodes using their individual adaptive timeouts."""
        now = time.monotonic()
        with self.lock:
            for node_id, last_time in self.last_heartbeat.items():
                node_timeout = self.timeout.get(node_id, 5.0)
                if now - last_time > node_timeout:
                    self.suspected.add(node_id)
                else:
                    self.suspected.discard(node_id)
            return set(self.suspected)
```

### 2.3 감지 속도 vs 오탐률

다음 표는 실제 수치로 트레이드오프를 보여줍니다:

| 시나리오 | 평균 지연 | 표준편차 | 타임아웃 | 감지 시간 | 오탐률 |
|----------|-----------|----------|----------|-----------|--------|
| LAN, tight | 0.5ms | 0.1ms | 2ms | ~2ms | ~0.001% |
| LAN, loose | 0.5ms | 0.1ms | 10ms | ~10ms | ~0% |
| WAN, tight | 50ms | 20ms | 100ms | ~100ms | ~1% |
| WAN, loose | 50ms | 20ms | 300ms | ~300ms | ~0.001% |
| Cloud, tight | 5ms | 10ms | 30ms | ~30ms | ~2% |
| Cloud, loose | 5ms | 10ms | 100ms | ~100ms | ~0.01% |

**핵심 관찰**:
- 클라우드 환경은 높은 지터(분산)를 가지므로 고정 타임아웃이 특히 문제가 됨
- 적응형 타임아웃은 환경 간 차이를 자연스럽게 처리함
- Jacobson/Karels의 K=4 안전 마진은 오탐률을 매우 낮추지만 감지가 더 느림

---

## 3. Phi Accrual 장애 감지기

### 3.1 핵심 아이디어

Phi accrual 장애 감지기(Hayashibara et al., 2004)는 근본적으로 다른 접근 방식을 나타냅니다: 이진 alive/dead 결정 대신, φ(phi)라 불리는 **연속적인 의심 수준**을 출력합니다. 그런 다음 애플리케이션이 자신의 요구사항에 따라 어떤 임계값이 "의심됨"을 구성하는지 결정합니다.

```
Traditional detector:    alive ──────────|──────────── dead
                                      timeout

Phi accrual detector:    alive ═══════════════════════▶ dead
                         φ=0   φ=1   φ=3   φ=5   φ=10  φ=∞
                                       │           │
                                  moderate      very
                                  suspicion    suspicious
```

### 3.2 수학적 기초

phi 값은 관측된 하트비트 이력을 고려했을 때, 모니터링되는 프로세스가 충돌하지 않았을 음의 로그 확률을 나타냅니다.

**단계 1**: 도착 간 시간(연속 하트비트 사이의 시간)의 슬라이딩 윈도우를 유지합니다:

```
t₁, t₂, t₃, ..., tₙ   (recent inter-arrival times)
```

**단계 2**: 이를 분포에 맞춥니다. 원 논문은 **정규 분포**를 가정합니다:

```
μ = mean(t₁, ..., tₙ)
σ² = variance(t₁, ..., tₙ)
```

**단계 3**: 마지막 하트비트 이후 시간 `t_now`이 주어졌을 때, 하트비트가 단지 늦는 것일 확률을 계산합니다(프로세스가 충돌한 것이 아니라):

```
P_later(t) = 1 - F(t) = 1 - Φ((t - μ) / σ)
```

여기서 Φ는 표준 정규 CDF입니다.

**단계 4**: phi를 계산합니다:

```
φ(t) = -log₁₀(P_later(t))
```

**phi 값의 해석**:

| φ 값 | P(단지 늦음) | 해석 |
|------|-------------|------|
| 0 | 100% | 방금 하트비트를 수신함 |
| 1 | 10% | 아마 아직 살아있음 |
| 2 | 1% | 의심스러워짐 |
| 3 | 0.1% | 충돌했을 가능성이 매우 높음 |
| 4 | 0.01% | 거의 확실히 충돌함 |
| 8 | 0.000001% | 확실히 충돌함 |

### 3.3 임계값 선택

임계값은 감지 속도와 정확성 사이의 트레이드오프를 결정합니다:

| 환경 | 권장 φ 임계값 | 근거 |
|------|--------------|------|
| LAN | 8 | 낮은 지연 분산, 높은 임계값 허용 가능 |
| WAN | 3-5 | 높은 분산으로 낮은 임계값 필요 |
| Cloud | 5-8 | 제공자의 네트워크 안정성에 따라 다름 |
| 데이터센터 간 | 3-4 | 높고 변동이 큰 지연 시간 |

**Cassandra**는 기본적으로 φ = 8을 사용합니다(`phi_convict_threshold`로 설정 가능).
**Akka**는 기본적으로 φ = 8을 사용합니다(`akka.cluster.failure-detector.threshold`로 설정 가능).

### 3.4 이진 감지기 대비 장점

1. **애플리케이션과 분리**: 감지기가 정보를 제공하고, 애플리케이션이 결정함
2. **자체 조정**: 슬라이딩 윈도우를 통해 네트워크 조건에 적응함
3. **설정 가능한 정확도**: 동일한 감지기에서 서로 다른 서비스가 다른 임계값을 사용할 수 있음
4. **점진적 의심**: 점진적 대응을 가능하게 함 (예: φ=3에서 새 요청 전송 중지, φ=8에서 마이그레이션 시작)

### 3.5 정규 분포 가정

원 논문은 하트비트 도착 간 시간이 정규 분포를 따른다고 가정합니다. 이는 종종 합리적인 근사이지만, 실제로는:

```
Real distribution of inter-arrival times:

                    ╭───────╮
                   ╱│        ╲
                  ╱ │         ╲
                 ╱  │          ╲──────── Long tail (GC pauses,
                ╱   │           ╲        network congestion)
    ───────────╱    │            ╲───────────────────
              μ-2σ  μ   μ+2σ    μ+4σ

    Normal approximation works well for the bulk,
    but underestimates tail probability.
```

**Cassandra의 개선**: 정규 분포 대신 지수 분포를 사용하여 우측 꼬리를 더 잘 모델링합니다. phi 계산은 다음과 같아집니다:

```
φ(t) = t / mean_interval × log₁₀(e)
```

이것은 더 단순하고 가끔 늦는 하트비트를 더 우아하게 처리합니다.

---

## 4. SWIM 프로토콜

### 4.1 개요

SWIM(Scalable Weakly-consistent Infection-style Process Group Membership Protocol, Das et al. 2002)은 두 가지 문제를 동시에 해결합니다:

1. **장애 감지**: 충돌한 멤버를 감지
2. **멤버십 전파**: 멤버십 변경 사항을 전파

전통적인 전체-대-전체 하트비트는 프로토콜 기간당 O(n²) 메시지를 생성합니다. SWIM은 강한 완전성을 유지하면서 멤버당 프로토콜 기간당 O(1) 메시지 부하를 달성합니다.

```
Traditional heartbeating:          SWIM protocol:

  1 ──▶ 2                           1 ──ping──▶ 2
  1 ──▶ 3                           1 ◀──ack─── 2
  1 ──▶ 4                           (next period: 1 pings 3)
  2 ──▶ 1
  2 ──▶ 3                           Message load per node: O(1)
  2 ──▶ 4                           Total messages: O(n)
  3 ──▶ 1
  3 ──▶ 2
  3 ──▶ 4
  4 ──▶ 1
  4 ──▶ 2
  4 ──▶ 3

  Messages: O(n²) per period
```

### 4.2 3단계 감지

각 프로토콜 기간마다, 노드 M_i는 다음을 수행합니다:

```
Phase 1: Direct Ping
┌──────┐    ping    ┌──────┐
│ M_i  │───────────▶│ M_j  │
│      │◀───────────│      │
│      │    ack     │      │
└──────┘            └──────┘
If ack received → M_j is alive. Done for this period.

Phase 2: Indirect Ping (if no ack from Phase 1)
┌──────┐  ping-req  ┌──────┐   ping   ┌──────┐
│ M_i  │───────────▶│ M_k1 │─────────▶│ M_j  │
│      │───────────▶│ M_k2 │─────────▶│      │
│      │───────────▶│ M_k3 │─────────▶│      │
│      │            └──────┘◀──────── │      │
│      │◀── ack ────│ M_k2 │   ack    │      │
└──────┘            └──────┘          └──────┘
M_i selects K random members (k1, k2, k3) and asks them to ping M_j.
If any ack arrives → M_j is alive. Done.

Phase 3: Suspect (if no ack from Phase 2)
┌──────┐
│ M_i  │──── marks M_j as SUSPECT
│      │──── disseminates {suspect, M_j, incarnation} via piggyback
└──────┘
```

### 4.3 의심 메커니즘

SWIM은 노드를 즉시 dead로 선언하지 않습니다. 대신 **의심 하위 프로토콜**에 진입합니다:

```
Timeline for node M_j:

  ──────|──────────────|──────────────|──────────────▶ time
     suspect         suspicion       confirmed
     starts          timeout          dead

  During suspicion period:
  - M_j can refute by sending an ALIVE message
    with a higher incarnation number
  - Other nodes can corroborate suspicion
  - If timeout expires without refutation → M_j is declared dead
```

**Incarnation 번호**: 각 노드는 자신의 incarnation 번호를 유지합니다. 노드가 자신이 의심받고 있음을 들으면, incarnation 번호를 증가시키고 ALIVE 메시지를 브로드캐스트합니다. 더 높은 incarnation 번호를 가진 메시지가 이전 것을 덮어씁니다:

```
Priority order (highest to lowest):
  {dead, M_j, inc_n}     > any {suspect/alive} for M_j
  {suspect, M_j, inc_n}  > {alive, M_j, inc_m} if n > m
  {alive, M_j, inc_n}    > {suspect, M_j, inc_m} if n > m
  {alive, M_j, inc_n}    > {alive, M_j, inc_m} if n > m
```

### 4.4 감염 방식 전파

멤버십 업데이트를 전파하기 위해 별도의 프로토콜을 사용하는 대신, SWIM은 장애 감지 메시지(ping, ping-req, ack)에 업데이트를 피기백합니다:

```python
# Each SWIM message carries piggyback updates
class SwimMessage:
    def __init__(self, msg_type, target, sender):
        self.type = msg_type          # "ping", "ping-req", "ack"
        self.target = target
        self.sender = sender
        self.piggyback = []           # List of membership updates

    def add_piggyback(self, update):
        """Attach a membership update to this message."""
        self.piggyback.append(update)
```

각 멤버십 업데이트는 최대 `λ × log(n)` 메시지에 피기백됩니다. 여기서 λ는 설정 가능한 프로토콜 매개변수입니다. 이것은 다음을 제공합니다:

- **전파 시간**: 모든 멤버가 업데이트를 배우는 데 O(log n) 프로토콜 기간
- **메시지 오버헤드**: 각 메시지는 제한된 수의 피기백 항목을 전달
- **추가 메시지 없음**: 멤버십 전파를 위한 추가 네트워크 비용 제로

### 4.5 속성과 보장

| 속성 | 보장 |
|------|------|
| 감지 시간 | 기대값으로 O(protocol_period) |
| 오탐률 | K (간접 ping 대상)에 따라 지수적으로 감소 |
| 메시지 부하 | 기간당 멤버당 O(1) |
| 전파 지연 | O(log n) 기간 (전염병 확산) |
| 강한 완전성 | 예 (의심 메커니즘 포함) |
| 정확성 | 확률적 (K와 의심 타임아웃으로 설정 가능) |

### 4.6 실제 사용 사례

**HashiCorp Memberlist** (Consul, Nomad, Serf에서 사용):
- 확장 기능이 포함된 SWIM의 Go 구현
- 대용량 페이로드를 위한 TCP 폴백 추가
- 설정 가능한 의심 승수
- 피기백을 통한 노드 메타데이터 전파 지원

**Lifeguard 확장** (HashiCorp의 SWIM 개선):
- 오탐률에 따라 의심 타임아웃을 동적으로 조정
- Local Health Aware Probe (LHAP): 많은 다른 노드를 의심하는 노드가 먼저 자신을 의심
- 별도 메시지 대신 프로토콜 기간을 통한 반박

---

## 5. 가십 프로토콜

### 5.1 가십 기본 원리

가십(전염병) 프로토콜은 질병과 소문의 확산에서 영감을 받았습니다. 각 노드는 주기적으로 랜덤 피어를 선택하고 정보를 교환합니다. 무작위성에도 불구하고, 가십 프로토콜은 놀랍도록 빠르게 수렴합니다.

**세 가지 가십 스타일**:

```
Push Gossip:                    Pull Gossip:
┌───┐  "I have update X"       ┌───┐  "What updates do you have?"
│ A │──────────────────▶│ B │  │ A │──────────────────────────▶│ B │
└───┘                   └───┘  │   │◀──────────────────────────│   │
                               └───┘  "Here are updates X, Y"  └───┘

Push-Pull Gossip:
┌───┐  "I have X, Y. You?"
│ A │──────────────────────▶│ B │
│   │◀──────────────────────│   │
└───┘  "I have Y, Z. Here's Z" └───┘
Both nodes now have {X, Y, Z}
```

### 5.2 수렴 분석

n개의 노드가 있는 시스템을 고려합니다. 각 라운드에서 감염된(정보를 가진) 모든 노드는 하나의 랜덤 피어에 연락합니다.

`S(t)` = 라운드 t에서 감염된 노드 수라 하면:

```
S(t+1) = S(t) + S(t) × (n - S(t)) / n
       = S(t) × (1 + (n - S(t)) / n)
```

이것은 로지스틱 성장 모델을 따릅니다. S(0) = 1에서 시작하면:

- O(log n) 라운드 후, 약 n/2 노드가 감염됨
- O(log n) 라운드 더 후, 높은 확률로 모든 노드가 감염됨

**정리**: push-pull 가십에서 `⌈log₂(n)⌉ + O(ln ln n)` 라운드 후, 확률 1 - 1/n으로 모든 노드가 감염됩니다.

```
Example: n = 1000 nodes
  Push only: ~20 rounds for full dissemination
  Push-pull: ~13 rounds for full dissemination

Example: n = 1,000,000 nodes
  Push only: ~40 rounds
  Push-pull: ~23 rounds
```

### 5.3 Anti-Entropy 프로토콜

Anti-entropy는 랜덤 피어와 주기적으로 전체 상태를 교환하여 일관된 상태로의 수렴을 보장합니다:

```python
class AntiEntropyNode:
    """Node participating in anti-entropy gossip."""

    def __init__(self, node_id: str, peers: list):
        self.node_id = node_id
        self.peers = peers
        # State: key -> (value, version)
        self.state: Dict[str, tuple] = {}

    def merge_state(self, remote_state: Dict[str, tuple]) -> None:
        """Merge remote state with local state, keeping higher versions."""
        for key, (value, version) in remote_state.items():
            if key not in self.state or self.state[key][1] < version:
                self.state[key] = (value, version)

    def anti_entropy_round(self, peer: 'AntiEntropyNode') -> None:
        """Exchange state with a peer (push-pull)."""
        # Push: send our state to peer
        peer.merge_state(self.state)
        # Pull: get peer's state
        self.merge_state(peer.state)

    def update(self, key: str, value: str) -> None:
        """Update a local key with a new version."""
        current_version = self.state.get(key, (None, 0))[1]
        self.state[key] = (value, current_version + 1)
```

### 5.4 소문 전파 (Rumor Mongering)

Anti-entropy(전체 상태를 교환)와 달리, 소문 전파는 새로운 업데이트만 전파합니다. 새로운 것을 배운 노드는 "전파자"가 되어, 이미 알고 있는 충분한 노드를 만날 때까지 소문을 가십합니다:

```python
import random

class RumorMonger:
    """Rumor mongering gossip protocol."""

    def __init__(self, node_id: str, peers: list, k: int = 3):
        self.node_id = node_id
        self.peers = peers
        self.k = k  # Stop after k consecutive "already known" responses
        self.rumors: Dict[str, dict] = {}  # rumor_id -> {data, counter}

    def receive_rumor(self, rumor_id: str, data: any) -> bool:
        """Receive a rumor. Returns True if it was new."""
        if rumor_id in self.rumors:
            return False  # Already known
        self.rumors[rumor_id] = {"data": data, "stale_count": 0}
        return True

    def gossip_round(self) -> list:
        """Perform one gossip round. Returns list of (peer, rumor_id, data)."""
        messages = []
        dead_rumors = []

        for rumor_id, info in self.rumors.items():
            if info["stale_count"] >= self.k:
                dead_rumors.append(rumor_id)
                continue

            # Select random peer
            peer = random.choice(self.peers)
            messages.append((peer, rumor_id, info["data"]))

        # Remove dead rumors (stop spreading)
        for rid in dead_rumors:
            del self.rumors[rid]

        return messages

    def process_response(self, rumor_id: str, was_new: bool) -> None:
        """Process response from a gossip target."""
        if rumor_id in self.rumors:
            if was_new:
                self.rumors[rumor_id]["stale_count"] = 0
            else:
                self.rumors[rumor_id]["stale_count"] += 1
```

### 5.5 가십 vs 브로드캐스트

| 측면 | 가십 | 트리 기반 브로드캐스트 | 플러딩 |
|------|------|----------------------|--------|
| 메시지 복잡도 | O(n log n) | O(n) | O(n²) |
| 지연 (라운드) | O(log n) | O(log n) | O(1) |
| 장애 허용 | 매우 높음 | 낮음 (트리 파손) | 매우 높음 |
| 신뢰성 | 확률적 | 결정론적 | 결정론적 |
| 대역폭 | 중간 | 낮음 | 높음 |
| 구현 | 간단 | 복잡 (트리 유지보수) | 간단 |

---

## 6. 그룹 멤버십

### 6.1 그룹 멤버십 문제

멤버십 서비스는 현재 그룹에 속한 프로세스에 대한 일관된 뷰를 유지합니다. 이것은 다음에 필수적입니다:

- **합의 프로토콜**: 투표자 집합을 알아야 함
- **복제**: 복제본 집합을 알아야 함
- **로드 밸런싱**: 사용 가능한 서버를 알아야 함

### 6.2 뷰 동기화

**뷰(View)**: 주어진 시점의 멤버 정렬 목록.

**뷰 변경(View change)**: 가입, 탈퇴, 또는 장애에 의해 트리거되는 하나의 멤버십 뷰에서 다음으로의 전환.

```
View v1 = {A, B, C}     View v2 = {A, B, D}     View v3 = {A, D, E}
     │                        │                        │
  C crashes              D joins                  B leaves, E joins
  detected               D admitted               view change
```

**뷰 동기화** 보장:
1. **합의(Agreement)**: 뷰의 모든 멤버는 멤버십에 동의함
2. **무결성(Integrity)**: 뷰 v에서 수신된 메시지는 뷰 v에서 전송된 것
3. **가상 동기화(Virtual synchrony)**: 프로세스 p가 뷰 v에서 메시지 m을 보내고 뷰 v'를 설치하면, v'를 설치하는 모든 프로세스는 m을 수신한 것

### 6.3 가상 동기화 (Isis)

코넬 대학의 Ken Birman(1987)이 개발한 Isis 시스템은 가상 동기화를 개척했습니다. 핵심 보장:

```
Process A (view v1 = {A,B,C}):    send(m1) ──── send(m2) ──── install(v2 = {A,B})
Process B (view v1 = {A,B,C}):    recv(m1) ──── recv(m2) ──── install(v2 = {A,B})
Process C (view v1 = {A,B,C}):    recv(m1) ──── CRASH

Guarantee: If C crashes during view v1, then either:
  - Both A and B received m1 and m2 before installing v2, OR
  - Neither A nor B received a given message

Virtual synchrony = "as if" the crash and the messages were synchronous
```

**구현 개요**:
1. 멤버가 장애를 감지하거나 가입 요청을 수신함
2. 모든 현재 멤버에게 새 뷰를 제안함
3. 멤버들이 뷰 변경 전에 모든 보류 중인 메시지를 플러시함
4. 모든 메시지가 플러시되면, 새 뷰가 원자적으로 설치됨

### 6.4 경량 그룹 멤버십

가상 동기화는 비용이 많이 듭니다(플러시 프로토콜, 뷰에 대한 합의 필요). 많은 현대 시스템은 대신 **최종 일관성 멤버십**을 사용합니다:

| 접근 방식 | 일관성 | 지연 | 복잡도 |
|-----------|--------|------|--------|
| 가상 동기화 | 강한 | 높음 | 높음 |
| SWIM | 최종 일관성 | 낮음 | 낮음 |
| 가십 기반 | 최종 일관성 | O(log n) | 낮음 |
| 합의 기반 (ZooKeeper) | Linearizable | 중간 | 중간 |

**어떤 것을 사용할지**:
- 가상 동기화: 메시지가 뷰와 조정되어야 하는 상태 머신 복제
- SWIM/가십: 일시적 불일치가 허용되는 서비스 디스커버리, 모니터링
- 합의 기반: 멤버십 변경이 전체 순서화되어야 할 때 (예: Raft 설정 변경)

### 6.5 확장성 비교

```
Members (n)    All-to-all     SWIM      Gossip     Consensus-backed
                heartbeat
    10            100          10        ~33            30
   100          10,000        100       ~660           300
  1,000      1,000,000      1,000     ~9,966          N/A*
 10,000    100,000,000     10,000    ~132,877          N/A*

* Consensus-backed membership doesn't scale beyond hundreds
  of members — use for metadata, not direct membership.

Messages per period (approximate)
```

---

## 7. 구현: Phi Accrual 장애 감지기

```python
import math
import time
import threading
from collections import deque
from typing import Dict, Optional

class PhiAccrualFailureDetector:
    """
    Phi Accrual Failure Detector (Hayashibara et al., 2004).

    Instead of a binary alive/dead decision, outputs a continuous
    suspicion level (phi) that the application can threshold.

    Uses normal distribution assumption for inter-arrival times.
    Cassandra-style exponential distribution variant is also provided.
    """

    def __init__(
        self,
        threshold: float = 8.0,
        max_sample_size: int = 1000,
        min_std_deviation_ms: float = 100.0,
        acceptable_heartbeat_pause_ms: float = 0.0,
        first_heartbeat_estimate_ms: float = 500.0,
    ):
        self.threshold = threshold
        self.max_sample_size = max_sample_size
        self.min_std_deviation_ms = min_std_deviation_ms
        self.acceptable_heartbeat_pause_ms = acceptable_heartbeat_pause_ms
        self.first_heartbeat_estimate_ms = first_heartbeat_estimate_ms

        # Per-node heartbeat history
        self._state: Dict[str, _NodeState] = {}
        self._lock = threading.Lock()

    def heartbeat(self, node_id: str) -> None:
        """Record reception of a heartbeat from node_id."""
        timestamp_ms = time.monotonic() * 1000.0

        with self._lock:
            if node_id not in self._state:
                # First heartbeat: initialize with estimate
                state = _NodeState(self.max_sample_size)
                # Seed with an estimated interval
                state.intervals.append(self.first_heartbeat_estimate_ms)
                state.last_heartbeat_ms = timestamp_ms
                self._state[node_id] = state
            else:
                state = self._state[node_id]
                interval = timestamp_ms - state.last_heartbeat_ms
                # Only record positive intervals
                if interval > 0:
                    state.intervals.append(interval)
                    if len(state.intervals) > self.max_sample_size:
                        state.intervals.popleft()
                state.last_heartbeat_ms = timestamp_ms

    def phi(self, node_id: str) -> float:
        """
        Compute the phi value for a given node.

        Returns:
            float: The suspicion level. Higher = more suspicious.
                   Returns float('inf') if node is unknown.
        """
        timestamp_ms = time.monotonic() * 1000.0

        with self._lock:
            if node_id not in self._state:
                return float("inf")

            state = self._state[node_id]
            if state.last_heartbeat_ms is None:
                return float("inf")

            time_diff = timestamp_ms - state.last_heartbeat_ms

            # Compute mean and std deviation of intervals
            mean = self._mean(state.intervals)
            std_dev = max(
                self._std_dev(state.intervals),
                self.min_std_deviation_ms
            )

            # Add acceptable pause to account for GC, etc.
            adjusted_mean = mean + self.acceptable_heartbeat_pause_ms

            return self._compute_phi(time_diff, adjusted_mean, std_dev)

    def is_available(self, node_id: str) -> bool:
        """Check if phi is below the configured threshold."""
        return self.phi(node_id) < self.threshold

    def _compute_phi(
        self, time_diff: float, mean: float, std_dev: float
    ) -> float:
        """
        Compute phi using the normal distribution CDF.

        phi = -log10(1 - CDF(timeDiff))

        where CDF is the cumulative distribution function of the
        normal distribution N(mean, std_dev^2).
        """
        # Standardize
        y = (time_diff - mean) / std_dev
        # Approximate the CDF of standard normal
        # using the logistic approximation
        e = math.exp(-y * (1.5976 + 0.070566 * y * y))
        if time_diff > mean:
            p_later = e / (1.0 + e)
        else:
            p_later = 1.0 - 1.0 / (1.0 + e)

        # Avoid log(0)
        if p_later < 1e-15:
            p_later = 1e-15

        return -math.log10(p_later)

    @staticmethod
    def _mean(values: deque) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _std_dev(values: deque) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)


class _NodeState:
    """Internal state maintained for each monitored node."""

    def __init__(self, max_size: int):
        self.intervals: deque = deque(maxlen=max_size)
        self.last_heartbeat_ms: Optional[float] = None


class ExponentialPhiDetector:
    """
    Cassandra-style phi detector using exponential distribution.

    Simpler math: phi = t / mean * log10(e)
    Better handles right-tailed distributions (GC pauses).
    """

    def __init__(
        self,
        threshold: float = 8.0,
        max_sample_size: int = 1000,
        first_heartbeat_estimate_ms: float = 500.0,
    ):
        self.threshold = threshold
        self.max_sample_size = max_sample_size
        self.first_heartbeat_estimate_ms = first_heartbeat_estimate_ms
        self._state: Dict[str, _NodeState] = {}
        self._lock = threading.Lock()
        self._log10_e = math.log10(math.e)

    def heartbeat(self, node_id: str) -> None:
        """Record heartbeat from node."""
        timestamp_ms = time.monotonic() * 1000.0
        with self._lock:
            if node_id not in self._state:
                state = _NodeState(self.max_sample_size)
                state.intervals.append(self.first_heartbeat_estimate_ms)
                state.last_heartbeat_ms = timestamp_ms
                self._state[node_id] = state
            else:
                state = self._state[node_id]
                interval = timestamp_ms - state.last_heartbeat_ms
                if interval > 0:
                    state.intervals.append(interval)
                state.last_heartbeat_ms = timestamp_ms

    def phi(self, node_id: str) -> float:
        """Compute phi using exponential distribution."""
        timestamp_ms = time.monotonic() * 1000.0
        with self._lock:
            if node_id not in self._state:
                return float("inf")

            state = self._state[node_id]
            time_diff = timestamp_ms - state.last_heartbeat_ms
            mean = sum(state.intervals) / len(state.intervals)

            if mean <= 0:
                return float("inf")

            # Exponential distribution: phi = (t / mean) * log10(e)
            return (time_diff / mean) * self._log10_e

    def is_available(self, node_id: str) -> bool:
        return self.phi(node_id) < self.threshold


# --- Demonstration ---

def demo_phi_detector():
    """Demonstrate the phi accrual failure detector."""
    import random

    detector = PhiAccrualFailureDetector(
        threshold=8.0,
        first_heartbeat_estimate_ms=1000.0,
    )

    node = "node-1"

    # Simulate normal heartbeats (every ~1 second with jitter)
    print("=== Simulating normal heartbeats ===")
    for i in range(20):
        detector.heartbeat(node)
        phi_val = detector.phi(node)
        print(f"  Heartbeat {i+1:2d}: phi = {phi_val:.3f}"
              f"  available = {detector.is_available(node)}")
        # Simulate passage of time (fake sleep via internal state)
        time.sleep(0.05)  # Short sleep for demo

    # Simulate missed heartbeats
    print("\n=== Simulating node failure (no heartbeats) ===")
    for i in range(10):
        time.sleep(0.1)
        phi_val = detector.phi(node)
        print(f"  After {(i+1)*100:4d}ms silence: phi = {phi_val:.3f}"
              f"  available = {detector.is_available(node)}")

    print("\n=== Recovery (heartbeats resume) ===")
    for i in range(5):
        detector.heartbeat(node)
        phi_val = detector.phi(node)
        print(f"  Heartbeat {i+1}: phi = {phi_val:.3f}"
              f"  available = {detector.is_available(node)}")
        time.sleep(0.05)


if __name__ == "__main__":
    demo_phi_detector()
```

---

## 8. 구현: SWIM 프로토콜 시뮬레이터

```python
import random
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

class MemberStatus(Enum):
    ALIVE = "alive"
    SUSPECT = "suspect"
    DEAD = "dead"


@dataclass
class MemberInfo:
    """Membership information for a single node."""
    node_id: str
    status: MemberStatus = MemberStatus.ALIVE
    incarnation: int = 0
    suspicion_start: Optional[float] = None


@dataclass
class MembershipUpdate:
    """A membership update to be disseminated via piggyback."""
    node_id: str
    status: MemberStatus
    incarnation: int
    timestamp: float = field(default_factory=time.monotonic)
    dissemination_count: int = 0


class SwimNode:
    """
    SWIM protocol node implementation.

    Implements:
      - Three-phase failure detection (ping -> ping-req -> suspect)
      - Incarnation-based refutation
      - Infection-style dissemination via piggyback
      - Configurable suspicion timeout
    """

    def __init__(
        self,
        node_id: str,
        protocol_period: float = 1.0,
        ping_timeout: float = 0.3,
        ping_req_targets: int = 3,
        suspicion_timeout: float = 5.0,
        max_piggyback_per_msg: int = 6,
        dissemination_limit_multiplier: float = 3.0,
    ):
        self.node_id = node_id
        self.protocol_period = protocol_period
        self.ping_timeout = ping_timeout
        self.ping_req_targets = ping_req_targets
        self.suspicion_timeout = suspicion_timeout
        self.max_piggyback_per_msg = max_piggyback_per_msg
        self.dissemination_limit_multiplier = dissemination_limit_multiplier

        # Own incarnation number
        self.incarnation = 0

        # Membership table: node_id -> MemberInfo
        self.members: Dict[str, MemberInfo] = {
            node_id: MemberInfo(node_id, MemberStatus.ALIVE, 0)
        }

        # Updates to disseminate
        self.update_queue: List[MembershipUpdate] = []

        # Round-robin target selection
        self._target_index = 0
        self._target_order: List[str] = []

        # Simulated network and crash state
        self._alive = True
        self._network: Optional['SwimNetwork'] = None

    @property
    def alive_members(self) -> List[str]:
        """Return list of members considered alive."""
        return [
            m.node_id for m in self.members.values()
            if m.status == MemberStatus.ALIVE and m.node_id != self.node_id
        ]

    @property
    def member_count(self) -> int:
        """Total known members including self."""
        return len(self.members)

    def join(self, existing_member: str) -> None:
        """Join the group by contacting an existing member."""
        if existing_member not in self.members:
            self.members[existing_member] = MemberInfo(
                existing_member, MemberStatus.ALIVE, 0
            )

    def _shuffle_targets(self) -> None:
        """Randomize target order for round-robin probing."""
        self._target_order = [
            nid for nid in self.members
            if nid != self.node_id
            and self.members[nid].status != MemberStatus.DEAD
        ]
        random.shuffle(self._target_order)
        self._target_index = 0

    def _next_target(self) -> Optional[str]:
        """Select the next probe target in round-robin order."""
        if not self._target_order or self._target_index >= len(self._target_order):
            self._shuffle_targets()
        if not self._target_order:
            return None
        target = self._target_order[self._target_index]
        self._target_index += 1
        return target

    def protocol_round(self, network: 'SwimNetwork') -> dict:
        """
        Execute one SWIM protocol period.

        Returns a summary of what happened this round.
        """
        if not self._alive:
            return {"node": self.node_id, "action": "crashed"}

        self._network = network
        summary = {"node": self.node_id, "actions": []}

        # 1. Check suspicion timeouts
        self._check_suspicion_timeouts(summary)

        # 2. Select probe target
        target = self._next_target()
        if target is None:
            return summary

        # 3. Phase 1: Direct ping
        ack = network.send_ping(self.node_id, target)

        if ack:
            summary["actions"].append(f"ping {target} -> ACK")
            self._process_updates(ack.get("piggyback", []))
            return summary

        summary["actions"].append(f"ping {target} -> TIMEOUT")

        # 4. Phase 2: Indirect ping via random k members
        indirect_targets = random.sample(
            [m for m in self.alive_members if m != target],
            min(self.ping_req_targets, len(self.alive_members) - 1)
        ) if len(self.alive_members) > 1 else []

        got_ack = False
        for helper in indirect_targets:
            ack = network.send_ping_req(self.node_id, helper, target)
            if ack:
                got_ack = True
                summary["actions"].append(
                    f"ping-req via {helper} for {target} -> ACK"
                )
                self._process_updates(ack.get("piggyback", []))
                break

        if got_ack:
            return summary

        summary["actions"].append(f"ping-req for {target} -> ALL TIMEOUT")

        # 5. Phase 3: Mark as suspect
        self._suspect_node(target, summary)

        return summary

    def _suspect_node(self, target: str, summary: dict) -> None:
        """Mark a node as suspected."""
        if target not in self.members:
            return

        member = self.members[target]
        if member.status == MemberStatus.DEAD:
            return

        if member.status != MemberStatus.SUSPECT:
            member.status = MemberStatus.SUSPECT
            member.suspicion_start = time.monotonic()
            summary["actions"].append(f"SUSPECT {target}")

            # Queue update for dissemination
            self.update_queue.append(MembershipUpdate(
                node_id=target,
                status=MemberStatus.SUSPECT,
                incarnation=member.incarnation,
            ))

    def _check_suspicion_timeouts(self, summary: dict) -> None:
        """Promote suspects to dead if suspicion timeout elapsed."""
        now = time.monotonic()
        for member in list(self.members.values()):
            if (member.status == MemberStatus.SUSPECT
                    and member.suspicion_start is not None
                    and now - member.suspicion_start > self.suspicion_timeout):
                member.status = MemberStatus.DEAD
                summary["actions"].append(f"CONFIRMED DEAD: {member.node_id}")
                self.update_queue.append(MembershipUpdate(
                    node_id=member.node_id,
                    status=MemberStatus.DEAD,
                    incarnation=member.incarnation,
                ))

    def handle_ping(self, sender: str) -> dict:
        """Handle incoming ping message. Return ack with piggyback."""
        if not self._alive:
            return None

        # Update sender as alive
        if sender not in self.members:
            self.members[sender] = MemberInfo(sender, MemberStatus.ALIVE, 0)
        elif self.members[sender].status == MemberStatus.SUSPECT:
            # Sender is alive — clear suspicion
            self.members[sender].status = MemberStatus.ALIVE

        return {
            "type": "ack",
            "from": self.node_id,
            "piggyback": self._get_piggyback_updates(),
        }

    def handle_ping_req(self, sender: str, target: str) -> Optional[dict]:
        """Handle ping-req: ping the target on behalf of sender."""
        if not self._alive:
            return None

        # Ping the target
        ack = self._network.send_ping(self.node_id, target)
        if ack:
            return {
                "type": "ack",
                "from": self.node_id,
                "original_target": target,
                "piggyback": self._get_piggyback_updates(),
            }
        return None

    def refute_suspicion(self) -> None:
        """
        If this node learns it has been suspected, increment incarnation
        and broadcast ALIVE.
        """
        self.incarnation += 1
        self.members[self.node_id].incarnation = self.incarnation
        self.members[self.node_id].status = MemberStatus.ALIVE

        self.update_queue.append(MembershipUpdate(
            node_id=self.node_id,
            status=MemberStatus.ALIVE,
            incarnation=self.incarnation,
        ))

    def _process_updates(self, updates: List[MembershipUpdate]) -> None:
        """Process piggybacked membership updates."""
        for update in updates:
            if update.node_id == self.node_id:
                # Someone suspects us — refute!
                if update.status == MemberStatus.SUSPECT:
                    if update.incarnation >= self.incarnation:
                        self.refute_suspicion()
                continue

            existing = self.members.get(update.node_id)
            if existing is None:
                # New member
                self.members[update.node_id] = MemberInfo(
                    update.node_id, update.status, update.incarnation
                )
                continue

            # Apply update based on priority rules
            if self._update_overrides(update, existing):
                existing.status = update.status
                existing.incarnation = update.incarnation
                if update.status == MemberStatus.SUSPECT:
                    existing.suspicion_start = time.monotonic()

    def _update_overrides(
        self, update: MembershipUpdate, existing: MemberInfo
    ) -> bool:
        """Check if an update should override existing state."""
        # Dead always wins
        if update.status == MemberStatus.DEAD:
            return True
        if existing.status == MemberStatus.DEAD:
            return False

        # Higher incarnation wins
        if update.incarnation > existing.incarnation:
            return True
        if update.incarnation < existing.incarnation:
            return False

        # Same incarnation: suspect > alive
        if (update.status == MemberStatus.SUSPECT
                and existing.status == MemberStatus.ALIVE):
            return True

        return False

    def _get_piggyback_updates(self) -> List[MembershipUpdate]:
        """Get updates to piggyback on outgoing messages."""
        import math
        max_dissemination = int(
            self.dissemination_limit_multiplier
            * math.log2(max(len(self.members), 2))
        )

        # Sort by dissemination count (least disseminated first)
        self.update_queue.sort(key=lambda u: u.dissemination_count)

        piggyback = []
        for update in self.update_queue[:self.max_piggyback_per_msg]:
            piggyback.append(update)
            update.dissemination_count += 1

        # Remove fully disseminated updates
        self.update_queue = [
            u for u in self.update_queue
            if u.dissemination_count < max_dissemination
        ]

        return piggyback

    def crash(self) -> None:
        """Simulate a node crash."""
        self._alive = False

    def recover(self) -> None:
        """Simulate node recovery."""
        self._alive = True
        self.incarnation += 1
        self.members[self.node_id].incarnation = self.incarnation
        self.members[self.node_id].status = MemberStatus.ALIVE


class SwimNetwork:
    """
    Simulated network for SWIM protocol testing.

    Supports configurable message loss and delays.
    """

    def __init__(
        self,
        nodes: Dict[str, SwimNode],
        message_loss_rate: float = 0.0,
    ):
        self.nodes = nodes
        self.message_loss_rate = message_loss_rate
        self.message_log: List[dict] = []

    def send_ping(self, sender: str, target: str) -> Optional[dict]:
        """Send a ping from sender to target."""
        self.message_log.append({
            "type": "ping", "from": sender, "to": target
        })

        # Simulate message loss
        if random.random() < self.message_loss_rate:
            return None

        target_node = self.nodes.get(target)
        if target_node is None:
            return None

        return target_node.handle_ping(sender)

    def send_ping_req(
        self, sender: str, helper: str, target: str
    ) -> Optional[dict]:
        """Send a ping-req from sender to helper for target."""
        self.message_log.append({
            "type": "ping-req",
            "from": sender,
            "via": helper,
            "target": target,
        })

        if random.random() < self.message_loss_rate:
            return None

        helper_node = self.nodes.get(helper)
        if helper_node is None:
            return None

        helper_node._network = self
        return helper_node.handle_ping_req(sender, target)

    def run_simulation(
        self, rounds: int = 20, crash_at: Optional[Dict[int, str]] = None
    ) -> None:
        """
        Run SWIM simulation for a number of rounds.

        Args:
            rounds: Number of protocol periods to simulate
            crash_at: Dict mapping round number to node_id to crash
        """
        crash_at = crash_at or {}

        print(f"=== SWIM Simulation: {len(self.nodes)} nodes, "
              f"{rounds} rounds ===\n")

        for round_num in range(1, rounds + 1):
            # Inject crashes
            if round_num in crash_at:
                crash_node = crash_at[round_num]
                if crash_node in self.nodes:
                    self.nodes[crash_node].crash()
                    print(f"[Round {round_num:2d}] *** {crash_node} CRASHES ***")

            # Each alive node runs one protocol round
            for node_id, node in self.nodes.items():
                if not node._alive:
                    continue
                result = node.protocol_round(self)
                if result.get("actions"):
                    for action in result["actions"]:
                        print(f"[Round {round_num:2d}] {node_id}: {action}")

            # Print membership summary every 5 rounds
            if round_num % 5 == 0:
                self._print_membership_summary(round_num)

        print("\n=== Final Membership Views ===")
        self._print_membership_summary(rounds)

    def _print_membership_summary(self, round_num: int) -> None:
        """Print each node's view of the membership."""
        print(f"\n--- Membership at round {round_num} ---")
        for node_id, node in self.nodes.items():
            if not node._alive:
                status = "(CRASHED)"
            else:
                alive = [
                    m.node_id for m in node.members.values()
                    if m.status == MemberStatus.ALIVE
                ]
                suspect = [
                    m.node_id for m in node.members.values()
                    if m.status == MemberStatus.SUSPECT
                ]
                dead = [
                    m.node_id for m in node.members.values()
                    if m.status == MemberStatus.DEAD
                ]
                status = (
                    f"alive={alive} suspect={suspect} dead={dead}"
                )
            print(f"  {node_id}: {status}")
        print()


def demo_swim():
    """Run a SWIM protocol demonstration."""
    # Create 5 nodes
    node_ids = [f"node-{i}" for i in range(5)]
    nodes = {}
    for nid in node_ids:
        nodes[nid] = SwimNode(
            node_id=nid,
            protocol_period=1.0,
            suspicion_timeout=3.0,
        )

    # All nodes know about each other (pre-seeded membership)
    for nid, node in nodes.items():
        for other in node_ids:
            if other != nid:
                node.join(other)

    # Create network and run simulation
    network = SwimNetwork(nodes, message_loss_rate=0.05)
    network.run_simulation(
        rounds=15,
        crash_at={5: "node-2"},  # Crash node-2 at round 5
    )


if __name__ == "__main__":
    demo_swim()
```

---

## 9. 요약 및 추가 자료

### 핵심 정리

| 개념 | 핵심 통찰 |
|------|-----------|
| 장애 감지 | 비동기 시스템에서 완전하면서 정확하기는 불가능 |
| 고정 타임아웃 | 단순하지만 네트워크 변화에 적응 불가 |
| 적응형 타임아웃 | Jacobson/Karels: 도착 간 시간의 평균과 분산을 추적 |
| Phi accrual | 연속적 의심 수준으로 감지와 애플리케이션 정책을 분리 |
| SWIM | 멤버당 O(1) 메시지 부하; 3단계 감지 + 피기백 전파 |
| 가십 | O(log n) 수렴; 장애에 강인; 구현이 간단 |
| 그룹 멤버십 | 가상 동기화(강한)에서 최종 일관성(실용적)까지의 스펙트럼 |

### 필수 논문

1. **Chandra, Toueg (1996)** — "Unreliable failure detectors for reliable distributed systems" — 장애 감지기 클래스의 형식적 프레임워크
2. **Hayashibara et al. (2004)** — "The phi accrual failure detector" — 연속적 의심 수준
3. **Das, Gupta, Stemann (2002)** — "SWIM: Scalable Weakly-consistent Infection-style Process Group Membership Protocol"
4. **van Renesse, Minsky, Hayden (1998)** — "A gossip-style failure detection service"
5. **Birman, Joseph (1987)** — "Exploiting virtual synchrony in distributed systems" — Isis 시스템
6. **Jacobson (1988)** — "Congestion avoidance and control" — TCP 타임아웃 알고리즘

### 다른 레슨과의 연결

- **레슨 3 (FLP)**: 장애 감지기는 FLP 불가능성을 우회함
- **레슨 5 (Paxos)**와 **레슨 6 (Raft)**: 리더 선출 타임아웃에 장애 감지기를 사용
- **레슨 14 (조정)**: 그룹 멤버십은 서비스 디스커버리의 기반
- **레슨 16 (캡스톤)**: KV 스토어에서 하트비트 기반 장애 감지 사용

---

[다음: 분산 조정 프리미티브](./14_Distributed_Coordination_Primitives.md)

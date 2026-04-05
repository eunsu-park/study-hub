# 레슨 1: 시스템 모델과 장애 모드

[개요](./00_Overview.md) | [다음](./02_Time_Clocks_and_Ordering.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있게 됩니다:

1. 동기, 비동기, 부분 동기 시스템 모델을 구분하고 각각을 실제 배포 환경에 매핑
2. Crash-stop, crash-recovery, Byzantine 장애 모델을 형식적으로 정의하고 프로토콜 설계에 미치는 영향을 추론
3. 분산 시스템 속성을 safety 또는 liveness 보장으로 분류
4. FLP 불가능성 결과가 합의 프로토콜 설계를 어떻게 제약하는지 설명
5. 네트워크 링크 모델(reliable, fair-loss, arbitrary)과 프로세스 장애 가정과의 관계를 분석

---

## 목차

1. [왜 시스템 모델이 중요한가](#1-왜-시스템-모델이-중요한가)
2. [타이밍 모델](#2-타이밍-모델)
3. [네트워크 링크 모델](#3-네트워크-링크-모델)
4. [프로세스 장애 모델](#4-프로세스-장애-모델)
5. [Safety와 Liveness 속성](#5-safety와-liveness-속성)
6. [프로세스와 링크 장애의 결합](#6-프로세스와-링크-장애의-결합)
7. [FLP 미리보기: 불가능성 결과가 중요한 이유](#7-flp-미리보기-불가능성-결과가-중요한-이유)
8. [실제 시스템 모델 매핑](#8-실제-시스템-모델-매핑)
9. [코드: 장애 모드 시뮬레이션](#9-코드-장애-모드-시뮬레이션)
10. [요약과 핵심 정리](#10-요약과-핵심-정리)
11. [연습 문제](#11-연습-문제)
12. [참고 문헌](#12-참고-문헌)

---

## 1. 왜 시스템 모델이 중요한가

### 근본적 도전

분산 시스템은 네트워크를 통해 메시지를 전달하며 통신하는 독립적인 컴퓨팅 노드들의 집합입니다. 메모리와 단일 클럭을 공유하는 단일 머신과 달리, 분산 시스템은 세 가지 근본적인 불확실성에 대응해야 합니다:

- **무한 지연**: 메시지가 도착하는 데 임의로 긴 시간이 걸릴 수 있습니다.
- **독립적 장애**: 어떤 노드든 다른 노드와 독립적으로 장애가 발생할 수 있습니다.
- **글로벌 클럭 부재**: "지금"에 대한 공유된 개념이 없습니다.

*무엇이* 잘못될 수 있는지에 대한 정밀한 모델 없이는, 알고리즘이 올바른지 추론하는 것이 불가능합니다. 시스템 모델은 가정을 명시하고, 정확성을 증명하고, 달성 가능한 것의 한계를 이해하기 위한 형식적 프레임워크를 제공합니다.

### 추상화의 역할

실제 네트워크와 머신은 복잡합니다. 패킷이 손상되고, 커널이 패닉을 일으키고, SSD가 조용히 비트를 뒤집고, 클라우드 VM이 경고 없이 선점됩니다. 시스템 모델은 이 복잡성을 깔끔한 범주로 추상화하여:

1. **알고리즘 설계자**가 프로토콜이 어떤 장애를 허용해야 하는지 정확히 알 수 있게 합니다.
2. **구현자**가 어떤 실제 시나리오가 각 장애 클래스에 매핑되는지 알 수 있게 합니다.
3. **운영자**가 배포 환경에 적합한 알고리즘을 선택할 수 있게 합니다.

```
Real World                          Model
─────────────────────────────────────────────────────
Network cable cut                → Link failure
Kernel panic                     → Crash-stop failure
Disk corruption after restart    → Crash-recovery failure
Compromised server               → Byzantine failure
Data center network delay spike  → Partial synchrony
NTP-synchronized cluster         → Synchronous (approx.)
Internet-wide peer-to-peer       → Asynchronous
```

### 모델 계층 구조

시스템 모델은 가장 약한(프로토콜 구축이 가장 어려운) 것부터 가장 강한(가장 쉽지만 비현실적인) 것까지 계층 구조를 형성합니다:

```
Strongest assumptions (easiest to design for)
    ▲
    │  Synchronous + Crash-stop + Reliable links
    │  Partially synchronous + Crash-recovery + Fair-loss links
    │  Asynchronous + Byzantine + Arbitrary links
    ▼
Weakest assumptions (hardest to design for, most general)
```

약한 가정 하에서 올바르다고 증명된 알고리즘은 자동으로 강한 가정 하에서도 동작합니다. 그 역은 **참이 아닙니다** -- 동기성을 가정한 알고리즘은 비동기 환경에서 치명적으로 실패할 수 있습니다.

---

## 2. 타이밍 모델

타이밍 모델은 메시지 전달 지연과 상대적 처리 속도에 대해 어떤 가정을 하는지를 명시합니다.

### 2.1 동기 모델

**정의**: 동기 시스템에서는 다음에 대해 알려진 상한이 존재합니다:

- **메시지 지연**: 전송된 모든 메시지는 알려진 상한 `Δ` 시간 단위 내에 전달됩니다.
- **처리 시간**: 모든 프로세스는 알려진 상한 `Φ` 시간 단위 내에 계산 단계를 완료합니다.
- **클럭 드리프트**: 모든 프로세스는 실제 시간 대비 제한된 드리프트율 `ρ`을 가진 로컬 클럭에 접근합니다.

형식적으로, 실제 시간 `t`에 전송된 메시지 `m`에 대해:

```
delivery_time(m) ≤ t + Δ
```

그리고 실제 시간 `t`에 시작되는 계산 단계에 대해:

```
completion_time(step) ≤ t + Φ
```

**속성**:

| 속성 | 보장 |
|------|------|
| 메시지 전달 | Δ로 제한 |
| 처리 속도 | Φ로 제한 |
| 장애 감지 | Perfect (Δ + Φ만 대기) |
| 합의 | 결정론적으로 해결 가능 |
| 타임아웃 기반 감지 | 항상 정확 |

**장애 감지 원리**: 프로세스 `p`가 프로세스 `q`에 메시지를 보내고 `2Δ + Φ` 시간 내에 응답을 받지 못하면, `p`는 `q`가 crash되었다고 **확정적으로** 판단할 수 있습니다. 오탐(false positive)이 없습니다.

**실제 근사**: 전용 네트워크 스위치, 제한된 큐 깊이, 실시간 운영 체제를 갖춘 엄격히 통제된 클러스터가 동기 모델에 근접합니다. 예시:

- 하드 실시간 임베디드 시스템 (예: 항공전자, 자동차 CAN 버스)
- 지연 시간이 제한된 전용 InfiniBand 클러스터
- 결정론적 타이밍을 가진 FPGA 기반 트레이딩 시스템

**한계**: 순수 동기성은 일반 분산 시스템에서 거의 달성할 수 없습니다. 단일 가비지 컬렉션 일시 정지, 네트워크 혼잡 이벤트, 또는 페이지 폴트가 타이밍 상한을 위반할 수 있습니다.

### 2.2 비동기 모델

**정의**: 비동기 시스템에서는 **어떤** 타이밍 가정도 하지 않습니다:

- 메시지가 전달되는 데 임의로 긴 시간이 걸릴 수 있습니다 (단, 링크가 fair-loss 또는 reliable이면 결국 전달됩니다).
- 프로세스가 한 단계를 실행하는 데 임의로 긴 시간이 걸릴 수 있습니다.
- 서로 다른 프로세스의 로컬 클럭 사이에 관계가 없습니다.

형식적으로:

```
∀ bound B, ∃ execution where delivery_time(m) > B
```

**속성**:

| 속성 | 보장 |
|------|------|
| 메시지 전달 | 상한 없음 (링크가 fair이면 최종 전달) |
| 처리 속도 | 상한 없음 |
| 장애 감지 | 느린 것과 crash된 것을 구분 **불가능** |
| 합의 | 결정론적으로 **불가능** (FLP) |
| 타임아웃 기반 감지 | 항상 신뢰 불가 |

**핵심 문제**: 비동기 시스템에서는 crash된 프로세스와 매우 느린 프로세스를 **구분할 수 없습니다**. 설정한 어떤 타임아웃도 오탐(살아있는 프로세스를 죽었다고 선언)이나 미탐(죽은 프로세스를 너무 오래 기다림)을 유발할 수 있습니다.

**실제 매핑**:

- 메시지가 여러 ISP를 통과하는 인터넷 규모 시스템
- 무한 가비지 컬렉션 일시 정지가 있는 시스템 (예: 실시간 GC 없는 JVM)
- 이질적인 노드 역량을 가진 피어투피어 네트워크

**연구 이유**: 비동기 모델은 이론적 결과의 **금본위**입니다. 비동기 모델에서 동작하는 알고리즘은 어디서나 동작합니다. FLP 불가능성 정리(레슨 03)는 이 모델에서 결정론적 합의가 불가능함을 보여주며, 실용적 우회 방법의 탐색을 촉발합니다.

### 2.3 부분 동기 모델

**정의** (Dwork, Lynch, Stockmeyer 1988): 부분 동기 시스템은 타이밍 상한을 만족하지만, 조건이 있습니다. 두 가지 동등한 형식화가 있습니다:

**형식화 1 (알 수 없는 상한)**: 상한 `Δ`가 존재하지만, 그 값은 프로세스에게 알려져 있지 않습니다. 어떤 `Δ`에 대해서도 동작하는 프로토콜을 설계해야 하지만, `Δ`가 무엇인지는 모릅니다.

**형식화 2 (Global Stabilization Time, GST)**: 시간 `GST`(프로세스에게 알려지지 않음)가 존재하여, 그 이후에는 시스템이 상한 `Δ`로 동기적으로 동작합니다. GST 이전에는 시스템이 완전히 비동기입니다.

```
Formally:  ∃ GST, Δ such that:
  ∀ messages m sent at time t ≥ GST:
    delivery_time(m) ≤ t + Δ
```

**직관**: 네트워크가 한동안 임의로 나쁠 수 있지만(파티션, 혼잡 폭주, 라우팅 루프), *결국* 안정화되어 메시지가 제한된 시간 내에 도착하기 시작합니다.

**속성**:

| 속성 | 보장 |
|------|------|
| 메시지 전달 | GST 이후 제한됨 |
| 처리 속도 | GST 이후 제한됨 |
| 장애 감지 | 최종적으로 정확 |
| 합의 | **해결 가능** (Paxos, Raft, PBFT) |
| Safety | 항상 유지 (GST 이전에도) |
| Liveness | GST 이후에만 보장 |

**중요한 이유**: 부분 동기성은 실용적 분산 시스템의 **최적 지점**입니다. 네트워크가 보통은 잘 동작하지만 가끔 장애가 발생하는 현실을 포착합니다. 대부분의 실제 합의 프로토콜(Paxos, Raft, PBFT)은 부분 동기성을 위해 설계됩니다:

- **Safety**는 타이밍과 관계없이 보장됩니다 (네트워크 파티션 중에도).
- **Liveness** (진행)는 시스템이 안정화된 후에만 보장됩니다.

### 2.4 비교 표

| 차원 | 동기 | 부분 동기 | 비동기 |
|------|------|-----------|--------|
| 지연 상한 | 알려진 Δ | ∃Δ, 알 수 없거나 GST 이후 | 없음 |
| 처리 상한 | 알려진 Φ | ∃Φ, 알 수 없거나 GST 이후 | 없음 |
| 장애 감지 | Perfect | Eventually perfect | 불가능 |
| 합의 | 간단히 해결 가능 | 해결 가능 (Paxos, Raft) | 불가능 (FLP) |
| 실제 예시 | 하드 실시간 | 클라우드 데이터 센터 | 인터넷 |
| 알고리즘 복잡도 | 단순 | 보통 | N/A (무작위화 필요) |

### 2.5 실제 타이밍 스펙트럼

```
    Hard Real-Time    LAN Cluster    Cloud Region     Internet      Tor Network
         │                │              │               │              │
    ◄────┼────────────────┼──────────────┼───────────────┼──────────────┼────►
    Synchronous    ≈Synchronous    Partial Sync      Async       Adversarial
                                                                   Async
```

대부분의 프로덕션 시스템은 "부분 동기성" 영역에서 운영됩니다. 핵심 설계 원칙:

> **비동기성 하에서 safety를 설계하고, liveness에는 부분 동기성만 의존하라.**

---

## 3. 네트워크 링크 모델

프로세스 간 통신은 네트워크 링크를 통해 이루어집니다. 링크 모델은 메시지 전송에서 무엇이 잘못될 수 있는지를 명시합니다.

### 3.1 Reliable 링크

**정의**: 올바른 프로세스 `p`가 올바른 프로세스 `q`에 메시지 `m`을 보내면, `q`는 결국 `m`을 전달합니다. 추가로:

1. **중복 없음**: `m`은 최대 한 번 전달됩니다.
2. **생성 없음**: `q`가 `m`을 전달하면, `p`가 이전에 `m`을 보냈습니다.
3. **신뢰 전달**: `p`가 올바르고 올바른 `q`에 `m`을 보내면, `q`가 결국 `m`을 전달합니다.

```
Process p ──── m ────► Process q     (always delivered, exactly once)
```

**구축 방법**: Reliable 링크는 시퀀스 번호와 재전송을 사용하여 fair-loss 링크 위에 구축할 수 있습니다:

```python
class ReliableLink:
    """Build reliable delivery on top of a fair-loss link."""

    def __init__(self, fair_loss_link):
        self.link = fair_loss_link
        self.seq_num = 0
        self.delivered = set()  # track delivered (sender, seq) pairs
        self.pending = {}       # messages awaiting acknowledgment

    def send(self, dest, message):
        self.seq_num += 1
        tagged = (self.seq_num, message)
        self.pending[self.seq_num] = (dest, tagged)
        self._retransmit_loop(dest, tagged)

    def _retransmit_loop(self, dest, tagged):
        """Retransmit until acknowledged (stubborn delivery)."""
        while tagged[0] in self.pending:
            self.link.send(dest, tagged)
            # In practice, use exponential backoff
            time.sleep(self.timeout)

    def on_receive(self, sender, tagged):
        seq, message = tagged
        msg_id = (sender, seq)
        self.link.send(sender, ("ACK", seq))
        if msg_id not in self.delivered:
            self.delivered.add(msg_id)
            self.deliver(sender, message)  # deliver to application
```

### 3.2 Fair-Loss 링크

**정의**: Fair-loss 링크는 세 가지를 보장합니다:

1. **공정한 손실**: 올바른 프로세스 `p`가 올바른 프로세스 `q`에 메시지 `m`을 무한히 자주 보내면, `q`가 `m`을 무한히 자주 전달합니다.
2. **유한 중복**: `p`가 `m`을 유한 횟수 보내면, `q`가 `m`을 유한 횟수 전달합니다.
3. **생성 없음**: Reliable 링크와 동일합니다.

**직관**: 개별 메시지는 손실될 수 있지만, 계속 재전송하면 결국 통과합니다. 비적대적 네트워크에서의 UDP를 모델링합니다.

```
Process p ──── m ────► Process q     (may be lost)
Process p ──── m ────► Process q     (may be lost)
Process p ──── m ────► Process q     (delivered!)
```

### 3.3 Arbitrary (적대적) 링크

**정의**: 어떤 보장도 없습니다. 메시지는:

- 손실될 수 있음
- 중복될 수 있음
- 순서가 바뀔 수 있음
- 수정(손상)될 수 있음
- 무에서 생성(위조)될 수 있음

이것은 공격자가 통신 채널을 완전히 통제하는 적대적 네트워크를 모델링합니다.

**완화**: 암호학적 기법을 사용합니다:

- **무결성**: MAC(Message Authentication Code) 또는 디지털 서명이 수정과 위조를 방지합니다.
- **기밀성**: 암호화가 도청을 방지합니다.
- **재전송 방지**: 넌스와 시퀀스 번호가 재전송 공격을 방지합니다.

암호학적 보호를 적용하면, arbitrary 링크를 fair-loss 링크로 축소할 수 있습니다 (공격자가 암호학적 기본 요소를 깨뜨릴 수 없다고 가정).

### 3.4 링크 모델 비교

| 속성 | Reliable | Fair-Loss | Arbitrary |
|------|----------|-----------|-----------|
| 메시지 손실 | 없음 | 있음 (유한) | 있음 |
| 중복 | 없음 | 있음 (유한) | 있음 (무한) |
| 손상 | 없음 | 없음 | 있음 |
| 위조 | 없음 | 없음 | 있음 |
| 순서 변경 | 가능 | 가능 | 있음 |
| 구축 기반 | Fair-loss + 재전송 | 물리 계층 | 물리 계층 |
| 실제 예시 | TCP (근사) | UDP | 공개 인터넷 |

### 3.5 네트워크 파티션

**네트워크 파티션**은 네트워크가 두 개 이상의 노드 그룹으로 분리되어, 각 그룹 내에서는 통신할 수 있지만 그룹 간에는 통신할 수 없는 상황입니다.

```
┌─────────────────┐         ┌─────────────────┐
│  Partition A     │   ✕✕✕   │  Partition B     │
│  Node 1          │ ◄─────► │  Node 3          │
│  Node 2          │  NO     │  Node 4          │
│                  │  COMM   │  Node 5          │
└─────────────────┘         └─────────────────┘
```

파티션은 프로세스 장애가 아닌 링크 장애의 한 형태입니다. 모든 노드가 여전히 살아있고 처리하고 있지만, 파티션 경계를 넘어 통신할 수 없습니다. 이것이 바로 CAP 정리가 다루는 시나리오입니다 (레슨 04).

**부분 파티션**도 가능합니다: 노드 A가 노드 B에 도달할 수 있고, 노드 B가 노드 C에 도달할 수 있지만, 노드 A는 노드 C에 직접 도달할 수 없습니다.

---

## 4. 프로세스 장애 모델

**프로세스 장애 모델**은 개별 노드가 올바른 동작에서 어떻게 벗어날 수 있는지를 명시합니다.

### 4.1 Crash-Stop 장애

**정의**: 프로세스가 특정 시점까지 알고리즘을 올바르게 실행하다가, 그 시점에서 **영구적으로 실행을 중단**합니다. 한번 crash되면, 절대 복구되지 않습니다.

형식적으로, 단순화된 프로세스 대수 표기법을 사용하면:

```
Process behavior: p ::= action.p | STOP

Correct process: always eventually takes the next action
Crashed process: transitions to STOP and remains there permanently

Timeline:
  ─────────────────────┬───────────────────
    correct execution  │  STOP (permanent)
                     crash
```

**속성**:

- Crash 전에는 프로세스가 알고리즘을 완벽하게 따릅니다.
- Crash 후에는 프로세스가 메시지를 보내지 않고 어떤 단계도 수행하지 않습니다.
- 비동기 모델에서 다른 프로세스는 "crash됨"과 "매우 느림"을 구분할 수 없습니다.
- 프로세스가 복구되지 않습니다 (이것이 crash-recovery와의 핵심 차이점입니다).

**장애 허용 요건**: `n`개 프로세스로 구성된 시스템은 대부분의 합의 프로토콜에서 `n ≥ 2f + 1`이면 최대 `f`개의 crash-stop 장애를 허용할 수 있습니다.

**실제 매핑**:

- OS에 의해 종료된 프로세스 (OOM killer, segfault)
- 중복 스토리지 없는 하드웨어 장애 (고장난 SSD, 탄 마더보드)
- 재시작 없이 클라우드 제공자에 의해 종료된 VM

### 4.2 Crash-Recovery 장애

**정의**: 프로세스가 crash된 후 나중에 **복구**되어 실행을 재개할 수 있습니다. 복구 시, 프로세스는 모든 인메모리 상태를 잃지만, crash를 견디는 영구(안정) 스토리지에 접근할 수 있습니다.

```
Process behavior:
  p ::= action.p | CRASH.RECOVER.p'

Timeline:
  ────────┬──────────┬────────────┬──────────┬──────────
  correct │  crashed │  recovered │  crashed │ recovered
          │          │  (state    │          │ (state
          │          │   from     │          │  from
          │          │   disk)    │          │  disk)
```

**안정 스토리지 추상화**: 프로세스는 crash를 견디는 안정 스토리지에 쓸 수 있습니다. 복구 시, 안정 상태를 읽고 재개합니다.

```python
class CrashRecoveryProcess:
    """A process that can crash and recover from stable storage."""

    def __init__(self, node_id, stable_storage_path):
        self.node_id = node_id
        self.stable_path = stable_storage_path
        self.state = self._recover_state()

    def _recover_state(self):
        """Read state from stable storage on startup/recovery."""
        try:
            with open(self.stable_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"term": 0, "voted_for": None, "log": []}

    def _persist_state(self):
        """Write state to stable storage before responding."""
        tmp_path = self.stable_path + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(self.state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, self.stable_path)  # atomic on POSIX

    def handle_vote_request(self, candidate_id, candidate_term):
        """Must persist voted_for before responding (Raft safety)."""
        if candidate_term > self.state["term"]:
            self.state["term"] = candidate_term
            self.state["voted_for"] = candidate_id
            self._persist_state()  # MUST persist before ACK
            return True
        return False
```

**핵심 통찰**: Crash-recovery 모델에서는 **영속화 규율**이 정확성을 결정합니다. 투표를 영속화하기 전에 투표 요청에 응답한 프로세스는 복구 시 safety를 위반할 수 있습니다.

**장애 허용**: Crash-recovery와 안정 스토리지가 있으면, crash 후 복구된 프로세스는 "죽은" 것이 아닌 "느린" 것으로 취급됩니다. 이는 프로토콜이 더 많은 장애를 허용할 수 있음을 의미하지만, 대가가 있습니다:
- 더 높은 지연 (모든 상태 변경은 응답 전에 디스크에 기록되어야 함)
- 더 복잡한 복구 로직

### 4.3 Byzantine 장애

**정의**: Byzantine 장애가 있는 프로세스는 **임의의 동작**을 보일 수 있습니다:

- 서로 다른 프로세스에 상충하는 메시지를 전송
- 자신의 상태에 대해 거짓말
- 다른 Byzantine 프로세스와 공모
- 침묵 유지 (crash 장애를 포함)
- 프로토콜을 완벽하게 따르기 (감지하기 가장 어려운 경우)

**Byzantine Generals Problem** (Lamport, Shostak, Pease 1982)에서 명명되었습니다.

```
Correct process: follows the protocol exactly
Byzantine process: ANY behavior is possible

Examples of Byzantine behavior:
  - Send "YES" to node A and "NO" to node B for the same proposal
  - Claim to have received a message that was never sent
  - Delay responses strategically to cause maximum disruption
  - Forge messages from other processes (without cryptography)
```

프로세스 대수와 유사한 표기법을 사용한 **형식 정의**:

```
Correct process:  p_correct ::= receive(m). f(m). send(f(m)). p_correct
Byzantine process: p_byz ::= receive(m). send(ANYTHING). p_byz
                          |  send(ANYTHING). p_byz        (no input needed)
                          |  STOP. p_byz                  (crash)
```

**장애 허용 한계**:

| 속성 | 필요 노드 수 | 공식 |
|------|-------------|------|
| Byzantine Agreement (서명 없는 메시지) | n ≥ 3f + 1 | f ≥ n/3 허용 불가 |
| Byzantine Agreement (서명된 메시지) | n ≥ 2f + 1 | 디지털 서명이 도움 |
| Byzantine Broadcast | n ≥ 3f + 1 | Agreement와 동일 |

`3f + 1` 한계는 **타이트**합니다 -- 구두(서명 없는) 메시지로의 Byzantine agreement에 대해 필요 조건이자 충분 조건입니다.

**3f + 1에 대한 직관**: `f = 1` Byzantine 노드를 고려합니다. `n ≥ 4`가 필요합니다. 3개 노드와 1개 Byzantine로는 두 올바른 노드가 나머지 둘 중 누가 거짓말을 하는지 구분할 수 없습니다:

```
n = 3, f = 1 (IMPOSSIBLE):

Node A (correct): "I say value X"
Node B (Byzantine): tells A "I say X", tells C "I say Y"
Node C (correct): "I say Y"

A sees: A=X, B=X, C=Y → majority X
C sees: A=X, B=Y, C=Y → majority Y
→ Correct nodes disagree! No consensus.
```

```
n = 4, f = 1 (POSSIBLE):

Node A (correct): "I say X"
Node B (Byzantine): arbitrary
Node C (correct): "I say X"
Node D (correct): "I say X"

Even if B lies, the 3 correct nodes can outvote it.
```

**Byzantine 동작의 실제 원인**:

- 잘못된 상태 전이를 유발하는 소프트웨어 버그
- 하드웨어 오류 (비트 플립, 펌웨어 버그)
- 해킹된 노드 (침해된 서버)
- 악의적 참가자 (블록체인 네트워크)
- 스플릿 브레인 동작을 유발하는 설정 오류

### 4.4 장애 모델 계층 구조

```
                 Byzantine
                 (arbitrary behavior)
                    ▲
                    │ strictly stronger
                    │
              Crash-Recovery
              (crash + recover with disk state)
                    ▲
                    │ strictly stronger
                    │
               Crash-Stop
               (crash permanently)
                    ▲
                    │ strictly stronger
                    │
              No Failures
              (all processes correct)
```

Byzantine 장애를 허용하는 프로토콜은 자동으로 crash 장애를 허용합니다 (crash는 가능한 Byzantine 동작 중 하나이므로). Crash-stop을 위해 설계된 프로토콜은 Byzantine 장애 시 치명적으로 실패할 수 있습니다.

### 4.5 장애 모델 비교 표

| 차원 | Crash-Stop | Crash-Recovery | Byzantine |
|------|-----------|----------------|-----------|
| 장애 전 동작 | 올바름 | 올바름 | 올바름 (또는 아님) |
| 장애 행동 | 영구 중단 | 중단 후 복구 | 아무거나 |
| 장애 후 상태 | 소실 | 안정 스토리지 보존 | 손상됨 |
| 거짓말 가능? | 아니오 | 아니오 | 예 |
| 감지 난이도 | 보통 | 보통 | 매우 어려움 |
| f개 장애에 필요한 최소 노드 | 2f + 1 | 2f + 1 (스토리지 포함) | 3f + 1 |
| 프로토콜 복잡도 | 낮음 | 보통 | 높음 |
| 메시지 복잡도 | O(n) ~ O(n²) | O(n²) | O(n³) 일반적 |
| 실제 예시 | OOM kill | 서버 재부팅 | 해킹된 노드 |
| 핵심 프로토콜 | Paxos, Raft | Paxos, Raft | PBFT, HotStuff |

---

## 5. Safety와 Liveness 속성

분산 시스템의 모든 정확성 속성은 **safety** 속성 또는 **liveness** 속성으로 분류할 수 있습니다. Alpern과 Schneider(1985)에 의한 이 분류는 분산 시스템 이론에서 가장 근본적인 구분 중 하나입니다.

### 5.1 Safety 속성

**정의**: Safety 속성은 "나쁜 일이 일어나지 않음"을 명시합니다. 형식적으로, 속성 `P`가 safety 속성이 되려면:

- `P`의 모든 위반은 **유한 접두사**를 가집니다 -- 속성이 위반된 실행의 특정 지점이 있습니다.
- 한번 위반되면, 취소될 수 없습니다 (나쁜 일은 이미 일어났습니다).

```
Safety property violation:
  ──────────────────────X────────────────
  correct execution    │ violation (irrecoverable)
                       │
                  finite prefix that
                  demonstrates violation
```

**Safety 속성의 예시**:

| Safety 속성 | 방지하는 "나쁜 일" |
|------------|-------------------|
| Agreement | 두 올바른 프로세스가 다른 값을 결정 |
| Validity | 결정된 값이 제안된 적이 없음 |
| Integrity | 메시지가 두 번 이상 전달됨 |
| Mutual exclusion | 두 프로세스가 동시에 락을 보유 |
| Consistency (linearizability) | 연산이 실시간 순서를 벗어나 나타남 |
| No data loss | 확인된 쓰기가 손실됨 |

**핵심 통찰**: Safety 속성은 유한 시간 내에 위반될 수 있지만, 유한 시간 내에 "달성"될 수는 없습니다 -- safety 속성이 유지됨을 확인하려면 전체 실행을 관찰해야 합니다.

### 5.2 Liveness 속성

**정의**: Liveness 속성은 "좋은 일이 결국 일어남"을 명시합니다. 형식적으로:

- 모든 유한 실행 접두사는 속성을 만족하도록 확장할 수 있습니다.
- 어떤 유한 접두사도 liveness 속성을 위반할 수 없습니다 (항상 희망이 있습니다).

```
Liveness property:
  ──────────────────────────────────?─────
  no progress so far...            │
                              but still hope!
                              (could be satisfied
                               in the future)
```

**Liveness 속성의 예시**:

| Liveness 속성 | 결국 일어나는 "좋은 일" |
|-------------|----------------------|
| Termination | 모든 올바른 프로세스가 결국 결정 |
| Eventual delivery | 모든 전송된 메시지가 결국 전달 |
| Progress | 프로세스가 락을 요청하면 결국 획득 |
| Availability | 모든 요청이 결국 응답을 수신 |
| Eventual consistency | 모든 복제본이 결국 수렴 |

### 5.3 Safety-Liveness 분해 정리

**정리** (Alpern & Schneider, 1985): 분산 시스템의 모든 정확성 속성은 safety 속성과 liveness 속성의 교집합으로 표현할 수 있습니다.

```
Any property P = Safety(P) ∩ Liveness(P)
```

**예시 -- 합의**:

```
Consensus = Safety ∩ Liveness

Safety component:
  - Agreement: No two correct processes decide differently
  - Validity: The decided value was proposed by some process

Liveness component:
  - Termination: Every correct process eventually decides
```

### 5.4 근본적 트레이드오프

장애가 있는 비동기 시스템에서는 safety와 liveness를 **동시에** 보장할 수 없습니다. 이것이 FLP 불가능성 결과의 본질입니다 (레슨 03).

프로토콜 설계에 대한 실용적 결론:

> **항상 safety를 보존하라. 필요할 때 liveness를 희생하라.**

| 접근법 | Safety | Liveness | 예시 |
|--------|--------|----------|------|
| Paxos/Raft | 항상 보장 | GST 이후 보장 | 대부분의 데이터베이스 |
| 낙관적 복제 | 최종적 (CRDT) | 항상 보장 | 협업 편집 |
| 블록체인 (PoW) | 확률적 | 확률적 | Bitcoin |

이것은 Paxos 기반 시스템이 네트워크 파티션 동안 진행을 멈출 수 있지만(liveness 희생), 절대로 잘못된 결과를 반환하지 **않는다**는(safety 보존) 것을 의미합니다. 이것은 데이터베이스에 정확히 올바른 트레이드오프입니다.

### 5.5 형식 분류 연습

다음 각각을 safety(S), liveness(L), 또는 둘 다 아닌 것(N)으로 분류하세요:

```
1. "At most one leader exists at any time"                    → S
2. "A new leader is eventually elected"                       → L
3. "If a value is committed, it is never lost"                → S
4. "Every client request eventually gets a response"          → L
5. "The system never enters an inconsistent state"            → S
6. "The system processes at least 1000 requests per second"   → Neither
   (This is a performance requirement, not a correctness property)
7. "Every message sent is eventually delivered"               → L
8. "A delivered message was actually sent"                    → S
```

---

## 6. 프로세스와 링크 장애의 결합

실제 시스템은 프로세스와 링크 장애의 조합을 경험합니다. 이 장애 모드 간의 상호 작용이 달성 가능한 것을 결정합니다.

### 6.1 장애 조합 매트릭스

| 프로세스 모델 | 링크 모델 | 달성 가능 | 예시 시스템 |
|-------------|----------|----------|-----------|
| Crash-stop | Reliable | 합의 (2f+1) | LAN 클러스터 |
| Crash-stop | Fair-loss | 합의 (2f+1, 재전송 포함) | WAN 복제본 |
| Crash-recovery | Reliable | 합의 (2f+1, 안정 스토리지 포함) | 데이터베이스 복제본 |
| Crash-recovery | Fair-loss | 합의 (더 어려움, 영구 상태 + 재전송 필요) | 클라우드 데이터베이스 |
| Byzantine | Reliable | 합의 (3f+1) | 허가형 블록체인 |
| Byzantine | Arbitrary | 합의 (3f+1, 암호 포함) | 공개 블록체인 |

### 6.2 동등한 구성

일부 조합은 달성 가능성 측면에서 동등합니다:

**주장**: Fair-loss 링크를 가진 crash-stop 프로세스는 reliable 링크를 가진 crash-stop 프로세스와 동등합니다 (해결 가능한 문제 측면에서).

**증명 스케치**: 재전송을 사용하여 fair-loss 링크 위에 reliable 링크를 구축할 수 있습니다 (섹션 3.1에서 보여준 대로). 이 구성은 전송 프로세스가 살아있기만 하면 되므로(crash되지 않음), 올바른 프로세스에 대해 동작합니다. Crash된 프로세스는 어차피 보내거나 받을 필요가 없습니다.

**주장**: Reliable 링크를 가진 Byzantine 프로세스는 arbitrary 링크를 가진 crash-stop 프로세스와 동등하지 **않습니다**.

**이유**: Byzantine 프로세스는 적극적으로 오해를 유발하는 메시지를 보낼 수 있지만, arbitrary 링크는 메시지를 손상시키거나 유실할 수만 있습니다. Byzantine 프로세스는 링크 수준 손상으로는 모델링할 수 없는 전략적 기만에 관여할 수 있습니다.

### 6.3 네트워크 파티션 vs 노드 Crash 모호성

Crash-stop 장애와 fair-loss 링크가 있는 비동기 시스템에서:

```
Scenario 1: Node B crashed
  A ────────► B (dead)     No response

Scenario 2: Network partition
  A ────✕───► B (alive)    No response (messages lost)

Scenario 3: Node B is slow
  A ────────► B (alive)    Response coming... eventually
```

A의 관점에서 세 시나리오 모두 **구분 불가능**합니다. 이것이 바로:

- 비동기 시스템에서 장애 감지가 불가능한 이유입니다.
- 타임아웃이 오탐(살아있는 노드를 죽었다고 선언)을 만드는 이유입니다.
- CAP 정리가 파티션 동안 일관성과 가용성 사이의 선택을 강제하는 이유입니다.

---

## 7. FLP 미리보기: 불가능성 결과가 중요한 이유

### 7.1 합의 문제

**합의 문제**는 프로세스 집합이 단일 값에 합의하도록 요구합니다:

1. **Agreement**: 모든 올바른 프로세스가 같은 값을 결정합니다.
2. **Validity**: 결정된 값은 어떤 프로세스가 제안한 것입니다.
3. **Termination**: 모든 올바른 프로세스가 결국 결정합니다.

### 7.2 FLP 불가능성 결과 (미리보기)

**정리** (Fischer, Lynch, Paterson 1985): Reliable 링크를 가진 비동기 시스템에서, 단 **하나의** 프로세스만 crash될 수 있어도 합의를 해결하는 결정론적 프로토콜은 **없습니다**.

```
Asynchronous + Deterministic + Even 1 crash → Consensus IMPOSSIBLE

Formally:
  ¬∃ protocol P: (asynchronous ∧ deterministic ∧ f ≥ 1) → consensus
```

이것이 말하는 것이 **아닌** 것:
- 실제로 합의가 불가능하다 (그렇지 않습니다)
- 분산 시스템을 포기해야 한다
- 유용한 작업이 불가능하다

이것이 **말하는** 것:
- 모든 올바른 합의 프로토콜은 다음 중 적어도 하나를 사용해야 합니다:
  - **무작위화** (예: Ben-Or 프로토콜)
  - **타이밍 가정** (예: Paxos/Raft의 부분 동기성)
  - **장애 감지기** (예: Chandra-Toueg의 오라클 모델)
- 순수 비동기 시스템에서 결정론적으로 합의를 해결한다고 주장하는 프로토콜은 **틀렸습니다**.

### 7.3 실무자에게 중요한 이유

| 프로토콜 | FLP를 우회하는 방법 |
|---------|-------------------|
| Paxos | Liveness를 위해 부분 동기성 가정 |
| Raft | 타임아웃 사용 (부분 동기성) |
| PBFT | View 변경을 위해 부분 동기성 가정 |
| Bitcoin PoW | 무작위화 (마이닝) + 확률적 safety |
| Ben-Or | 무작위 코인 플립 |
| Chandra-Toueg | 장애 감지기 오라클 (◇S) 가정 |

FLP를 레슨 03에서 엄밀히 증명하고, 각 우회 전략을 탐구할 것입니다.

---

## 8. 실제 시스템 모델 매핑

### 8.1 클라우드 제공자 모델

| 시스템 | 타이밍 모델 | 장애 모델 | 링크 모델 | 참고 |
|--------|-----------|----------|----------|------|
| AWS DynamoDB | 부분 동기 | Crash-recovery | Reliable (AZ 내) | Sloppy quorum, hinted handoff |
| Google Spanner | 부분 동기 + TrueTime | Crash-recovery | Reliable (리전 내) | TrueTime이 제한된 불확실성 제공 |
| Azure Cosmos DB | 부분 동기 | Crash-recovery | Reliable | 5가지 일관성 수준 |
| Apache ZooKeeper | 부분 동기 | Crash-recovery | Fair-loss (TCP 재시도) | ZAB 프로토콜 |
| etcd | 부분 동기 | Crash-recovery | Reliable (TCP) | Raft 기반 |
| CockroachDB | 부분 동기 | Crash-recovery | Reliable | Raft + MVCC |

### 8.2 블록체인 모델

| 시스템 | 타이밍 모델 | 장애 모델 | 링크 모델 | 참고 |
|--------|-----------|----------|----------|------|
| Bitcoin | **동기** (최대 블록 전파 시간 가정) | **Byzantine** (해시 파워의 50%까지) | Fair-loss (gossip) | Nakamoto consensus |
| Ethereum (PoS) | 부분 동기 | Byzantine (검증자의 1/3까지) | Fair-loss | Casper FFG + LMD GHOST |
| Tendermint | 부분 동기 | Byzantine (1/3까지) | Reliable (재전송 포함) | BFT + DPoS |
| Hyperledger Fabric | 부분 동기 | Crash (CFT 모드) 또는 Byzantine (BFT 모드) | Reliable | 플러그형 합의 |

### 8.3 핵심 관찰

**AWS 모델**: AWS 서비스는 일반적으로 데이터 센터 내에서 crash-recovery 장애를 가정하고, 전체 가용 영역 장애를 crash-stop으로 처리합니다. AZ 내 네트워크 링크는 reliable로 취급되고(이중 스위칭 덕분), 리전 간 링크는 fair-loss로 취급됩니다.

**Bitcoin 모델**: Bitcoin은 **동기** 가정을 합니다: 블록이 제한된 시간 내에(대략 10분) 모든 노드에 전파됩니다. 이 가정이 위반되면(예: 네트워크 수준 공격으로), Bitcoin의 safety 보장이 약화됩니다. "6 확인" 규칙은 제한된 지연 가정에 대한 실용적 적응입니다.

**Google Spanner 모델**: Spanner의 TrueTime API는 현재 시간에 대해 구간 `[earliest, latest]`을 제공하며, 실제 시간이 이 구간 내에 있다는 보장을 합니다. 이것은 클럭 불확실성 문제를 **제한된 대기** 문제로 전환합니다 -- 트랜잭션은 글로벌 순서를 보장하기 위해 불확실성 구간이 지나가기를 기다리기만 하면 됩니다.

---

## 9. 코드: 장애 모드 시뮬레이션

### 9.1 메시지 전달 프레임워크

```python
"""
Distributed system simulator with configurable failure modes.
Demonstrates crash-stop, crash-recovery, and Byzantine failures.
"""

import random
import threading
import time
import json
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


class FailureMode(Enum):
    NONE = "none"
    CRASH_STOP = "crash_stop"
    CRASH_RECOVERY = "crash_recovery"
    BYZANTINE = "byzantine"


class LinkMode(Enum):
    RELIABLE = "reliable"
    FAIR_LOSS = "fair_loss"
    ARBITRARY = "arbitrary"


@dataclass
class Message:
    sender: str
    receiver: str
    content: dict
    timestamp: float = field(default_factory=time.time)
    seq_num: int = 0


class Network:
    """Simulated network with configurable link failure modes."""

    def __init__(self, link_mode: LinkMode, loss_rate: float = 0.3):
        self.link_mode = link_mode
        self.loss_rate = loss_rate
        self.message_queues: dict[str, list[Message]] = defaultdict(list)
        self.lock = threading.Lock()
        self.delivered_count = 0
        self.lost_count = 0
        self.corrupted_count = 0

    def send(self, msg: Message):
        """Send a message through the network, applying link failure model."""
        with self.lock:
            if self.link_mode == LinkMode.RELIABLE:
                # Always deliver, no corruption
                self.message_queues[msg.receiver].append(msg)
                self.delivered_count += 1

            elif self.link_mode == LinkMode.FAIR_LOSS:
                # May lose messages, but no corruption
                if random.random() > self.loss_rate:
                    self.message_queues[msg.receiver].append(msg)
                    self.delivered_count += 1
                else:
                    self.lost_count += 1

            elif self.link_mode == LinkMode.ARBITRARY:
                # May lose, corrupt, duplicate, or fabricate
                roll = random.random()
                if roll < 0.3:
                    # Lost
                    self.lost_count += 1
                elif roll < 0.5:
                    # Corrupted
                    corrupted = Message(
                        sender=msg.sender,
                        receiver=msg.receiver,
                        content={"corrupted": True, "original": str(msg.content)},
                        timestamp=msg.timestamp,
                    )
                    self.message_queues[msg.receiver].append(corrupted)
                    self.corrupted_count += 1
                elif roll < 0.6:
                    # Duplicated
                    self.message_queues[msg.receiver].append(msg)
                    self.message_queues[msg.receiver].append(msg)
                    self.delivered_count += 2
                else:
                    # Normal delivery
                    self.message_queues[msg.receiver].append(msg)
                    self.delivered_count += 1

    def receive(self, node_id: str) -> Optional[Message]:
        """Receive next message for a node (non-blocking)."""
        with self.lock:
            if self.message_queues[node_id]:
                return self.message_queues[node_id].pop(0)
            return None

    def stats(self) -> dict:
        return {
            "delivered": self.delivered_count,
            "lost": self.lost_count,
            "corrupted": self.corrupted_count,
        }


class Process:
    """A process in the distributed system with configurable failure mode."""

    def __init__(
        self,
        node_id: str,
        network: Network,
        failure_mode: FailureMode = FailureMode.NONE,
        stable_storage_path: Optional[str] = None,
    ):
        self.node_id = node_id
        self.network = network
        self.failure_mode = failure_mode
        self.stable_storage_path = stable_storage_path

        # Process state
        self.alive = True
        self.state = {"value": None, "term": 0, "log": []}
        self.messages_sent = 0
        self.messages_received = 0

        # Recover from stable storage if crash-recovery
        if failure_mode == FailureMode.CRASH_RECOVERY and stable_storage_path:
            self._recover()

    def _recover(self):
        """Recover state from stable storage."""
        if self.stable_storage_path and os.path.exists(self.stable_storage_path):
            with open(self.stable_storage_path, 'r') as f:
                saved = json.load(f)
                self.state.update(saved)
                print(f"[{self.node_id}] Recovered state from disk: {saved}")

    def _persist(self):
        """Write current state to stable storage."""
        if self.stable_storage_path:
            with open(self.stable_storage_path, 'w') as f:
                json.dump(self.state, f)

    def crash(self):
        """Simulate a process crash."""
        self.alive = False
        print(f"[{self.node_id}] CRASHED (mode={self.failure_mode.value})")

        if self.failure_mode == FailureMode.CRASH_RECOVERY:
            self._persist()  # save state before crash

    def recover(self):
        """Recover from a crash (only for crash-recovery mode)."""
        if self.failure_mode != FailureMode.CRASH_RECOVERY:
            raise RuntimeError("Only crash-recovery processes can recover")
        self.alive = True
        self._recover()
        print(f"[{self.node_id}] RECOVERED")

    def send(self, receiver_id: str, content: dict):
        """Send a message, applying failure mode behavior."""
        if not self.alive:
            return  # crashed processes do not send

        if self.failure_mode == FailureMode.BYZANTINE:
            # Byzantine process may send arbitrary content
            if random.random() < 0.4:
                # Send correct message
                msg = Message(self.node_id, receiver_id, content)
            elif random.random() < 0.5:
                # Send conflicting value
                fake_content = dict(content)
                if "value" in fake_content:
                    fake_content["value"] = f"FAKE_{random.randint(0,99)}"
                msg = Message(self.node_id, receiver_id, fake_content)
                print(f"[{self.node_id}] BYZANTINE: sent fake to {receiver_id}")
            else:
                # Send different messages to different nodes (equivocation)
                fake_content = {"value": f"EQUIVOC_{receiver_id}"}
                msg = Message(self.node_id, receiver_id, fake_content)
                print(f"[{self.node_id}] BYZANTINE: equivocation to {receiver_id}")
        else:
            msg = Message(self.node_id, receiver_id, content)

        self.network.send(msg)
        self.messages_sent += 1

    def receive(self) -> Optional[Message]:
        """Receive a message."""
        if not self.alive:
            return None  # crashed processes do not receive
        msg = self.network.receive(self.node_id)
        if msg:
            self.messages_received += 1
        return msg
```

### 9.2 장애 시뮬레이션 실행

```python
def simulate_broadcast(
    num_nodes: int = 5,
    failure_mode: FailureMode = FailureMode.NONE,
    link_mode: LinkMode = LinkMode.RELIABLE,
    num_faulty: int = 1,
):
    """
    Simulate a simple broadcast protocol under different failure models.
    Node 0 broadcasts a value; all nodes try to agree on it.
    """
    print(f"\n{'='*60}")
    print(f"Simulation: {num_nodes} nodes, {failure_mode.value} failures, "
          f"{link_mode.value} links, {num_faulty} faulty")
    print(f"{'='*60}\n")

    network = Network(link_mode)
    processes = []

    for i in range(num_nodes):
        mode = failure_mode if i < num_faulty else FailureMode.NONE
        storage_path = f"/tmp/node_{i}.json" if mode == FailureMode.CRASH_RECOVERY else None
        p = Process(f"node_{i}", network, mode, storage_path)
        processes.append(p)

    # Node with highest ID is the broadcaster
    broadcaster = processes[-1]
    proposal = {"type": "PROPOSE", "value": "COMMIT_TX_42"}

    # Phase 1: Broadcaster sends proposal to all
    print(f"[{broadcaster.node_id}] Broadcasting: {proposal}")
    for p in processes:
        if p.node_id != broadcaster.node_id:
            broadcaster.send(p.node_id, proposal)

    # Inject failures
    if failure_mode == FailureMode.CRASH_STOP and num_faulty > 0:
        processes[0].crash()
    elif failure_mode == FailureMode.CRASH_RECOVERY and num_faulty > 0:
        processes[0].crash()
        time.sleep(0.1)
        processes[0].recover()

    # Phase 2: Each node receives and echoes
    decisions = {}
    for p in processes:
        msg = p.receive()
        if msg:
            print(f"[{p.node_id}] Received: {msg.content}")
            decisions[p.node_id] = msg.content.get("value")
        else:
            if p.alive:
                print(f"[{p.node_id}] No message received")
            else:
                print(f"[{p.node_id}] Crashed, cannot receive")

    # Check agreement
    print(f"\nDecisions: {decisions}")
    unique_values = set(decisions.values())
    if len(unique_values) <= 1:
        print("RESULT: Agreement achieved")
    else:
        print(f"RESULT: DISAGREEMENT detected! Values: {unique_values}")

    print(f"Network stats: {network.stats()}")
    return decisions


# Run simulations with different failure models
if __name__ == "__main__":
    # Scenario 1: No failures, reliable links
    simulate_broadcast(5, FailureMode.NONE, LinkMode.RELIABLE, 0)

    # Scenario 2: One crash-stop failure
    simulate_broadcast(5, FailureMode.CRASH_STOP, LinkMode.RELIABLE, 1)

    # Scenario 3: Crash-recovery failure
    simulate_broadcast(5, FailureMode.CRASH_RECOVERY, LinkMode.RELIABLE, 1)

    # Scenario 4: Byzantine failure
    simulate_broadcast(5, FailureMode.BYZANTINE, LinkMode.RELIABLE, 1)

    # Scenario 5: No process failures, but fair-loss links
    simulate_broadcast(5, FailureMode.NONE, LinkMode.FAIR_LOSS, 0)

    # Scenario 6: Byzantine + arbitrary links (worst case)
    simulate_broadcast(5, FailureMode.BYZANTINE, LinkMode.ARBITRARY, 1)
```

### 9.3 샘플 출력 분석

```
============================================================
Simulation: 5 nodes, crash_stop failures, reliable links, 1 faulty
============================================================

[node_4] Broadcasting: {'type': 'PROPOSE', 'value': 'COMMIT_TX_42'}
[node_0] CRASHED (mode=crash_stop)
[node_0] Crashed, cannot receive
[node_1] Received: {'type': 'PROPOSE', 'value': 'COMMIT_TX_42'}
[node_2] Received: {'type': 'PROPOSE', 'value': 'COMMIT_TX_42'}
[node_3] Received: {'type': 'PROPOSE', 'value': 'COMMIT_TX_42'}

Decisions: {'node_1': 'COMMIT_TX_42', 'node_2': 'COMMIT_TX_42', 'node_3': 'COMMIT_TX_42'}
RESULT: Agreement achieved
Network stats: {'delivered': 4, 'lost': 0, 'corrupted': 0}
```

**관찰**: Crash-stop 장애에서는 crash된 노드가 단순히 참여하지 않습니다. 나머지 4개 노드(broadcaster 포함)가 값에 합의합니다. `n = 5`이고 `f = 1`이면, `n = 5 ≥ 2(1) + 1 = 3`이므로 합의가 달성 가능합니다.

### 9.4 장애 감지기 시뮬레이션

```python
class FailureDetector:
    """
    Simulates different classes of failure detectors.

    - Perfect (P): No false positives, no false negatives. Only possible in synchronous systems.
    - Eventually Perfect (◇P): May make mistakes initially, but eventually becomes accurate.
    - Eventually Strong (◇S): Eventually suspects every crashed process, and eventually stops
      suspecting some correct process.
    """

    def __init__(self, nodes: list[str], timeout: float = 1.0, detector_type: str = "eventually_perfect"):
        self.nodes = nodes
        self.timeout = timeout
        self.detector_type = detector_type
        self.last_heartbeat: dict[str, float] = {n: time.time() for n in nodes}
        self.suspected: set[str] = set()
        self.mistakes = 0  # track false positives
        self.actually_crashed: set[str] = set()
        self._gst_reached = False
        self._gst_time = time.time() + random.uniform(2, 5)

    def heartbeat(self, node_id: str):
        """Receive a heartbeat from a node."""
        self.last_heartbeat[node_id] = time.time()
        if node_id in self.suspected:
            self.suspected.discard(node_id)
            if node_id not in self.actually_crashed:
                print(f"  Detector: Corrected false suspicion of {node_id}")

    def mark_crashed(self, node_id: str):
        """Mark a node as actually crashed (ground truth)."""
        self.actually_crashed.add(node_id)

    def check(self) -> set[str]:
        """Run failure detection and return suspected nodes."""
        now = time.time()

        if self.detector_type == "perfect":
            # Perfect detector: knows exactly who crashed (unrealistic)
            self.suspected = self.actually_crashed.copy()

        elif self.detector_type == "eventually_perfect":
            # Before GST: may make mistakes
            if now < self._gst_time:
                for node in self.nodes:
                    elapsed = now - self.last_heartbeat[node]
                    if elapsed > self.timeout:
                        self.suspected.add(node)
                    # Random false suspicion before GST
                    if random.random() < 0.1 and node not in self.actually_crashed:
                        self.suspected.add(node)
                        self.mistakes += 1
            else:
                # After GST: accurate
                if not self._gst_reached:
                    print(f"  Detector: GST reached at t={now:.2f}, becoming accurate")
                    self._gst_reached = True
                self.suspected = set()
                for node in self.nodes:
                    elapsed = now - self.last_heartbeat[node]
                    if elapsed > self.timeout:
                        self.suspected.add(node)

        return self.suspected.copy()

    def accuracy_report(self) -> dict:
        """Report on detector accuracy."""
        true_positives = self.suspected & self.actually_crashed
        false_positives = self.suspected - self.actually_crashed
        false_negatives = self.actually_crashed - self.suspected
        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "total_mistakes": self.mistakes,
            "gst_reached": self._gst_reached,
        }
```

### 9.5 Perfect 감지의 불가능성 시연

```python
def demonstrate_detection_impossibility():
    """
    Show why perfect failure detection is impossible
    in an asynchronous system.
    """
    print("\n" + "="*60)
    print("Demonstrating: Failure Detection in Async Systems")
    print("="*60)

    scenarios = [
        {
            "name": "Crashed node",
            "node_b_delay": float('inf'),  # never responds
            "node_b_alive": False,
        },
        {
            "name": "Slow node (GC pause)",
            "node_b_delay": 5.0,  # responds after 5 seconds
            "node_b_alive": True,
        },
        {
            "name": "Network partition",
            "node_b_delay": float('inf'),  # messages never arrive
            "node_b_alive": True,
        },
    ]

    timeout = 2.0  # detector timeout

    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"  Node B alive: {scenario['node_b_alive']}")
        print(f"  Response delay: {scenario['node_b_delay']}s")
        print(f"  Detector timeout: {timeout}s")

        if scenario['node_b_delay'] > timeout:
            print(f"  Detector verdict: SUSPECTED (no response within {timeout}s)")
            if scenario['node_b_alive']:
                print(f"  Reality: FALSE POSITIVE - Node B is alive but slow/partitioned!")
            else:
                print(f"  Reality: CORRECT - Node B is indeed crashed")
        else:
            print(f"  Detector verdict: ALIVE (response within {timeout}s)")
            if not scenario['node_b_alive']:
                print(f"  Reality: This case is impossible (dead nodes can't respond)")
            else:
                print(f"  Reality: CORRECT - Node B is alive")

    print(f"\nConclusion: With timeout={timeout}s, the detector CANNOT distinguish")
    print(f"a crashed node from a slow/partitioned node. This is fundamental,")
    print(f"not a limitation of the timeout value.")


demonstrate_detection_impossibility()
```

---

## 10. 요약과 핵심 정리

### 시스템 모델 치트 시트

```
┌─────────────────────────────────────────────────────────────────┐
│                     SYSTEM MODEL COMPONENTS                     │
├───────────────┬───────────────────┬─────────────────────────────┤
│ TIMING        │ FAILURE           │ LINK                        │
│               │                   │                             │
│ Synchronous   │ Crash-Stop        │ Reliable                    │
│  • Known Δ    │  • Stop forever   │  • No loss, no corruption   │
│  • Known Φ    │  • n ≥ 2f+1       │  • Exactly-once delivery    │
│               │                   │                             │
│ Partial Sync  │ Crash-Recovery    │ Fair-Loss                   │
│  • Δ after GST│  • Stop + recover │  • May lose, no corruption  │
│  • Most real  │  • Stable storage │  • Retransmit → reliable    │
│    systems    │  • n ≥ 2f+1       │                             │
│               │                   │                             │
│ Asynchronous  │ Byzantine         │ Arbitrary                   │
│  • No bounds  │  • Any behavior   │  • Lose, corrupt, fabricate │
│  • FLP applies│  • n ≥ 3f+1       │  • Need crypto to mitigate  │
└───────────────┴───────────────────┴─────────────────────────────┘
```

### 핵심 원칙

1. **구축 전에 모델링하라**: 프로토콜을 설계하기 전에 항상 시스템 모델 가정을 명시적으로 기술하라.
2. **약한 가정 = 강한 보장**: 약한 모델에서 올바르다고 증명된 알고리즘이 더 많은 환경에서 동작한다.
3. **Liveness보다 Safety**: 둘 다 가질 수 없을 때(FLP), 항상 safety를 보존하라.
4. **부분 동기성이 실용적 최적 지점**: 실제 동작을 포착하고 합의를 가능하게 한다.
5. **Byzantine 허용은 비싸다**: Crash에서 Byzantine 허용으로의 전환은 노드(3f+1 vs 2f+1)와 메시지(O(n^3) vs O(n^2)) 모두에 비용이 든다.

---

## 11. 연습 문제

### 문제 1: 모델 분류

아래 각 시스템에 대해 가장 적절한 타이밍 모델, 장애 모델, 링크 모델을 식별하세요:

1. 전용 스위치로 연결된 같은 랙의 5대 서버 클러스터에서 데이터베이스 실행
2. 전 세계 10,000개 노드에 걸친 P2P 파일 공유 네트워크
3. 누구나 참여하거나 떠날 수 있는 블록체인 네트워크
4. 자율 주행 차량을 위한 실시간 제어 시스템
5. 전용 회선으로 연결된 3개 데이터 센터에서 금융 데이터베이스 복제

### 문제 2: Safety vs Liveness

각 속성을 분류하고 추론을 설명하세요:

1. "동일한 출금에 대해 두 대의 ATM이 돈을 인출하지 않는다"
2. "모든 ATM 출금 요청이 결국 처리된다"
3. "은행 잔고는 항상 0 이상이다"
4. "입금된 모든 수표가 결국 입금 처리된다"
5. "이체 금액은 출발지에서 출금된 후 목적지에 입금된다"

### 문제 3: 장애 허용 계산

시스템에 7개 노드가 있습니다. 다음 하에서 허용할 수 있는 장애 노드 `f`의 최대 수를 계산하세요:

1. Crash-stop 장애 (합의에 n ≥ 2f + 1 필요)
2. 서명 없는 메시지를 사용한 Byzantine 장애 (n ≥ 3f + 1 필요)
3. 디지털 서명을 사용한 Byzantine 장애 (n ≥ 2f + 1 필요)

### 문제 4: 코드 도전

섹션 9의 시뮬레이션 코드를 확장하여:

1. Fair-loss 링크를 reliable 링크로 변환하는 stubborn 재전송 계층 구현
2. 장애 노드가 다수에게는 올바른 값을, 소수에게는 잘못된 값을 보내는(감지를 어렵게 만드는) Byzantine 장애 모드 추가
3. 시간에 따른 정확도를 추적하는 eventually perfect 장애 감지기 구현

### 문제 5: 실제 분석

선택한 분산 데이터베이스의 Jepsen 분석을 읽으세요 (https://jepsen.io/analyses). 답하세요:

1. 데이터베이스가 주장하는 시스템 모델은 무엇인가?
2. 실제로 운영되는 시스템 모델은 무엇인가 (Jepsen의 발견 기반)?
3. 어떤 safety 위반이 발견되었나?
4. 이러한 위반은 잘못된 모델 가정 때문인가, 구현 버그 때문인가?

---

## 12. 참고 문헌

1. Fischer, M. J., Lynch, N. A., & Paterson, M. S. (1985). "Impossibility of Distributed Consensus with One Faulty Process." *Journal of the ACM*, 32(2), 374-382.
2. Lamport, L., Shostak, R., & Pease, M. (1982). "The Byzantine Generals Problem." *ACM Transactions on Programming Languages and Systems*, 4(3), 382-401.
3. Dwork, C., Lynch, N., & Stockmeyer, L. (1988). "Consensus in the Presence of Partial Synchrony." *Journal of the ACM*, 35(2), 288-323.
4. Alpern, B., & Schneider, F. B. (1985). "Defining Liveness." *Information Processing Letters*, 21(4), 181-185.
5. Chandra, T. D., & Toueg, S. (1996). "Unreliable Failure Detectors for Reliable Distributed Systems." *Journal of the ACM*, 43(2), 225-267.
6. Cachin, C., Guerraoui, R., & Rodrigues, L. (2011). *Introduction to Reliable and Secure Distributed Programming*. Springer.
7. Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly Media.

---

[다음: 레슨 02 - 시간, 클럭, 순서 결정](./02_Time_Clocks_and_Ordering.md)

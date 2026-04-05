# 레슨 2: 시간, 클럭, 순서 결정

[개요](./00_Overview.md) | [이전](./01_System_Models_and_Failure_Modes.md) | [다음](./03_FLP_Impossibility_and_Bounds.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있게 됩니다:

1. 물리적 클럭이 분산 시스템에서 이벤트 순서 결정에 불충분한 이유를 설명하고 클럭 오류의 원인을 정량화
2. Happens-before 관계를 정의하고 이를 사용하여 이벤트의 인과적 순서를 결정
3. Lamport 타임스탬프를 구현하고 동시성 감지에 대한 한계를 이해
4. Vector 클럭을 구현하여 이벤트의 전체 인과 이력을 캡처하고 동시 연산을 감지
5. Hybrid Logical Clock(HLC)과 Google Spanner의 TrueTime을 분산 시간의 실용적 접근법으로 설명

---

## 목차

1. [물리적 시간의 문제](#1-물리적-시간의-문제)
2. [Happens-Before 관계](#2-happens-before-관계)
3. [Lamport 타임스탬프](#3-lamport-타임스탬프)
4. [Vector 클럭](#4-vector-클럭)
5. [Version Vector vs Vector 클럭](#5-version-vector-vs-vector-클럭)
6. [Hybrid Logical Clock (HLC)](#6-hybrid-logical-clock-hlc)
7. [TrueTime과 구간 기반 클럭](#7-truetime과-구간-기반-클럭)
8. [클럭 메커니즘 비교](#8-클럭-메커니즘-비교)
9. [코드: 전체 구현](#9-코드-전체-구현)
10. [연습 문제](#10-연습-문제)
11. [요약](#11-요약)
12. [참고 문헌](#12-참고-문헌)

---

## 1. 물리적 시간의 문제

### 1.1 "순서"라는 개념이 필요한 이유

단일 프로세스 프로그램에서 이벤트의 순서는 자명합니다: 명령어 `A`가 프로그램에서 `B`보다 먼저 나타나면 `A`가 `B` 전에 발생합니다. 분산 시스템에서는 단일 명령어 시퀀스가 없습니다. 이벤트는 서로 다른 노드에서 발생하며, "이벤트 `A`가 이벤트 `B` 전에 발생했는가?"라는 질문에 답할 방법이 필요합니다.

이 질문이 중요한 이유:

- **데이터베이스 일관성**: 쓰기가 읽기 전에 발생했는가, 후에 발생했는가?
- **분산 디버깅**: 어떤 로그 항목이 먼저였는가?
- **충돌 해결**: 어떤 업데이트가 승리해야 하는가?
- **인과적 메시징**: 이 응답이 응답 대상 메시지 후에 왔는가?

### 1.2 물리적 클럭 소스

모든 컴퓨터에는 물리적 클럭이 있지만, 이 클럭들은 불완전합니다:

| 클럭 소스 | 일반적 정확도 | 드리프트율 | 참고 |
|----------|-------------|----------|------|
| 수정 발진기 | ~10 ppm | ~1초/일 | 표준 PC 하드웨어 |
| NTP 동기화 | ~1-10 ms | 주기적 보정 | 네트워크 경로에 의존 |
| PTP (IEEE 1588) | ~1-100 μs | 하드웨어 지원 | PTP 지원 NIC 필요 |
| GPS 수신기 | ~10-100 ns | 원자 시계 수준 | 안테나, 하늘 가시성 필요 |
| 원자 시계 (Cs/Rb) | ~1 ns | ~10⁻¹² s/s | 고가, 데이터 센터에서 사용 |
| Google TrueTime | ~1-7 ms 불확실성 | GPS + 원자 | 커스텀 인프라 |

### 1.3 클럭 오류의 원인

**클럭 드리프트**: 수정 발진기는 온도, 전압, 노화에 따라 달라지는 주파수로 진동합니다. 일반적인 드리프트율 10 ppm(백만분의 일)은:

```
10 ppm = 10 × 10⁻⁶ = 0.00001

Over 1 day (86,400 seconds):
  drift = 86,400 × 0.00001 = 0.864 seconds

Over 1 week:
  drift ≈ 6 seconds

Two independent clocks can diverge by up to:
  2 × drift = 1.7 seconds/day
```

**클럭 스큐**: 주어진 실제 시간에서 두 클럭 간의 순간적 차이입니다.

```
skew(A, B) = C_A(t) - C_B(t)

where C_A(t) is the reading of clock A at real time t
```

**NTP 보정 문제**:

1. **단계 조정**: NTP가 클럭을 앞이나 뒤로 점프시킬 수 있어, 타임스탬프가 뒤로 가는(비단조적) 현상이 발생합니다.
2. **슬루 조정**: NTP가 드리프트를 점진적으로 보정하기 위해 클럭 속도를 높이거나 낮출 수 있어, 시간 흐름의 속도가 변합니다.
3. **네트워크 비대칭**: NTP는 대칭적 네트워크 지연을 가정하지만, 실제 경로는 종종 비대칭이어서 체계적 오류를 도입합니다.

```
NTP round-trip measurement:

  Client ──── request ────► Server
    t₁                        t₂
                              t₃
  Client ◄─── response ──── Server
    t₄

  Round-trip delay: δ = (t₄ - t₁) - (t₃ - t₂)
  Estimated offset: θ = ((t₂ - t₁) + (t₃ - t₄)) / 2

  If network is asymmetric (d₁ ≠ d₂):
    True offset = θ + (d₁ - d₂) / 2
    Error bound = ± δ/2
```

**윤초**: UTC는 지구의 자전에 맞추기 위해 가끔 1초를 추가(또는 이론적으로 제거)합니다. 이는 `23:59:59` 다음에 `00:00:00` 전에 `23:59:60`이 오는 것을 의미합니다. 분당 60초 또는 단조 증가 타임스탬프를 가정하는 소프트웨어가 깨질 수 있습니다.

### 1.4 물리적 클럭이 이벤트를 순서화할 수 없는 이유

서로 다른 노드의 두 이벤트를 고려합니다:

```
Node A: event_a at physical time T_A = 100.003
Node B: event_b at physical time T_B = 100.005

Question: Did event_a happen before event_b?

If clock skew between A and B is ±5ms:
  Real time of event_a could be anywhere in [99.998, 100.008]
  Real time of event_b could be anywhere in [100.000, 100.010]

  These intervals OVERLAP, so we CANNOT determine the order.
```

이것은 클럭의 한계가 아닙니다 -- 빛의 속도의 유한성과 네트워크 지연의 불확실성의 근본적인 결과입니다. 완벽한 클럭이 있더라도, 서로 다른 위치에서 두 위치 간 빛 이동 시간 내에 발생하는 두 이벤트는 잘 정의된 순서가 없습니다 (이것은 본질적으로 특수 상대성 이론의 동시성에 대한 통찰과 동일합니다).

**결론**: 물리적 시간에 의존하지 않고 이벤트의 인과적 구조를 포착하는 **논리적 클럭**이 필요합니다.

---

## 2. Happens-Before 관계

### 2.1 Lamport의 통찰 (1978)

Leslie Lamport의 1978년 기념비적 논문 "Time, Clocks, and the Ordering of Events in a Distributed System"은 `→`로 표시되는 **happens-before** 관계를 소개했습니다.

**핵심 통찰**: 이벤트가 어떤 시간에 발생했는지 알 필요가 없습니다. 한 이벤트가 다른 이벤트에 **인과적으로 영향을 줄 수 있었는지** 여부만 알면 됩니다.

### 2.2 형식적 정의

프로세스 `p₁, p₂, ..., pₙ`을 가진 분산 시스템에서, happens-before 관계 `→`는 다음을 만족하는 가장 작은 관계입니다:

1. **프로세스 순서**: `a`와 `b`가 같은 프로세스의 이벤트이고, 해당 프로세스의 실행에서 `a`가 `b` 전에 발생하면, `a → b`입니다.

2. **메시지 인과성**: `a`가 한 프로세스에 의한 메시지 전송이고 `b`가 다른 프로세스에 의한 같은 메시지의 수신이면, `a → b`입니다.

3. **전이성**: `a → b`이고 `b → c`이면, `a → c`입니다.

`a → b`도 아니고 `b → a`도 아니면, `a`와 `b`는 **동시적**이며, `a ‖ b`로 씁니다.

### 2.3 Happens-Before 시각화

```
Process P1    Process P2    Process P3
    │              │              │
    a              │              │
    │──── m1 ─────►│              │
    │              b              │
    │              │──── m2 ─────►│
    │              │              c
    │              │              │
    d              │              │
    │              │              │
    │              e              │
    │◄──── m3 ────│              │
    f              │              │
    │              │              │

Happens-before pairs:
  a → b  (m1: send before receive)
  b → c  (m2: send before receive)
  a → c  (transitivity: a → b → c)
  e → f  (m3: send before receive)
  a → d  (process order in P1)
  a → f  (process order: a → d → f, or transitivity)

Concurrent pairs:
  d ‖ b  (no causal path between them)
  d ‖ c  (no causal path)
  d ‖ e  (no causal path)
  a ‖ e  (no causal path from a to e or e to a)
```

### 2.4 Happens-Before의 속성

| 속성 | 성립? | 설명 |
|------|-------|------|
| 비반사적 | 예 | ¬(a → a); 이벤트는 자기 자신 전에 발생하지 않음 |
| 반대칭적 | 예 | a → b이면 ¬(b → a) |
| 전이적 | 예 | a → b ∧ b → c ⟹ a → c |
| 전순서? | **아니오** | 동시적 이벤트는 비교 불가능 |
| 부분순서? | 예 (엄격한) | 엄격한 부분순서 |

### 2.5 인과성 vs Happens-Before

Happens-before 관계는 실제 인과성이 아닌 **잠재적 인과성**을 포착합니다. `a → b`이면, `a`가 `b`에 영향을 *줄 수 있었다*는 것이지, *주었다*는 것은 아닙니다.

```
a → b means:
  "Information could have flowed from a to b"
  NOT: "a caused b"

a ‖ b means:
  "Information could NOT have flowed between a and b"
  Therefore: "a and b are causally independent"
```

이 구분이 중요한 이유:
- `a → b`는 `a`가 `b`를 야기하기 위한 필요 조건이지, 충분 조건은 아닙니다.
- `a ‖ b`는 `a`가 `b`를 야기하지 않았다는 것을 보장하기에 충분합니다.

---

## 3. Lamport 타임스탬프

### 3.1 알고리즘

각 프로세스 `pᵢ`는 카운터 `Cᵢ` (0으로 초기화)를 유지합니다. 규칙은 다음과 같습니다:

1. **내부 이벤트**: 프로세스 `pᵢ`에서 어떤 이벤트 전에 카운터를 증가:
   ```
   Cᵢ = Cᵢ + 1
   ```

2. **전송 이벤트**: 메시지 `m`을 보내기 전에 카운터를 증가시키고 메시지에 부착:
   ```
   Cᵢ = Cᵢ + 1
   send(m, Cᵢ)
   ```

3. **수신 이벤트**: 타임스탬프 `t`를 가진 메시지 `m`을 수신하면 카운터를 갱신:
   ```
   Cᵢ = max(Cᵢ, t) + 1
   ```

### 3.2 클럭 조건

Lamport 타임스탬프는 **클럭 조건**을 만족합니다:

```
If a → b, then C(a) < C(b)
```

**하지만 역은 참이 아닙니다**:

```
C(a) < C(b) does NOT imply a → b
```

이는 Lamport 타임스탬프가 "`a`가 `b` 전에 발생했다면, `C(a) < C(b)`"라고 말할 수 있지만, `C(a) < C(b)`를 관찰하더라도 `a → b`라고 결론짓지 못한다는 것을 의미합니다. 이벤트들은 동시적일 수 있습니다.

### 3.3 시각화

```
Process P1 (C₁)    Process P2 (C₂)    Process P3 (C₃)
    │                    │                    │
  C₁=1  a               │                    │
    │──── m1(1) ────────►│                    │
    │                  C₂=2  b                │
    │                    │──── m2(2) ─────────►│
    │                    │                   C₃=3  c
  C₁=2  d               │                    │
    │                    │                    │
    │                  C₂=3  e                │
    │◄──── m3(3) ────────│                    │
  C₁=4  f               │                    │
    │                    │                    │

Timestamps:
  C(a) = 1, C(b) = 2, C(c) = 3, C(d) = 2, C(e) = 3, C(f) = 4

Note: C(d) = 2 and C(b) = 2, yet d ‖ b (concurrent)
      C(d) = 2 < C(e) = 3, yet d ‖ e (concurrent)

Lamport timestamps CANNOT detect concurrency!
```

### 3.4 Lamport 타임스탬프를 이용한 전순서

**전순서**(상호 배제 같은 것에 유용)를 만들려면, 프로세스 ID로 동점을 해결합니다:

```
(C(a), pid_a) < (C(b), pid_b) iff:
  C(a) < C(b), or
  C(a) = C(b) and pid_a < pid_b
```

이것은 happens-before 관계와 **일관된** 전순서를 제공하지만, 동일하지는 않습니다.

### 3.5 구현

```python
class LamportClock:
    """Lamport logical clock implementation."""

    def __init__(self, process_id: str):
        self.process_id = process_id
        self.counter = 0

    def tick(self) -> int:
        """Increment clock for an internal event."""
        self.counter += 1
        return self.counter

    def send_timestamp(self) -> int:
        """Get timestamp for an outgoing message."""
        self.counter += 1
        return self.counter

    def receive_timestamp(self, msg_timestamp: int) -> int:
        """Update clock upon receiving a message."""
        self.counter = max(self.counter, msg_timestamp) + 1
        return self.counter

    def current(self) -> int:
        """Return current clock value without incrementing."""
        return self.counter

    def __repr__(self):
        return f"LamportClock({self.process_id}, t={self.counter})"
```

### 3.6 한계

| 한계 | 결과 |
|------|------|
| 동시성 감지 불가 | `C(a) < C(b)`가 `a → b`를 의미하지 않음 |
| 인과 이력 없음 | 이벤트의 인과적 과거를 결정할 수 없음 |
| 카운터 크기 | 무한 성장 (실제로는 느리게) |
| 동점 해결이 임의적 | 전순서가 실시간 순서와 일치하지 않을 수 있음 |

동시성을 감지할 수 없다는 것이 **vector 클럭**의 주요 동기입니다.

---

## 4. Vector 클럭

### 4.1 동기

다음과 같은 클럭 메커니즘을 원합니다:

```
V(a) < V(b)  if and only if  a → b
```

이를 위해서는 각 프로세스가 자신의 카운터뿐만 아니라, 시스템의 **모든 프로세스의 최신 카운터**를 추적해야 합니다.

### 4.2 정의

`n`개 프로세스 `{p₁, p₂, ..., pₙ}` 시스템에서, 각 프로세스 `pᵢ`는 `n`개 정수의 벡터 `Vᵢ`를 유지하며, `Vᵢ[j]`는 `pᵢ`가 알고 있는 `pⱼ`의 최신 이벤트 카운터를 나타냅니다.

**규칙**:

1. **pᵢ에서 내부 이벤트**: 자신의 컴포넌트를 증가:
   ```
   Vᵢ[i] = Vᵢ[i] + 1
   ```

2. **pᵢ에서 전송 이벤트**: 자신의 컴포넌트를 증가시키고 벡터를 메시지에 부착:
   ```
   Vᵢ[i] = Vᵢ[i] + 1
   send(m, Vᵢ)
   ```

3. **pᵢ에서 pⱼ로부터 부착 벡터 Vₘ으로 수신 이벤트**: 요소별 최대값을 취한 후, 자신의 컴포넌트를 증가:
   ```
   Vᵢ[k] = max(Vᵢ[k], Vₘ[k])  for all k
   Vᵢ[i] = Vᵢ[i] + 1
   ```

### 4.3 비교 규칙

두 벡터 타임스탬프 `V(a)`와 `V(b)`가 주어지면:

```
V(a) ≤ V(b)   iff  ∀k: V(a)[k] ≤ V(b)[k]
V(a) < V(b)    iff  V(a) ≤ V(b) and V(a) ≠ V(b)
V(a) ‖ V(b)   iff  ¬(V(a) ≤ V(b)) and ¬(V(b) ≤ V(a))
```

**Vector 클럭의 기본 정리**:

```
a → b  ⟺  V(a) < V(b)
a ‖ b  ⟺  V(a) ‖ V(b)
```

이 양방향 함의가 vector 클럭을 Lamport 타임스탬프보다 엄밀히 더 강력하게 만듭니다.

### 4.4 시각화

```
Process P1         Process P2         Process P3
V₁=[0,0,0]        V₂=[0,0,0]        V₃=[0,0,0]
    │                   │                   │
  [1,0,0]  a            │                   │
    │──── m1 ──────────►│                   │
    │                [1,1,0]  b              │
    │                   │──── m2 ──────────►│
    │                   │               [1,1,1]  c
  [2,0,0]  d            │                   │
    │                   │                   │
    │                [1,2,0]  e              │
    │◄──── m3 ──────────│                   │
  [2,2,0]               │                   │
  [3,2,0]  f            │                   │
    │                   │                   │

Checking relationships:
  a → b?  V(a)=[1,0,0] < V(b)=[1,1,0]?  [1≤1, 0≤1, 0≤0] and ≠  → YES
  d ‖ b?  V(d)=[2,0,0] vs V(b)=[1,1,0]?  2>1 but 0<1 → incomparable → YES
  d ‖ e?  V(d)=[2,0,0] vs V(e)=[1,2,0]?  2>1 but 0<2 → incomparable → YES
  a → c?  V(a)=[1,0,0] < V(c)=[1,1,1]?  [1≤1, 0≤1, 0≤1] and ≠  → YES
  a ‖ e?  V(a)=[1,0,0] vs V(e)=[1,2,0]?  [1≤1, 0≤2, 0≤0] → V(a) ≤ V(e) → a → e
  Wait! Is that right? Let us trace:
    a → b (via m1), b is at P2, then P2 does e. Process order: b → e.
    Therefore a → b → e, so a → e. Confirmed!
```

### 4.5 구현

```python
from copy import deepcopy


class VectorClock:
    """Vector clock implementation for a distributed system."""

    def __init__(self, process_id: str, all_process_ids: list[str]):
        self.process_id = process_id
        self.process_ids = sorted(all_process_ids)
        self.index = self.process_ids.index(process_id)
        self.vector = [0] * len(self.process_ids)

    def tick(self) -> list[int]:
        """Increment clock for an internal event."""
        self.vector[self.index] += 1
        return self.get()

    def send_timestamp(self) -> list[int]:
        """Get timestamp for an outgoing message."""
        self.vector[self.index] += 1
        return self.get()

    def receive_timestamp(self, msg_vector: list[int]) -> list[int]:
        """Update clock upon receiving a message with attached vector."""
        for i in range(len(self.vector)):
            self.vector[i] = max(self.vector[i], msg_vector[i])
        self.vector[self.index] += 1
        return self.get()

    def get(self) -> list[int]:
        """Return a copy of the current vector."""
        return list(self.vector)

    @staticmethod
    def happens_before(v1: list[int], v2: list[int]) -> bool:
        """Check if v1 < v2 (v1 happens-before v2)."""
        leq = all(a <= b for a, b in zip(v1, v2))
        neq = any(a < b for a, b in zip(v1, v2))
        return leq and neq

    @staticmethod
    def concurrent(v1: list[int], v2: list[int]) -> bool:
        """Check if v1 ‖ v2 (concurrent)."""
        return (not VectorClock.happens_before(v1, v2) and
                not VectorClock.happens_before(v2, v1) and
                v1 != v2)

    @staticmethod
    def compare(v1: list[int], v2: list[int]) -> str:
        """Return the causal relationship between two vector timestamps."""
        if v1 == v2:
            return "EQUAL"
        elif VectorClock.happens_before(v1, v2):
            return "BEFORE"
        elif VectorClock.happens_before(v2, v1):
            return "AFTER"
        else:
            return "CONCURRENT"

    def __repr__(self):
        labels = [f"{pid}:{val}" for pid, val in zip(self.process_ids, self.vector)]
        return f"VC({self.process_id}: [{', '.join(labels)}])"
```

### 4.6 인과 관계 감지

```python
def demonstrate_vector_clocks():
    """Reproduce the example from Section 4.4 and verify relationships."""
    pids = ["P1", "P2", "P3"]
    vc1 = VectorClock("P1", pids)
    vc2 = VectorClock("P2", pids)
    vc3 = VectorClock("P3", pids)

    # Event a: internal at P1
    a = vc1.tick()
    print(f"Event a at P1: {a}")

    # Send m1: P1 -> P2
    m1_ts = vc1.send_timestamp()
    # Since tick already happened, we use the current vector
    # Actually, let us redo: a is the send event itself
    vc1 = VectorClock("P1", pids)
    vc2 = VectorClock("P2", pids)
    vc3 = VectorClock("P3", pids)

    # Event a: P1 sends m1 to P2
    m1_ts = vc1.send_timestamp()
    a = m1_ts
    print(f"Event a (P1 sends m1): {a}")

    # Event b: P2 receives m1, then sends m2 to P3
    vc2.receive_timestamp(m1_ts)
    m2_ts = vc2.send_timestamp()
    b = m2_ts
    print(f"Event b (P2 receives m1, sends m2): {b}")

    # Event c: P3 receives m2
    c = vc3.receive_timestamp(m2_ts)
    print(f"Event c (P3 receives m2): {c}")

    # Event d: internal at P1
    d = vc1.tick()
    print(f"Event d (P1 internal): {d}")

    # Event e: P2 sends m3 to P1
    m3_ts = vc2.send_timestamp()
    e = m3_ts
    print(f"Event e (P2 sends m3): {e}")

    # Event f: P1 receives m3
    f = vc1.receive_timestamp(m3_ts)
    print(f"Event f (P1 receives m3): {f}")

    # Check relationships
    print(f"\nCausal relationships:")
    pairs = [("a", a, "b", b), ("d", d, "b", b), ("a", a, "c", c),
             ("d", d, "e", e), ("a", a, "e", e), ("e", e, "f", f)]
    for name1, v1, name2, v2 in pairs:
        rel = VectorClock.compare(v1, v2)
        print(f"  {name1} vs {name2}: {rel}")


demonstrate_vector_clocks()
```

### 4.7 확장성 문제

| 차원 | 영향 |
|------|------|
| 메시지당 공간 | O(n) 정수, n = 프로세스 수 |
| 이벤트당 공간 | O(n) 정수 저장 |
| 비교 비용 | 비교당 O(n) |
| 프로세스 추가 | 모든 벡터를 확장해야 함 (조율 필요) |
| 프로세스 제거 | 단순히 축소 불가 (향후 비교에 항목이 필요할 수 있음) |

수천 개의 프로세스가 있는 시스템에서는 메시지 크기 때문에 vector 클럭이 비실용적이 됩니다. 해결책:

- **Plausible 클럭**: 제한된 크기의 근사 vector 클럭
- **Matrix 클럭**: 지식에 대한 지식을 추적 (O(n^2)이지만 가비지 컬렉션 가능)
- **하이브리드 접근**: 소수의 복제본에 vector 클럭 사용, 더 거친 순서에 물리적 시간 사용

---

## 5. Version Vector vs Vector 클럭

이 두 개념은 자주 혼동되지만 서로 다른 목적을 가집니다.

### 5.1 핵심 차이

| 측면 | Vector 클럭 | Version Vector |
|------|-----------|---------------|
| 추적 대상 | 이벤트 (전송, 수신, 내부) | 데이터 항목의 업데이트 |
| 증가 시점 | 모든 이벤트 | 데이터 항목에 쓰기 시에만 |
| 목적 | 완전한 인과 이력 캡처 | 복제본 간 충돌 감지 |
| 크기 성장 | 프로세스당 하나의 항목 | 복제본당 하나의 항목 |
| 사용처 | 인과적 브로드캐스트, 디버깅 | 멀티마스터 복제 (Dynamo) |

### 5.2 Version Vector 예시

3개 노드에 복제된 키-값 저장소를 고려합니다:

```
Initial state: key "x" = null, version vector VV = [0, 0, 0]

Node A writes x = 1:
  VV_A = [1, 0, 0], value = 1

Node B reads from A, then writes x = 2:
  VV_B = [1, 1, 0], value = 2

Node C writes x = 3 (without reading A or B):
  VV_C = [0, 0, 1], value = 3

Now compare:
  VV_B = [1, 1, 0] vs VV_C = [0, 0, 1]
  1 > 0 but 0 < 1 → CONCURRENT → CONFLICT!

Resolution strategy:
  - Last-writer-wins (LWW): use physical timestamp to pick one
  - Application-level merge: return both values to the client
  - CRDTs: use conflict-free data structure (Lesson 10)
```

### 5.3 Dotted Version Vector

표준 version vector는 **거짓 충돌**(sibling explosion)을 생성할 수 있습니다. Dotted version vector(Preguica et al., 2012)는 각 sibling을 만든 정확한 이벤트(dot)를 추적하여 이를 해결합니다:

```python
@dataclass
class Dot:
    """A dot represents a specific write event: (node_id, counter)."""
    node_id: str
    counter: int


class DottedVersionVector:
    """
    Dotted version vector for accurate conflict detection.
    Avoids the sibling explosion problem of plain version vectors.
    """

    def __init__(self):
        self.version_vector: dict[str, int] = {}  # node_id -> max counter
        self.dot: Optional[Dot] = None             # the event that created this value

    def increment(self, node_id: str) -> 'DottedVersionVector':
        """Create a new version for a write at the given node."""
        new_dvv = DottedVersionVector()
        new_dvv.version_vector = dict(self.version_vector)
        counter = self.version_vector.get(node_id, 0) + 1
        new_dvv.version_vector[node_id] = counter
        new_dvv.dot = Dot(node_id, counter)
        return new_dvv

    def descends(self, other: 'DottedVersionVector') -> bool:
        """Check if self descends from (is causally after) other."""
        if other.dot is None:
            return True
        return self.version_vector.get(other.dot.node_id, 0) >= other.dot.counter

    def concurrent_with(self, other: 'DottedVersionVector') -> bool:
        """Check if self and other are concurrent (conflict)."""
        return not self.descends(other) and not other.descends(self)
```

---

## 6. Hybrid Logical Clock (HLC)

### 6.1 동기

Vector 클럭은 완벽한 인과성 추적을 제공하지만 물리적 시간과 연관되지 않습니다. Lamport 타임스탬프는 컴팩트하지만 동시성 정보를 잃습니다. **Hybrid Logical Clock** (Kulkarni et al., 2014)은 두 장점을 결합합니다:

- 제한된 크기 (상수, O(n) 아님)
- Happens-before 관계 캡처 (Lamport 클럭과 동일)
- 물리적 시간에 근접 유지 (순수 논리적 클럭과 달리)
- 물리적 시간에서의 스냅샷 쿼리에 사용 가능

### 6.2 HLC 구조

HLC 타임스탬프는 쌍 `(l, c)`이며:

- `l`: 지금까지 본 최대 물리적 시간 (물리적 시간을 추적하는 논리 컴포넌트)
- `c`: 물리적 클럭이 동일할 때 동점을 해결하기 위한 제한된 카운터

불변식은:

```
l ≥ pt (physical time)  — HLC is always ≥ physical time
l is bounded above by  pt + ε  where ε is the max clock skew
```

### 6.3 알고리즘

```
On local event or send at process j:
    l'_j = l_j                          # save old l
    l_j  = max(l'_j, pt_j)              # advance l to max of old l and physical time
    if l_j = l'_j:
        c_j = c_j + 1                   # same l, increment counter
    else:
        c_j = 0                         # new l, reset counter
    timestamp = (l_j, c_j, j)

On receive of message m with timestamp (l_m, c_m, _) at process j:
    l'_j = l_j                          # save old l
    l_j  = max(l'_j, l_m, pt_j)         # advance l to max of all three
    if l_j = l'_j = l_m:
        c_j = max(c_j, c_m) + 1         # all three equal, increment max counter
    elif l_j = l'_j:
        c_j = c_j + 1                   # l stayed same as local, increment own
    elif l_j = l_m:
        c_j = c_m + 1                   # l advanced to message's, continue message's counter
    else:
        c_j = 0                         # l advanced to physical time, reset
    timestamp = (l_j, c_j, j)
```

### 6.4 속성

| 속성 | 보장 |
|------|------|
| 클럭 조건 | `a → b ⟹ HLC(a) < HLC(b)` (Lamport과 동일) |
| 물리적 시간 근접성 | `l - pt ≤ ε` 여기서 ε는 최대 클럭 스큐 |
| 카운터 상한 | 최악의 경우 `c ≤ n × ε × event_rate` |
| 타임스탬프당 공간 | O(1) -- (l, c, process_id)만 |
| 비교 | (l, c, process_id)에 대한 사전식 순서 |

### 6.5 HLC vs Lamport vs Vector 클럭

```
                    Lamport    HLC        Vector
Space per timestamp O(1)       O(1)       O(n)
Detects causality?  one-way    one-way    both-ways
Physical time?      No         Yes        No
Snapshot queries?   No         Yes        No
Total order?        Yes*       Yes*       No
Suitable for n>>1?  Yes        Yes        No

* with process ID tie-breaking
```

### 6.6 구현

```python
import time


class HybridLogicalClock:
    """
    Hybrid Logical Clock (Kulkarni et al., 2014).
    Combines physical time awareness with logical clock properties.
    """

    def __init__(self, process_id: str, physical_clock=None):
        self.process_id = process_id
        self.l = 0   # logical component (tracks max physical time)
        self.c = 0   # counter for tie-breaking
        # Allow injecting a custom physical clock for testing
        self._physical_clock = physical_clock or (lambda: int(time.time() * 1000))

    def _pt(self) -> int:
        """Get current physical time in milliseconds."""
        return self._physical_clock()

    def now(self) -> tuple[int, int, str]:
        """
        Generate a timestamp for a local or send event.
        Returns (l, c, process_id).
        """
        pt = self._pt()
        old_l = self.l
        self.l = max(old_l, pt)

        if self.l == old_l:
            self.c += 1
        else:
            self.c = 0

        return (self.l, self.c, self.process_id)

    def receive(self, msg_l: int, msg_c: int) -> tuple[int, int, str]:
        """
        Generate a timestamp for a receive event.
        Takes the l and c from the received message.
        """
        pt = self._pt()
        old_l = self.l

        self.l = max(old_l, msg_l, pt)

        if self.l == old_l == msg_l:
            self.c = max(self.c, msg_c) + 1
        elif self.l == old_l:
            self.c = self.c + 1
        elif self.l == msg_l:
            self.c = msg_c + 1
        else:
            self.c = 0

        return (self.l, self.c, self.process_id)

    @staticmethod
    def compare(ts1: tuple[int, int, str], ts2: tuple[int, int, str]) -> int:
        """
        Compare two HLC timestamps.
        Returns: -1 if ts1 < ts2, 0 if equal, 1 if ts1 > ts2.
        """
        if ts1[0] != ts2[0]:
            return -1 if ts1[0] < ts2[0] else 1
        if ts1[1] != ts2[1]:
            return -1 if ts1[1] < ts2[1] else 1
        if ts1[2] != ts2[2]:
            return -1 if ts1[2] < ts2[2] else 1
        return 0

    def __repr__(self):
        return f"HLC({self.process_id}: l={self.l}, c={self.c})"
```

---

## 7. TrueTime과 구간 기반 클럭

### 7.1 Google Spanner의 접근법

Google Spanner(Corbett et al., 2012)는 근본적으로 다른 접근법을 취합니다: 물리적 시간을 포기하는 대신, 특수 하드웨어를 사용하여 물리적 시간의 **불확실성을 제한**합니다.

**TrueTime API**:

```
TT.now()    → TTinterval: [earliest, latest]
TT.after(t) → bool: true if t is definitely in the past
TT.before(t)→ bool: true if t is definitely in the future
```

핵심 보장: 실제 절대 시간 `t_abs`는 항상 반환된 구간 내에 있습니다:

```
earliest ≤ t_abs ≤ latest
uncertainty ε = (latest - earliest) / 2
```

### 7.2 TrueTime의 동작 방식

```
GPS Antenna ──► GPS Receiver ──► Time Server ──► Spanner Node
Atomic Clock ──► Cs/Rb Ref ─────►            ──►

Each data center has:
  - Multiple GPS receivers (for absolute time)
  - Multiple atomic clocks (for holdover during GPS outage)
  - Time servers that combine both sources

The uncertainty interval ε depends on:
  - Time since last GPS sync (~200 μs sawtooth)
  - Network delay to time server (~1 ms within data center)
  - Typical ε ≈ 1-7 ms
```

### 7.3 Commit-Wait 프로토콜

Spanner는 TrueTime을 사용하여 락킹 없이 **외부 일관성**(linearizable) 트랜잭션을 구현합니다:

```
Transaction commit protocol:
  1. Acquire locks (Paxos groups)
  2. Choose commit timestamp s = TT.now().latest
  3. WAIT until TT.after(s) is true        ← "commit wait"
  4. Release locks and apply

The commit wait ensures that:
  - s is definitely in the past when the transaction becomes visible
  - Any transaction that starts after this one will get a later timestamp
  - Therefore, the real-time order of transactions matches timestamp order
```

**비용**: Commit wait는 모든 트랜잭션에 `2ε`(불확실성의 두 배)만큼의 지연을 추가합니다. 일반적인 `ε ≈ 3.5 ms`에서, 이는 ~7 ms의 지연을 추가합니다. 이것이 Google이 더 나은 하드웨어를 통해 `ε`를 줄이는 데 대규모 투자를 하는 이유입니다.

### 7.4 구간 기반 순서화

```
Transaction T1: commit timestamp s1, uncertainty [s1 - ε, s1 + ε]
Transaction T2: commit timestamp s2, uncertainty [s2 - ε, s2 + ε]

If s1 + ε < s2 - ε:
  T1 definitely committed before T2 in real time → ordered

If intervals overlap:
  Cannot determine real-time order
  But commit-wait ensures this case does not arise for causally related transactions
```

### 7.5 TrueTime 시뮬레이션

```python
import random


class TrueTime:
    """
    Simulated TrueTime API.
    Models GPS + atomic clock time source with bounded uncertainty.
    """

    def __init__(self, epsilon_ms: float = 5.0):
        self.epsilon_ms = epsilon_ms  # half-width of uncertainty interval
        self._real_offset = random.uniform(-2, 2)  # simulated clock offset

    def now(self) -> dict:
        """
        Return a time interval [earliest, latest] guaranteed
        to contain the true absolute time.
        """
        real_time = time.time() * 1000  # ms
        local_time = real_time + self._real_offset

        # Add random jitter to simulate varying uncertainty
        jitter = random.uniform(0, self.epsilon_ms * 0.5)
        epsilon = self.epsilon_ms + jitter

        return {
            "earliest": local_time - epsilon,
            "latest": local_time + epsilon,
            "epsilon": epsilon,
        }

    def after(self, t: float) -> bool:
        """Return True if t is definitely in the past."""
        interval = self.now()
        return interval["earliest"] > t

    def before(self, t: float) -> bool:
        """Return True if t is definitely in the future."""
        interval = self.now()
        return interval["latest"] < t


class SpannerCommit:
    """Simulated Spanner-style commit with TrueTime."""

    def __init__(self, true_time: TrueTime):
        self.tt = true_time

    def commit(self, transaction_id: str) -> float:
        """
        Commit a transaction using the commit-wait protocol.
        Returns the commit timestamp.
        """
        # Step 1: Choose commit timestamp as latest bound
        interval = self.tt.now()
        commit_ts = interval["latest"]
        print(f"[{transaction_id}] Commit timestamp: {commit_ts:.3f}")
        print(f"[{transaction_id}] Uncertainty: ±{interval['epsilon']:.3f} ms")

        # Step 2: Wait until commit timestamp is definitely in the past
        wait_start = time.time() * 1000
        while not self.tt.after(commit_ts):
            time.sleep(0.001)  # 1ms polling
        wait_end = time.time() * 1000

        print(f"[{transaction_id}] Commit-wait duration: {wait_end - wait_start:.3f} ms")
        return commit_ts


def demonstrate_truetime():
    """Show how TrueTime enables external consistency."""
    tt = TrueTime(epsilon_ms=5.0)
    spanner = SpannerCommit(tt)

    print("Demonstrating TrueTime commit-wait protocol:\n")

    ts1 = spanner.commit("TX_001")
    print()
    ts2 = spanner.commit("TX_002")

    print(f"\nTS ordering: TX_001({ts1:.3f}) < TX_002({ts2:.3f}) = {ts1 < ts2}")
    print("This ordering is guaranteed to match real-time order")
    print("because each commit waits out the uncertainty interval.")
```

---

## 8. 클럭 메커니즘 비교

### 8.1 기능 비교

| 기능 | Physical | Lamport | Vector | HLC | TrueTime |
|------|----------|---------|--------|-----|----------|
| 크기 | O(1) | O(1) | O(n) | O(1) | O(1) |
| a→b ⟹ C(a)<C(b) | 아니오 | 예 | 예 | 예 | 예 |
| C(a)<C(b) ⟹ a→b | 아니오 | 아니오 | 예 | 아니오 | 아니오 |
| 동시성 감지 | 아니오 | 아니오 | 예 | 아니오 | 아니오 |
| 물리적 시간 | 예 | 아니오 | 아니오 | 예 | 예 (제한됨) |
| 스냅샷 쿼리 | 예* | 아니오 | 아니오 | 예 | 예 |
| 외부 일관성 | 아니오 | 아니오 | 아니오 | 아니오 | 예 |
| 필요 하드웨어 | 기본 | 없음 | 없음 | 기본 | GPS+원자 |
| n>100에서 실용적 | 예 | 예 | 아니오 | 예 | 예 |

`*` 물리적 시간 스냅샷은 클럭 스큐로 인해 신뢰 불가.

### 8.2 언제 무엇을 사용할 것인가

```
Decision tree:

  Need external consistency (linearizability with real-time)?
  ├── Yes → TrueTime (if you have the hardware) or HLC + bounded skew
  └── No
       Need to detect concurrent updates?
       ├── Yes → Vector Clocks (if n is small, < ~20 replicas)
       │         or Version Vectors (for per-key conflict detection)
       └── No
            Need causal ordering with physical time correlation?
            ├── Yes → HLC
            └── No → Lamport timestamps (simplest)
```

### 8.3 실제 시스템에서의 사용

| 시스템 | 클럭 메커니즘 | 이유 |
|--------|-------------|------|
| Amazon DynamoDB | Version vector | 복제본 간 쓰기 충돌 감지 |
| Google Spanner | TrueTime | 분산 락킹 없이 외부 일관성 |
| CockroachDB | HLC | GPS 하드웨어 없이 Spanner와 유사한 시맨틱 |
| Apache Kafka | Lamport 유사 | 파티션 내 단조 오프셋 순서 |
| Riak | Dotted version vector | 정확한 충돌 감지 |
| MongoDB | HLC (3.6부터) | 인과적 일관성 세션 |
| etcd | Raft 로그 인덱스 | 합의에서의 전순서 (암묵적 Lamport) |

---

## 9. 코드: 전체 구현

### 9.1 포괄적 클럭 비교

```python
"""
Comprehensive comparison of Lamport, Vector, and Hybrid Logical Clocks.
Simulates the same set of events and compares the results.
"""


def run_clock_comparison():
    """
    Simulate a scenario with 3 processes and compare
    Lamport, Vector, and HLC timestamps.
    """
    process_ids = ["A", "B", "C"]

    # Initialize all three clock types for each process
    lamport = {pid: LamportClock(pid) for pid in process_ids}
    vector = {pid: VectorClock(pid, process_ids) for pid in process_ids}

    # For HLC, use controllable physical clocks
    physical_times = {pid: [100] for pid in process_ids}  # mutable list for closure

    def make_clock(pid):
        return lambda: physical_times[pid][0]

    hlc = {pid: HybridLogicalClock(pid, make_clock(pid)) for pid in process_ids}

    events = {}

    # Event 1: A does internal event at physical time 100
    physical_times["A"][0] = 100
    events["e1"] = {
        "lamport": lamport["A"].tick(),
        "vector": vector["A"].tick(),
        "hlc": hlc["A"].now(),
        "desc": "A internal event"
    }

    # Event 2: A sends to B at physical time 105
    physical_times["A"][0] = 105
    l_ts = lamport["A"].send_timestamp()
    v_ts = vector["A"].send_timestamp()
    h_ts = hlc["A"].now()
    events["e2_send"] = {
        "lamport": l_ts,
        "vector": v_ts,
        "hlc": h_ts,
        "desc": "A sends m1 to B"
    }

    # Event 3: B receives from A at physical time 110
    physical_times["B"][0] = 110
    events["e3_recv"] = {
        "lamport": lamport["B"].receive_timestamp(l_ts),
        "vector": vector["B"].receive_timestamp(v_ts),
        "hlc": hlc["B"].receive(h_ts[0], h_ts[1]),
        "desc": "B receives m1 from A"
    }

    # Event 4: C does internal event at physical time 108 (concurrent with e3)
    physical_times["C"][0] = 108
    events["e4"] = {
        "lamport": lamport["C"].tick(),
        "vector": vector["C"].tick(),
        "hlc": hlc["C"].now(),
        "desc": "C internal event (concurrent with B)"
    }

    # Event 5: B sends to C at physical time 115
    physical_times["B"][0] = 115
    l_ts2 = lamport["B"].send_timestamp()
    v_ts2 = vector["B"].send_timestamp()
    h_ts2 = hlc["B"].now()
    events["e5_send"] = {
        "lamport": l_ts2,
        "vector": v_ts2,
        "hlc": h_ts2,
        "desc": "B sends m2 to C"
    }

    # Event 6: C receives from B at physical time 120
    physical_times["C"][0] = 120
    events["e6_recv"] = {
        "lamport": lamport["C"].receive_timestamp(l_ts2),
        "vector": vector["C"].receive_timestamp(v_ts2),
        "hlc": hlc["C"].receive(h_ts2[0], h_ts2[1]),
        "desc": "C receives m2 from B"
    }

    # Print all events
    print("="*80)
    print("CLOCK COMPARISON")
    print("="*80)
    print(f"{'Event':<12} {'Description':<30} {'Lamport':>8} {'Vector':<15} {'HLC':<20}")
    print("-"*80)

    for name, data in events.items():
        hlc_str = f"({data['hlc'][0]},{data['hlc'][1]})"
        vec_str = str(data['vector'])
        print(f"{name:<12} {data['desc']:<30} {data['lamport']:>8} {vec_str:<15} {hlc_str:<20}")

    # Check causal relationships
    print("\n" + "="*80)
    print("CAUSAL RELATIONSHIP ANALYSIS")
    print("="*80)

    pairs_to_check = [
        ("e2_send", "e3_recv", "A sends → B receives (causal)"),
        ("e2_send", "e4",      "A sends vs C event (should be concurrent)"),
        ("e3_recv", "e4",      "B receives vs C event (should be concurrent)"),
        ("e4",      "e6_recv", "C event vs C receives (causal via process order + msg)"),
    ]

    for name1, name2, description in pairs_to_check:
        v1 = events[name1]["vector"]
        v2 = events[name2]["vector"]
        vc_rel = VectorClock.compare(v1, v2)

        l1 = events[name1]["lamport"]
        l2 = events[name2]["lamport"]
        lamport_rel = "BEFORE" if l1 < l2 else ("AFTER" if l1 > l2 else "EQUAL")

        print(f"\n{description}")
        print(f"  Vector clock says: {vc_rel}")
        print(f"  Lamport says:      {lamport_rel} ({l1} vs {l2})")

        if vc_rel == "CONCURRENT" and lamport_rel == "BEFORE":
            print(f"  NOTE: Lamport incorrectly implies ordering for concurrent events!")


run_clock_comparison()
```

### 9.2 Vector 클럭을 이용한 인과적 브로드캐스트

```python
"""
Causal broadcast: deliver messages in causal order using vector clocks.

A causally ordered broadcast ensures that if message m1 causally
precedes message m2 (m1 → m2), then every process delivers m1 before m2.
"""

from collections import deque


class CausalBroadcast:
    """
    Causal broadcast protocol using vector clocks.
    Messages are buffered until all causally preceding messages have been delivered.
    """

    def __init__(self, process_id: str, all_ids: list[str]):
        self.process_id = process_id
        self.all_ids = sorted(all_ids)
        self.index = self.all_ids.index(process_id)
        self.n = len(all_ids)

        # Vector clock tracking deliveries
        self.vc = [0] * self.n

        # Buffer for messages waiting to be delivered
        self.pending: deque[tuple[list[int], str, dict]] = deque()

        # Delivered messages (for inspection)
        self.delivered: list[tuple[str, dict, list[int]]] = []

    def broadcast(self, content: dict) -> tuple[list[int], dict]:
        """
        Broadcast a message to all processes.
        Returns the (vector_clock, content) pair that should be sent.
        """
        # Increment own component
        self.vc[self.index] += 1
        timestamp = list(self.vc)

        # Deliver to self immediately
        self.delivered.append((self.process_id, content, timestamp))

        return (timestamp, content)

    def receive(self, sender: str, timestamp: list[int], content: dict):
        """
        Receive a broadcast message. Buffer it until causal dependencies are met.
        """
        self.pending.append((timestamp, sender, content))
        self._try_deliver()

    def _can_deliver(self, sender: str, timestamp: list[int]) -> bool:
        """
        Check if a message can be delivered (all causal dependencies met).

        A message from process j with timestamp V can be delivered at process i if:
          V[j] = vc_i[j] + 1     (it is the next expected message from j)
          V[k] <= vc_i[k]        for all k != j (we have seen all messages that j saw)
        """
        j = self.all_ids.index(sender)
        if timestamp[j] != self.vc[j] + 1:
            return False
        for k in range(self.n):
            if k != j and timestamp[k] > self.vc[k]:
                return False
        return True

    def _try_deliver(self):
        """Try to deliver buffered messages in causal order."""
        delivered_any = True
        while delivered_any:
            delivered_any = False
            new_pending = deque()
            for timestamp, sender, content in self.pending:
                if self._can_deliver(sender, timestamp):
                    j = self.all_ids.index(sender)
                    # Update vector clock
                    for k in range(self.n):
                        self.vc[k] = max(self.vc[k], timestamp[k])
                    self.delivered.append((sender, content, timestamp))
                    delivered_any = True
                    print(f"  [{self.process_id}] Delivered from {sender}: "
                          f"{content} (vc={timestamp})")
                else:
                    new_pending.append((timestamp, sender, content))
            self.pending = new_pending

    def pending_count(self) -> int:
        return len(self.pending)


def demonstrate_causal_broadcast():
    """Show how causal broadcast reorders messages."""
    ids = ["P1", "P2", "P3"]
    nodes = {pid: CausalBroadcast(pid, ids) for pid in ids}

    print("Causal Broadcast Demonstration")
    print("="*50)

    # P1 broadcasts m1
    print("\nP1 broadcasts m1:")
    ts1, content1 = nodes["P1"].broadcast({"msg": "m1", "data": "hello"})
    print(f"  Timestamp: {ts1}")

    # P2 receives m1 and broadcasts m2 (causally after m1)
    print("\nP2 receives m1:")
    nodes["P2"].receive("P1", ts1, content1)

    print("\nP2 broadcasts m2 (causally after m1):")
    ts2, content2 = nodes["P2"].broadcast({"msg": "m2", "data": "reply"})
    print(f"  Timestamp: {ts2}")

    # P3 receives m2 BEFORE m1 (out of causal order)
    print("\nP3 receives m2 before m1 (out of order):")
    nodes["P3"].receive("P2", ts2, content2)
    print(f"  Pending at P3: {nodes['P3'].pending_count()} messages buffered")

    # P3 receives m1 (now m2 can be delivered too)
    print("\nP3 receives m1 (causally enables m2 delivery):")
    nodes["P3"].receive("P1", ts1, content1)

    print(f"\nFinal delivery order at P3:")
    for sender, content, ts in nodes["P3"].delivered:
        print(f"  {content['msg']} from {sender} (vc={ts})")


demonstrate_causal_broadcast()
```

### 9.3 가비지 컬렉션을 위한 Matrix 클럭

```python
class MatrixClock:
    """
    Matrix clock: each process tracks its knowledge of every other
    process's knowledge. This enables garbage collection of old
    vector clock entries.

    Matrix[i][j] = process i's knowledge of process j's vector clock entry for j.
    In other words, "what process i knows about what process j knows."
    """

    def __init__(self, process_id: str, all_ids: list[str]):
        self.process_id = process_id
        self.all_ids = sorted(all_ids)
        self.index = self.all_ids.index(process_id)
        self.n = len(all_ids)
        # matrix[i][j] = our knowledge of what process i knows
        #                 about process j's progress
        self.matrix = [[0] * self.n for _ in range(self.n)]

    def tick(self):
        """Internal event."""
        self.matrix[self.index][self.index] += 1

    def send_timestamp(self) -> list[list[int]]:
        """Get matrix timestamp for outgoing message."""
        self.matrix[self.index][self.index] += 1
        return [row[:] for row in self.matrix]

    def receive_timestamp(self, sender_id: str, msg_matrix: list[list[int]]):
        """Update matrix upon receiving a message."""
        j = self.all_ids.index(sender_id)

        # Update our knowledge based on what the sender knows
        for k in range(self.n):
            for l in range(self.n):
                self.matrix[k][l] = max(self.matrix[k][l], msg_matrix[k][l])

        # Update sender's row with the sender's latest knowledge
        for l in range(self.n):
            self.matrix[j][l] = max(self.matrix[j][l], msg_matrix[j][l])

        # Increment own counter
        self.matrix[self.index][self.index] += 1

    def min_known_by_all(self) -> list[int]:
        """
        For each process j, compute the minimum value across all rows for column j.
        This tells us the minimum progress of j that ALL processes are aware of.
        Entries older than this can be safely garbage collected.
        """
        result = []
        for j in range(self.n):
            min_val = min(self.matrix[i][j] for i in range(self.n))
            result.append(min_val)
        return result

    def can_garbage_collect(self, event_vector: list[int]) -> bool:
        """
        Check if an event with the given vector clock can be garbage collected.
        An event can be GC'd if all processes know about all events
        that causally precede it.
        """
        min_known = self.min_known_by_all()
        return all(event_vector[j] <= min_known[j] for j in range(self.n))
```

---

## 10. 연습 문제

### 연습 1: Lamport 타임스탬프 계산

다음 이벤트 추적이 주어졌을 때, 각 이벤트의 Lamport 타임스탬프를 계산하세요:

```
Process A: a1(send to B), a2(internal), a3(receive from C)
Process B: b1(receive from A), b2(send to C), b3(internal)
Process C: c1(internal), c2(receive from B), c3(send to A)
```

### 연습 2: Vector 클럭 계산

연습 1과 같은 이벤트 추적에 대해, 각 이벤트의 vector 클럭을 계산하세요. 그런 다음 어떤 이벤트 쌍이 동시적인지 결정하세요.

### 연습 3: 클럭 스큐 하에서의 HLC 동작

프로세스 A는 10ms 빠른 물리적 클럭을 가지고 있습니다. 프로세스 B는 5ms 느린 물리적 클럭을 가지고 있습니다. 실제 시간 `t = 1000`에서:

- A의 물리적 클럭은 1010을 읽음
- B의 물리적 클럭은 995를 읽음

A가 B에 메시지를 보내고 실제 시간 `t = 1002`에 도착합니다 (B의 클럭은 997을 읽음).

1. A가 전송 이벤트에 할당하는 HLC 타임스탬프는?
2. B가 수신 이벤트에 할당하는 HLC 타임스탬프는?
3. HLC 순서가 happens-before 관계와 일관적인가?

### 연습 4: Version Vector 충돌 감지

세 복제본(R1, R2, R3)이 키 "user:profile"을 저장합니다. 다음 쓰기가 발생합니다:

1. R1이 값 V1을 쓰고, version vector가 [1, 0, 0]이 됨
2. R2가 R1에서 읽은 후, V2를 씀: version vector [1, 1, 0]
3. R3가 다른 복제본을 읽지 않고 V3를 씀: version vector [0, 0, 1]
4. 클라이언트가 R2에서 (V2, [1, 1, 0])와 R3에서 (V3, [0, 0, 1])를 읽음

충돌이 있는가? 어떻게 해결해야 하는가?

### 연습 5: 구현 도전

Vector 클럭을 사용하여 **비정상적** 메시지 순서 -- 메시지가 인과적 순서를 벗어나 전달되는 경우 -- 를 감지하는 메시지 로깅 시스템을 구현하세요. 시스템은:

1. 4개 프로세스에 걸친 모든 전송 및 수신 이벤트를 추적
2. 인과적으로 선행하는 메시지 전에 메시지가 전달되는 경우를 감지
3. 전체 vector 클럭 세부 정보와 함께 이상을 보고
4. 이상을 수정하기 위한 인과적 버퍼링을 구현

---

## 11. 요약

### 핵심 정리

1. **물리적 클럭은 신뢰할 수 없습니다** -- 드리프트, 스큐, NTP 한계, 빛의 속도 때문에 분산 시스템에서 이벤트 순서 결정에 부적합합니다.

2. **Happens-before (→)** 는 잠재적 인과성에 기반한 이벤트의 엄격한 부분순서를 정의합니다: 프로세스 순서와 메시지 전송-수신 쌍.

3. **Lamport 타임스탬프**는 `a → b ⟹ C(a) < C(b)`를 만족하는 컴팩트한(O(1)) 클럭을 제공하지만, 동시성을 감지할 수 없습니다.

4. **Vector 클럭**은 완전한 인과적 그림을 제공합니다: `a → b ⟺ V(a) < V(b)`, 하지만 타임스탬프당 O(n) 공간이 필요합니다.

5. **HLC**는 물리적 시간 인식과 Lamport 유사 속성을 O(1) 공간으로 결합하여, 대규모 시스템에 실용적입니다.

6. **TrueTime**은 하드웨어로 클럭 불확실성을 제한하여, commit-wait 프로토콜을 통한 외부 일관성을 가능하게 합니다.

7. 클럭 메커니즘의 선택은 일관성 요구 사항, 시스템 규모, 사용 가능한 하드웨어에 따라 달라집니다.

### 클럭 메커니즘 결정 프레임워크

```
                          Need external consistency?
                         /                          \
                       Yes                           No
                        |                             |
              Have GPS/atomic hw?             Need concurrency detection?
              /                  \              /                      \
            Yes                  No           Yes                      No
             |                    |             |                       |
         TrueTime          HLC + bounded     Vector Clocks          Lamport or HLC
                           clock sync        (small n)
```

---

## 12. 참고 문헌

1. Lamport, L. (1978). "Time, Clocks, and the Ordering of Events in a Distributed System." *Communications of the ACM*, 21(7), 558-565.
2. Fidge, C. J. (1988). "Timestamps in Message-Passing Systems That Preserve the Partial Ordering." *Proceedings of the 11th Australian Computer Science Conference*.
3. Mattern, F. (1989). "Virtual Time and Global States of Distributed Systems." *Parallel and Distributed Algorithms*, 215-226.
4. Kulkarni, S., Demirbas, M., et al. (2014). "Logical Physical Clocks and Consistent Snapshots in Globally Distributed Databases." *OPODIS 2014*.
5. Corbett, J. C., et al. (2012). "Spanner: Google's Globally-Distributed Database." *OSDI 2012*, 261-264.
6. Preguica, N., Baquero, C., et al. (2012). "Brief Announcement: Efficient Causality Tracking in Distributed Storage Systems with Dotted Version Vectors." *PODC 2012*.
7. Schwarz, R. & Mattern, F. (1994). "Detecting Causal Relationships in Distributed Computations." *Distributed Computing*, 7(3), 149-174.
8. Mills, D. L. (2006). *Computer Network Time Synchronization: The Network Time Protocol*. CRC Press.

---

[다음: 레슨 03 - FLP 불가능성과 이론적 한계](./03_FLP_Impossibility_and_Bounds.md)

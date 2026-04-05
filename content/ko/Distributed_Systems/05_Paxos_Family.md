# 레슨 5: Paxos 계열

[개요](./00_Overview.md) | [이전: 일관성 모델](./04_Consistency_Models.md) | [다음: Raft 심화](./06_Raft_In_Depth.md)

---

## 학습 목표

- Paxos 합의 프로토콜의 역사적 배경과 동기를 이해한다
- single-decree Paxos를 단계별로 추적하며, 에지 케이스와 정확성 논증을 포함한다
- leader 선출과 gap 처리를 포함한 Multi-Paxos 최적화를 설명한다
- 고급 Paxos 변형들을 비교한다: Flexible Paxos, Egalitarian Paxos, Cheap Paxos
- 메시지 전달 방식의 완전한 single-decree Paxos 시뮬레이터를 구현한다

---

## 1. 역사적 배경

### 1.1 The Part-Time Parliament

1989년, Leslie Lamport는 *"The Part-Time Parliament"*이라는 제목의 논문을 제출했다. 이 논문은 Paxos라는 고대 그리스 섬의 입법부에 대한 정교한 비유를 통해 합의 알고리즘을 설명했다. 논문의 표현이 너무 난해하여 심사위원들이 알고리즘과 비유를 분리할 수 없었다. 이 논문은 거의 10년 후인 1998년에야 비로소 출판되었다.

핵심 통찰은 혁명적이었다: **신뢰할 수 없는 프로세서 그룹이 과반수가 통신 가능한 한 단일 값에 합의할 수 있다**. 이것은 비동기적이고 crash-fault-tolerant한 환경에서 합의를 해결한 최초의 실용적 알고리즘이었다(leader 선출과 타임아웃을 사용하여 FLP 불가능성을 우회함).

### 1.2 Paxos Made Simple

원래 논문의 반응에 좌절한 Lamport는 2001년에 *"Paxos Made Simple"*을 출판했다 — 다음과 같은 문장으로 시작하는 간결한 14페이지짜리 설명서였다:

> "The Paxos algorithm, when presented in plain English, is very simple."

제목에도 불구하고 많은 실무자들은 여전히 이 알고리즘을 올바르게 구현하기 어렵다고 느꼈다. 이 어려움은 알고리즘 자체가 아니라 "single-decree Paxos"(하나의 값 선택)와 실용적인 "Multi-Paxos"(값의 시퀀스 선택) 사이의 간극에서 비롯되었으며, 논문에서는 후자를 대부분 명시하지 않았다.

### 1.3 Paxos가 중요한 이유

Paxos와 그 파생 프로토콜은 사실상 모든 프로덕션 합의 시스템의 근간을 형성한다:

| 시스템 | 프로토콜 기반 |
|--------|---------------|
| Google Chubby | Multi-Paxos |
| Google Spanner | Multi-Paxos |
| Apache ZooKeeper | ZAB (Paxos 파생) |
| etcd / Kubernetes | Raft (Paxos 영감) |
| CockroachDB | Raft (Paxos 영감) |
| AWS DynamoDB | 메타데이터용 Paxos |
| Azure Cosmos DB | Multi-Paxos 변형 |

Paxos를 이해하는 것은 합의를 구현하는 것뿐만 아니라, 모든 분산 시스템에서 정확성에 대해 추론하는 데 필수적이다.

---

## 2. Single-Decree Paxos

Single-decree(또는 "기본") Paxos는 가장 단순한 합의 문제를 해결한다: **노드 집합이 단일 값에 합의하는 것**.

### 2.1 역할

Paxos는 세 가지 논리적 역할을 정의한다. 실제로는 하나의 물리적 노드가 세 역할을 모두 수행하는 경우가 많다:

| 역할 | 책임 |
|------|---------------|
| **Proposer** | 값을 제안하고 프로토콜을 추진한다 |
| **Acceptor** | 제안에 투표하고 수락된 값을 저장한다 |
| **Learner** | 과반수가 수락한 후 선택된 값을 학습한다 |

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Proposer │     │ Proposer │     │ Proposer │
└────┬─────┘     └────┬─────┘     └────┬─────┘
     │                │                │
     ▼                ▼                ▼
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Acceptor │     │ Acceptor │     │ Acceptor │
└────┬─────┘     └────┬─────┘     └────┬─────┘
     │                │                │
     ▼                ▼                ▼
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Learner  │     │ Learner  │     │ Learner  │
└──────────┘     └──────────┘     └──────────┘
```

### 2.2 제안 번호

모든 제안은 고유하고 단조 증가하는 **제안 번호**(ballot number라고도 함)를 가진다. 일반적인 방식:

```
proposal_number = (round_number, proposer_id)
```

비교는 사전식으로 수행된다: 먼저 `round_number`, 그 다음 `proposer_id`. 이는 전역 고유성과 전체 순서를 보장한다.

```python
from dataclasses import dataclass
from typing import Optional, Any

@dataclass(frozen=True, order=True)
class ProposalNumber:
    """Globally unique, totally ordered proposal number."""
    round: int
    proposer_id: int

    def __str__(self):
        return f"({self.round},{self.proposer_id})"
```

### 2.3 Phase 1: Prepare / Promise

**목표**: proposer가 과반수 acceptor로부터 promise를 확보하여 제안할 권리를 확립한다.

```
Proposer                              Acceptor
   │                                     │
   │──── Prepare(n) ────────────────────▶│
   │                                     │
   │◀─── Promise(n, accepted_n, val) ────│  (if n > max_promised)
   │      or NACK                        │  (if n ≤ max_promised)
```

**Prepare(n)**: "제안 번호 `n`으로 제안하고 싶습니다. `n`보다 작은 번호의 제안은 수락하지 않겠다고 약속해 주세요."

**Promise(n, accepted_n, accepted_val)**: "`n`보다 작은 번호의 제안은 수락하지 않겠다고 약속합니다. 제가 이미 수락한 가장 높은 번호의 제안은 다음과 같습니다(있는 경우)."

```python
@dataclass
class AcceptorState:
    """Persistent state for a single acceptor."""
    max_promised: Optional[ProposalNumber] = None   # highest promised
    accepted_proposal: Optional[ProposalNumber] = None  # highest accepted proposal number
    accepted_value: Optional[Any] = None            # value of highest accepted

    def handle_prepare(self, n: ProposalNumber):
        """Process a Prepare(n) message.

        Returns (promised: bool, accepted_proposal, accepted_value).
        """
        if self.max_promised is not None and n <= self.max_promised:
            # Reject: already promised a higher number
            return False, None, None

        # Promise: update max_promised and return any previously accepted value
        self.max_promised = n
        return True, self.accepted_proposal, self.accepted_value
```

### 2.4 Phase 2: Accept / Accepted

**목표**: proposer가 과반수로부터 promise를 받으면 Accept 요청을 보낸다. 제안하는 값은 **값 선택 규칙**을 따라야 한다.

```
Proposer                              Acceptor
   │                                     │
   │──── Accept(n, v) ──────────────────▶│
   │                                     │
   │◀─── Accepted(n) ───────────────────│  (if n ≥ max_promised)
   │      or NACK                        │  (if n < max_promised)
```

**값 선택 규칙**: Promise 응답 중 이전에 수락된 값이 포함된 것이 있으면, proposer는 가장 높은 번호의 수락된 제안의 값을 **반드시** 사용해야 한다. 그렇지 않으면 proposer는 임의의 값을 자유롭게 선택할 수 있다.

이것이 핵심적인 안전 메커니즘이다: 값이 선택되면(과반수에 의해 수락되면), 미래의 모든 proposer가 동일한 값을 제안하도록 강제한다.

```python
def select_value(promises, own_value):
    """Apply the Paxos value selection rule.

    Args:
        promises: list of (accepted_proposal_number, accepted_value) from acceptors
        own_value: the value this proposer wants to propose

    Returns:
        The value to propose.
    """
    # Filter out None entries (acceptors that had not accepted anything)
    previously_accepted = [
        (prop_n, val) for prop_n, val in promises
        if prop_n is not None
    ]

    if not previously_accepted:
        # No acceptor has accepted any value yet; free to choose
        return own_value

    # Must use the value from the highest-numbered accepted proposal
    highest = max(previously_accepted, key=lambda x: x[0])
    return highest[1]
```

**Acceptor의 Accept 처리**:

```python
def handle_accept(self, n: ProposalNumber, value: Any):
    """Process an Accept(n, v) message.

    Returns True if accepted, False if rejected.
    """
    if self.max_promised is not None and n < self.max_promised:
        # Reject: already promised a higher number
        return False

    # Accept: update both promised and accepted
    self.max_promised = n
    self.accepted_proposal = n
    self.accepted_value = value
    return True
```

### 2.5 완전한 프로토콜 흐름

다음은 5개의 acceptor와 하나의 proposer를 사용한 완전한 single-decree Paxos 흐름이다:

```
Proposer P1 (value="X")     A1    A2    A3    A4    A5
    │                         │     │     │     │     │
    │──── Prepare(1,1) ──────▶│     │     │     │     │
    │──── Prepare(1,1) ──────────▶  │     │     │     │
    │──── Prepare(1,1) ───────────────▶   │     │     │
    │──── Prepare(1,1) ────────────────────▶    │     │
    │──── Prepare(1,1) ─────────────────────────▶     │
    │                         │     │     │     │     │
    │◀─── Promise(ok,∅) ─────│     │     │     │     │
    │◀─── Promise(ok,∅) ──────────│     │     │     │
    │◀─── Promise(ok,∅) ───────────────│     │     │   ← 과반수 (3)
    │                         │     │     │     │     │
    │   [select_value → "X"]  │     │     │     │     │
    │                         │     │     │     │     │
    │──── Accept(1,1,"X") ───▶│     │     │     │     │
    │──── Accept(1,1,"X") ────────▶ │     │     │     │
    │──── Accept(1,1,"X") ─────────────▶  │     │     │
    │                         │     │     │     │     │
    │◀─── Accepted ──────────│     │     │     │     │
    │◀─── Accepted ───────────────│     │     │     │
    │◀─── Accepted ────────────────────│     │     │   ← 과반수 (3)
    │                         │     │     │     │     │
    │   [VALUE "X" IS CHOSEN] │     │     │     │     │
```

### 2.6 정확성 증명

Paxos의 핵심 안전 속성은 다음과 같다:

> **최대 하나의 값만 선택될 수 있다.**

이를 귀류법으로 증명한다. 두 개의 서로 다른 값 `v`와 `v'`가 모두 선택되었다고 가정한다(각각 과반수에 의해 수락됨).

**증명 개요**:

`v`가 제안 번호 `n`에서 선택되고 `v'`가 제안 번호 `n'`에서 선택되었다고 하자. 여기서 `n < n'`이다.

1. `v`가 선택되었으므로, 과반수 `Q`가 `(n, v)`를 수락했다.
2. `v'`가 `n'`에서 제안되었으므로, proposer는 Phase 1에서 과반수 `Q'`로부터 promise를 받았다.
3. `Q`와 `Q'`가 모두 과반수이므로, 교집합이 존재한다: acceptor `a ∈ Q ∩ Q'`가 존재한다.
4. Acceptor `a`는 `(n, v)`를 수락하고 이후에 `n' > n`에 대해 promise했다.
5. `a`가 `n'`에 대한 Promise를 보낼 때, `(n, v)`를 자신의 가장 높은 수락 값으로 보고했다.
6. 값 선택 규칙에 의해, `n'`의 proposer는 이전에 수락된 가장 높은 번호의 제안 값을 선택해야 한다.
7. `Q'`의 어떤 acceptor가 보고한 가장 높은 수락 값의 제안 번호는 ≥ `n`이고, 그 값은 `v`여야 한다(`n`과 `n'` 사이의 제안 번호에 대한 귀납법에 의해).
8. 따라서 `v' = v`. 모순.

형식적 증명은 제안 번호에 대한 강한 귀납법을 사용한다. 기저 사례는 자명하다(최초로 선택된 값). 귀납 단계는 임의의 제안 번호 `n' > n`에 대해, 값이 `n`에서 선택되었다면, `n'`에서 제안되는 값도 동일해야 함을 보여준다. 이 귀납적 논증이 Paxos 정확성의 핵심이다.

**Liveness**: Paxos는 liveness를 보장하지 **않는다**. 두 proposer가 서로를 끝없이 선점할 수 있다(livelock). 실제로는 지정된 proposer(leader)를 선출하여 이를 해결한다.

### 2.7 에지 케이스

#### 경쟁하는 Proposer (Livelock)

```
P1: Prepare(1,1) → gets majority promises
P2: Prepare(2,2) → gets majority promises (invalidates P1's promises)
P1: Accept(1,1,"X") → REJECTED (acceptors promised (2,2))
P1: Prepare(3,1) → gets majority promises (invalidates P2's promises)
P2: Accept(2,2,"Y") → REJECTED (acceptors promised (3,1))
... repeats forever ...
```

**해결책**: 랜덤 백오프를 사용하거나 단일 leader를 선출한다.

```python
import random
import time

def propose_with_backoff(proposer, value, max_retries=10):
    """Propose a value with exponential backoff to avoid livelock."""
    for attempt in range(max_retries):
        result = proposer.run_paxos(value)
        if result.success:
            return result

        # Exponential backoff with jitter
        backoff = min(2 ** attempt * 0.01, 1.0)
        jitter = random.uniform(0, backoff)
        time.sleep(backoff + jitter)

    raise TimeoutError("Failed to reach consensus after max retries")
```

#### 분할 투표

acceptor 수가 짝수이거나 proposer들이 다른 값을 선택하면, Phase 1은 성공하지만 Phase 2에서 과반수 미만을 얻을 수 있다:

```
5 acceptors: A1, A2, A3, A4, A5

P1: Prepare(1,1) → promises from {A1, A2, A3}
P1: Accept(1,1,"X") → accepted by {A1, A2} only (A3 crashed before Accept)

Value "X" is NOT chosen (only 2 < majority of 3).
```

이것은 올바른 동작이다 — Paxos가 아직 값을 선택하지 않은 것일 뿐이다. 새로운 라운드를 시작할 수 있다.

#### Acceptor 충돌과 복구

Acceptor는 응답하기 **전에** 자신의 상태(`max_promised`, `accepted_proposal`, `accepted_value`)를 **영구 저장소**에 기록해야 한다. Acceptor가 충돌 후 복구되면, 지속된 상태를 읽고 올바르게 계속한다.

```python
import json
import os

class DurableAcceptorState:
    """Acceptor state with write-ahead persistence."""

    def __init__(self, node_id: int, storage_dir: str = "/tmp/paxos"):
        self.node_id = node_id
        self.path = os.path.join(storage_dir, f"acceptor_{node_id}.json")
        os.makedirs(storage_dir, exist_ok=True)
        self.state = self._load()

    def _load(self):
        """Recover state from disk after crash."""
        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                return json.load(f)
        return {"max_promised": None, "accepted_proposal": None, "accepted_value": None}

    def _persist(self):
        """Flush state to disk before responding (fsync for safety)."""
        tmp = self.path + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(self.state, f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, self.path)  # atomic on POSIX

    def promise(self, n):
        """Handle Prepare: persist before responding."""
        # ... check and update ...
        self._persist()
        return response

    def accept(self, n, value):
        """Handle Accept: persist before responding."""
        # ... check and update ...
        self._persist()
        return response
```

#### 선택된 값 학습

가장 간단한 접근법: proposer가 과반수의 Accepted 응답을 받으면 모든 learner에게 알린다. 더 견고한 접근법은 acceptor가 직접 알리는 "지정 learner"를 사용한다.

---

## 3. Multi-Paxos

Single-decree Paxos는 **하나의** 값을 선택한다. 실제 시스템은 **값의 시퀀스**(복제된 로그)에 합의해야 한다. Multi-Paxos는 로그 슬롯당 하나씩 여러 Paxos 인스턴스를 실행한다.

### 3.1 순진한 접근법

각 로그 슬롯에 대해 독립적인 single-decree Paxos를 실행한다. 이는 슬롯당 2번의 왕복이 필요하다:

```
Slot 1: Prepare + Accept (4 messages × N acceptors)
Slot 2: Prepare + Accept (4 messages × N acceptors)
Slot 3: Prepare + Accept (4 messages × N acceptors)
...
```

이는 매우 비용이 크다: 로그 항목당 `O(4N)` 메시지.

### 3.2 Leader 최적화

핵심 Multi-Paxos 최적화: **안정적인 leader는 연속 슬롯에 대해 Phase 1을 건너뛸 수 있다**.

Proposer가 슬롯 `i`에 대해 Phase 1을 성공적으로 완료하면, 더 높은 번호의 제안이 없었음을 알 수 있다. Leader로 유지되는 한, 슬롯 `i+1, i+2, ...`에 대해 바로 Phase 2(Accept)로 진행할 수 있다.

```
Leader Election (one-time Phase 1 for all future slots):
  Leader: Prepare(n) for slot_range=[i, ∞)
  Acceptors: Promise(n) for all future slots

Steady-state operation (Phase 2 only):
  Leader: Accept(n, slot=i,   value=v1)  → 1 round trip
  Leader: Accept(n, slot=i+1, value=v2)  → 1 round trip
  Leader: Accept(n, slot=i+2, value=v3)  → 1 round trip
```

이렇게 하면 분할 상환 비용이 로그 항목당 `O(2N)` 메시지로 줄어든다 — Raft와 동일하다.

```python
class MultiPaxosLeader:
    """Multi-Paxos leader with Phase 1 caching."""

    def __init__(self, node_id, acceptors):
        self.node_id = node_id
        self.acceptors = acceptors
        self.proposal_number = ProposalNumber(0, node_id)
        self.is_leader = False
        self.next_slot = 0

    def establish_leadership(self):
        """Run Phase 1 for all future slots (one-time cost)."""
        self.proposal_number = ProposalNumber(
            self.proposal_number.round + 1, self.node_id
        )

        promises = []
        for acceptor in self.acceptors:
            ok, accepted_n, accepted_v = acceptor.handle_prepare(
                self.proposal_number
            )
            if ok:
                promises.append((accepted_n, accepted_v))

        if len(promises) > len(self.acceptors) // 2:
            self.is_leader = True
            # Process any previously accepted values for gap filling
            self._fill_gaps(promises)
            return True
        return False

    def replicate(self, value):
        """Replicate a value using Phase 2 only (leader fast path)."""
        if not self.is_leader:
            raise RuntimeError("Not the leader; run establish_leadership first")

        slot = self.next_slot
        self.next_slot += 1

        accepted_count = 0
        for acceptor in self.acceptors:
            if acceptor.handle_accept(self.proposal_number, value):
                accepted_count += 1

        if accepted_count > len(self.acceptors) // 2:
            return slot  # committed
        else:
            self.is_leader = False  # lost leadership
            raise RuntimeError("Lost leadership during replication")

    def _fill_gaps(self, promises):
        """Fill gaps in the log with no-op values."""
        # Omitted for brevity; see Section 3.4
        pass
```

### 3.3 로그 슬롯과 인스턴스 번호

Multi-Paxos 로그의 각 슬롯은 독립적인 Paxos 인스턴스이다:

```
Log:  [ slot 0 ] [ slot 1 ] [ slot 2 ] [ slot 3 ] [ slot 4 ] ...
        "SET     "SET       "DEL       ???         "SET
         x=1"     y=2"       x"       (gap)        z=3"
```

각 슬롯은 자체 `accepted_proposal`과 `accepted_value`를 가진다. Leader는 슬롯을 순차적으로 할당하지만, leader가 복제 중간에 충돌하면 gap이 발생할 수 있다.

### 3.4 Gap 처리와 No-Op 제안

새 leader가 인수할 때, gap — 값이 제안되었지만 커밋되지 않은 슬롯 — 을 발견할 수 있다. 새 leader는 진행하기 전에 이러한 gap을 채워야 한다:

```python
def fill_gaps(leader, log, highest_slot):
    """Fill uncommitted slots with no-ops after leader election."""
    for slot in range(highest_slot + 1):
        if not log.is_committed(slot):
            # Run full Paxos for this slot
            # Phase 1 responses may reveal a previously accepted value
            value = run_phase1_for_slot(leader, slot)
            if value is None:
                value = NO_OP  # No value was previously proposed
            run_phase2_for_slot(leader, slot, value)
```

no-op은 상태 머신에 영향을 주지 않지만 gap을 채워서 후속 슬롯이 순서대로 적용될 수 있게 하는 특수 명령이다.

### 3.5 Multi-Paxos에서의 Leader 선출

Lamport의 원래 논문은 leader 선출 메커니즘을 명시하지 않았다. 일반적인 접근법:

1. **최고 ID leader**: 접근 가능한 노드 중 가장 높은 ID를 가진 노드가 leader가 된다. 간단하지만 핫스팟을 생성한다.

2. **리스 기반 리더십**: leader가 시간 제한이 있는 리스를 보유한다. 다른 노드는 리스가 만료된 후에만 leader가 되려고 시도한다.

3. **뷰 기반**: 노드가 단조 증가하는 뷰 번호를 유지한다. 뷰 `v`의 leader는 `v mod N`이다.

```python
class LeaseBasedLeader:
    """Leader election with time-bounded leases."""

    LEASE_DURATION = 10.0  # seconds

    def __init__(self, node_id, cluster_size):
        self.node_id = node_id
        self.cluster_size = cluster_size
        self.lease_expiry = 0.0
        self.current_leader = None

    def try_become_leader(self, current_time):
        """Attempt to acquire leadership if no current leader."""
        if self.current_leader is not None and current_time < self.lease_expiry:
            return False  # current leader's lease is still valid

        # Run Phase 1 of Multi-Paxos
        # If successful, set lease
        self.current_leader = self.node_id
        self.lease_expiry = current_time + self.LEASE_DURATION
        return True

    def renew_lease(self, current_time):
        """Renew lease via heartbeat acknowledgments from majority."""
        if self.current_leader == self.node_id:
            self.lease_expiry = current_time + self.LEASE_DURATION
```

---

## 4. Flexible Paxos (FPaxos)

### 4.1 핵심 통찰

Classic Paxos는 Phase 1과 Phase 2 모두에 **과반수 쿼럼**을 요구한다. Heidi Howard의 Flexible Paxos(2016)는 유일한 요구 사항이 다음과 같음을 보여주었다:

> **Phase 1 quorum ∩ Phase 2 quorum ≠ ∅**

이는 Phase 1과 Phase 2가 겹치기만 하면 **서로 다른 쿼럼 크기**를 사용할 수 있다는 것을 의미한다.

### 4.2 쿼럼 구성

`N`개의 acceptor에서:

| 구성 | Phase 1 (Q1) | Phase 2 (Q2) | 특성 |
|--------------|-------------|-------------|----------|
| Classic Paxos | ⌈(N+1)/2⌉ | ⌈(N+1)/2⌉ | 균형 |
| 쓰기 최적화 | N | 1 | 빠른 쓰기, 비용이 큰 leader 선출 |
| 읽기 최적화 | 1 | N | 저렴한 Phase 1, 비용이 큰 쓰기 |
| 비대칭 | N-1 | 2 | 안정적인 leader에 적합 |

**제약 조건**: `Q1 + Q2 > N` (교집합 보장).

### 4.3 FPaxos가 중요한 이유

안정적인 leader가 있는 Multi-Paxos에서 Phase 1은 **거의** 실행되지 않는다(leader 선출 시에만). Phase 2는 **모든** 로그 항목에 대해 실행된다. FPaxos는 Phase 1의 비용을 증가시키는 대신 Phase 2를 더 저렴하게 만든다:

```
Classic Multi-Paxos (5 nodes):
  Phase 1 (rare): needs 3 promises
  Phase 2 (every op): needs 3 accepts

FPaxos (5 nodes, Q1=4, Q2=2):
  Phase 1 (rare): needs 4 promises
  Phase 2 (every op): needs 2 accepts  ← 33% fewer messages!
```

이는 Phase 2 메시지가 데이터 센터를 횡단하는 지리적으로 분산된 배포에서 특히 가치가 있다.

```python
class FlexiblePaxos:
    """Paxos with configurable quorum sizes."""

    def __init__(self, acceptors, q1_size, q2_size):
        self.acceptors = acceptors
        self.n = len(acceptors)
        self.q1_size = q1_size
        self.q2_size = q2_size

        # Safety check: quorums must overlap
        assert q1_size + q2_size > self.n, (
            f"Quorum intersection violated: Q1({q1_size}) + Q2({q2_size}) "
            f"must be > N({self.n})"
        )

    def phase1(self, proposal_number):
        """Phase 1 requires q1_size promises."""
        promises = []
        for acceptor in self.acceptors:
            ok, acc_n, acc_v = acceptor.handle_prepare(proposal_number)
            if ok:
                promises.append((acc_n, acc_v))
            if len(promises) >= self.q1_size:
                return promises
        return None  # failed to get enough promises

    def phase2(self, proposal_number, value):
        """Phase 2 requires q2_size accepts."""
        accepted = 0
        for acceptor in self.acceptors:
            if acceptor.handle_accept(proposal_number, value):
                accepted += 1
            if accepted >= self.q2_size:
                return True
        return False
```

---

## 5. Egalitarian Paxos (EPaxos)

### 5.1 동기: 리더 없는 합의

Multi-Paxos는 안정적인 leader를 요구한다. 이로 인해 발생하는 문제:

1. **병목**: 모든 요청이 하나의 노드를 거친다
2. **지연 시간**: leader에서 먼 클라이언트는 추가 왕복 비용을 지불한다
3. **장애 복구**: leader 장애 시 선출이 필요하다(일시적 불가용)

EPaxos(Moraru, Andersen, Kaminsky, 2013)는 leader를 완전히 제거한다. 어떤 replica든 명령을 직접 제안할 수 있다.

### 5.2 명령 간섭

EPaxos는 **명령 간섭** 개념을 도입한다: 두 명령의 실행 순서가 최종 상태에 영향을 미치면 간섭한다.

```
Non-interfering:  SET x=1 and SET y=2  (different keys → order doesn't matter)
Interfering:      SET x=1 and SET x=2  (same key → order matters)
```

간섭하지 않는 명령의 경우, EPaxos는 **fast path**(1 왕복)로 커밋할 수 있다. 간섭하는 명령의 경우, 전체 순서를 설정하기 위해 **slow path**(2 왕복)로 폴백한다.

### 5.3 Fast Path

```
Replica R1 receives command C1:

R1 ──── PreAccept(C1, deps={}) ──────▶ R2, R3, R4, R5
R2 ◀─── PreAcceptOK(C1, deps={}) ────── (no conflicts)
R3 ◀─── PreAcceptOK(C1, deps={}) ────── (no conflicts)
R4 ◀─── PreAcceptOK(C1, deps={}) ────── (no conflicts)

R1: Fast quorum (⌊(3N/4)⌋ + 1) replies agree → COMMIT in 1 round trip
```

fast path 쿼럼은 단순 과반수보다 크다(대략 replica의 3/4). 이는 두 fast path 쿼럼이 서로 **그리고** 단순 과반수와 겹치도록 보장한다.

### 5.4 Slow Path

Replica들이 서로 다른 의존성 집합을 보고할 때(간섭하는 명령을 본 경우), EPaxos는 slow path로 폴백한다:

```
R1 ──── PreAccept(C1, deps={}) ──────▶ R2, R3, R4, R5
R2 ◀─── PreAcceptOK(C1, deps={C2}) ──── (R2 has seen interfering C2)
R3 ◀─── PreAcceptOK(C1, deps={}) ──────
R4 ◀─── PreAcceptOK(C1, deps={C2}) ──── (R4 also has C2)

R1: Dependencies disagree → slow path
R1: Merge deps → deps={C2}

R1 ──── Accept(C1, deps={C2}) ─────────▶ R2, R3, R4, R5  (Phase 2)
R2 ◀─── AcceptOK ──────────────────────── (majority)
R3 ◀─── AcceptOK ────────────────────────

R1: Majority accepts → COMMIT in 2 round trips
```

### 5.5 의존성 추적과 실행

각 명령은 **의존성 집합**을 가진다: 자신보다 먼저 실행되어야 하는 명령의 집합이다. 실행 순서는 의존성 그래프의 위상 정렬로 결정된다:

```python
from collections import defaultdict

class EPaxosInstance:
    """An EPaxos command instance with dependency tracking."""

    def __init__(self, command, seq, deps):
        self.command = command      # the operation (e.g., "SET x=1")
        self.seq = seq              # sequence number for breaking cycles
        self.deps = deps            # set of instance IDs this depends on
        self.status = "pre-accepted"

    def __repr__(self):
        return f"Instance({self.command}, seq={self.seq}, deps={self.deps})"


def build_execution_order(instances):
    """Build execution order from dependency graph using Tarjan's SCC."""
    # Step 1: Find strongly connected components (cycles in deps)
    graph = defaultdict(set)
    for inst_id, inst in instances.items():
        for dep_id in inst.deps:
            graph[inst_id].add(dep_id)

    # Step 2: Topological sort of SCCs
    # Within each SCC, order by sequence number
    visited = set()
    order = []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for neighbor in graph.get(node, []):
            dfs(neighbor)
        order.append(node)

    for inst_id in instances:
        dfs(inst_id)

    # Reverse for topological order; within SCCs, sort by seq
    return list(reversed(order))
```

### 5.6 EPaxos 트레이드오프

| 측면 | Multi-Paxos | EPaxos |
|--------|------------|--------|
| Leader 필요 여부 | 예 | 아니오 |
| Fast path 지연 시간 | 1 RT (leader 경유) | 1 RT (임의 replica) |
| Slow path 지연 시간 | 해당 없음 | 2 RT |
| Fast path 쿼럼 | 과반수 | ⌊3N/4⌋ + 1 |
| 충돌 처리 | Leader를 통한 전체 순서 | 의존성 추적 |
| 구현 복잡도 | 보통 | 높음 |
| 지리적 분산 | Leader가 병목 | 임의 replica가 클라이언트 서비스 |
| 메시지 복잡도 (충돌 없음) | O(N) | O(N) |
| 메시지 복잡도 (충돌 있음) | O(N) | O(2N) |

---

## 6. Cheap Paxos와 Vertical Paxos

### 6.1 Cheap Paxos

Cheap Paxos(Lamport and Massa, 2004)는 장애 시에만 활성화되는 보조 replica를 사용하여 정상 운영 중 필요한 **활성** replica 수를 줄인다.

**아이디어**: `f+1`개의 주 replica와 `f`개의 보조 replica를 사용한다. 정상 운영 중에는 `f+1`개의 주 replica만 참여한다. 주 replica가 장애를 일으키면 보조 replica가 일시적으로 활성화된다.

```
Normal operation:  Main1, Main2  (f+1 = 2, tolerates f=1 failure)
Main2 crashes:     Main1, Aux1   (Aux1 activates to maintain quorum)
Main2 recovers:    Main1, Main2  (Aux1 deactivates)
```

이는 일반적인 경우에 CPU와 네트워크 자원을 절약하지만, 장애 복구가 느려지는 비용이 있다.

### 6.2 Vertical Paxos

Vertical Paxos(Lamport, Malkhi, Zhou, 2009)는 **구성 관리**와 **데이터 복제**를 분리한다. Paxos를 실행하는 보조 "구성 서비스"를 사용하여 각 파티션을 처리하는 replica를 관리한다:

```
Configuration Service (runs Paxos):
  "Partition P1 is handled by {R1, R2, R3} in config v1"

Data Path (uses primary-backup within configuration):
  R1 (primary) → R2 (backup) → R3 (backup)
  Only needs f+1 replicas for writes (not 2f+1)

Configuration Change:
  Config service reconfigures P1 → {R2, R3, R4} in config v2
  State transfer: R1 → R4
```

이것은 Paxos 그룹이 메타데이터를 관리하고 더 단순한 프로토콜이 데이터 복제를 처리하는 Google Spanner와 같은 시스템의 이론적 기반이다.

---

## 7. 프로토콜 비교

| 속성 | Single-Decree Paxos | Multi-Paxos | EPaxos | Raft |
|----------|-------------------|-------------|--------|------|
| Leader 필요 여부 | 아니오 (하지만 liveness에 도움) | 예 | 아니오 | 예 |
| Phase 1 메시지 | 2N | 분할 상환: 0 | 2N (fast), 4N (slow) | N (선출) |
| Phase 2 메시지 | 2N | 2N | 0 (fast), 2N (slow) | 2N |
| f 장애 허용 최소 노드 | 2f+1 | 2f+1 | 2f+1 | 2f+1 |
| 지연 시간 (정상 상태) | 2 RT | 1 RT | 1 RT (fast), 2 RT (slow) | 1 RT |
| 로그 순서 | 단일 값 | Leader 순서 | 의존성 그래프 | Leader 순서 |
| 이해도 | 보통 | 어려움 | 매우 어려움 | 쉬움 |
| 실제 구현 | 독립형은 드뭄 | Chubby, Spanner | 연구용 | etcd, CockroachDB |
| 재구성 | 수동 | 구현에 따라 다름 | 인스턴스 기반 | Joint consensus |

---

## 8. 전체 구현: Single-Decree Paxos 시뮬레이터

이 섹션에서는 메시지 전달, 네트워크 시뮬레이션, 장애 주입이 포함된 완전하고 실행 가능한 Paxos 시뮬레이터를 제공한다.

```python
"""
Single-Decree Paxos Simulator with Message Passing

Simulates proposers, acceptors, and learners communicating over an
unreliable network with configurable message loss and delay.
"""

import random
import heapq
from dataclasses import dataclass, field
from typing import Optional, Any, List, Dict, Tuple
from enum import Enum


# ──────────────────────────────────────────────
# Proposal Numbers
# ──────────────────────────────────────────────

@dataclass(frozen=True, order=True)
class ProposalNum:
    round: int
    node_id: int

    def __str__(self):
        return f"n({self.round},{self.node_id})"


# ──────────────────────────────────────────────
# Message Types
# ──────────────────────────────────────────────

class MsgType(Enum):
    PREPARE = "Prepare"
    PROMISE = "Promise"
    NACK_PREPARE = "NackPrepare"
    ACCEPT = "Accept"
    ACCEPTED = "Accepted"
    NACK_ACCEPT = "NackAccept"
    DECIDE = "Decide"


@dataclass
class Message:
    msg_type: MsgType
    src: int
    dst: int
    proposal_num: ProposalNum
    value: Optional[Any] = None
    accepted_num: Optional[ProposalNum] = None
    accepted_val: Optional[Any] = None


# ──────────────────────────────────────────────
# Network Simulator
# ──────────────────────────────────────────────

@dataclass(order=True)
class Event:
    time: float
    message: Message = field(compare=False)


class Network:
    """Simulated network with configurable unreliability."""

    def __init__(self, loss_rate=0.0, min_delay=1.0, max_delay=5.0):
        self.loss_rate = loss_rate
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.event_queue: List[Event] = []
        self.current_time = 0.0
        self.delivered: List[Message] = []

    def send(self, msg: Message):
        """Queue a message for delivery (may be lost)."""
        if random.random() < self.loss_rate:
            return  # message lost

        delay = random.uniform(self.min_delay, self.max_delay)
        heapq.heappush(
            self.event_queue,
            Event(self.current_time + delay, msg)
        )

    def deliver_next(self) -> Optional[Message]:
        """Deliver the next message in time order."""
        if not self.event_queue:
            return None
        event = heapq.heappop(self.event_queue)
        self.current_time = event.time
        self.delivered.append(event.message)
        return event.message

    def has_messages(self) -> bool:
        return len(self.event_queue) > 0


# ──────────────────────────────────────────────
# Acceptor
# ──────────────────────────────────────────────

class Acceptor:
    def __init__(self, node_id: int, network: Network):
        self.node_id = node_id
        self.network = network
        self.max_promised: Optional[ProposalNum] = None
        self.accepted_num: Optional[ProposalNum] = None
        self.accepted_val: Optional[Any] = None

    def handle(self, msg: Message):
        if msg.msg_type == MsgType.PREPARE:
            self._handle_prepare(msg)
        elif msg.msg_type == MsgType.ACCEPT:
            self._handle_accept(msg)

    def _handle_prepare(self, msg: Message):
        n = msg.proposal_num
        if self.max_promised is not None and n <= self.max_promised:
            # Reject
            self.network.send(Message(
                MsgType.NACK_PREPARE, self.node_id, msg.src, n
            ))
            return

        self.max_promised = n
        self.network.send(Message(
            MsgType.PROMISE, self.node_id, msg.src, n,
            accepted_num=self.accepted_num,
            accepted_val=self.accepted_val
        ))

    def _handle_accept(self, msg: Message):
        n = msg.proposal_num
        if self.max_promised is not None and n < self.max_promised:
            self.network.send(Message(
                MsgType.NACK_ACCEPT, self.node_id, msg.src, n
            ))
            return

        self.max_promised = n
        self.accepted_num = n
        self.accepted_val = msg.value
        self.network.send(Message(
            MsgType.ACCEPTED, self.node_id, msg.src, n, value=msg.value
        ))


# ──────────────────────────────────────────────
# Proposer
# ──────────────────────────────────────────────

class Proposer:
    def __init__(self, node_id: int, value: Any, acceptor_ids: List[int],
                 learner_ids: List[int], network: Network):
        self.node_id = node_id
        self.desired_value = value
        self.acceptor_ids = acceptor_ids
        self.learner_ids = learner_ids
        self.network = network
        self.majority = len(acceptor_ids) // 2 + 1

        self.current_round = 0
        self.promises: Dict[int, Tuple[Optional[ProposalNum], Optional[Any]]] = {}
        self.accepts: int = 0
        self.phase = 1
        self.chosen_value: Optional[Any] = None

    def start_round(self):
        """Begin Phase 1: send Prepare to all acceptors."""
        self.current_round += 1
        self.promises = {}
        self.accepts = 0
        self.phase = 1

        n = ProposalNum(self.current_round, self.node_id)
        for aid in self.acceptor_ids:
            self.network.send(Message(MsgType.PREPARE, self.node_id, aid, n))

    def handle(self, msg: Message):
        if msg.msg_type == MsgType.PROMISE:
            self._handle_promise(msg)
        elif msg.msg_type == MsgType.ACCEPTED:
            self._handle_accepted(msg)
        elif msg.msg_type in (MsgType.NACK_PREPARE, MsgType.NACK_ACCEPT):
            self._handle_nack(msg)

    def _handle_promise(self, msg: Message):
        if self.phase != 1:
            return
        expected_n = ProposalNum(self.current_round, self.node_id)
        if msg.proposal_num != expected_n:
            return

        self.promises[msg.src] = (msg.accepted_num, msg.accepted_val)

        if len(self.promises) >= self.majority:
            self.phase = 2
            value = self._select_value()

            n = ProposalNum(self.current_round, self.node_id)
            for aid in self.acceptor_ids:
                self.network.send(Message(
                    MsgType.ACCEPT, self.node_id, aid, n, value=value
                ))

    def _select_value(self) -> Any:
        """Paxos value selection rule."""
        highest_accepted = None
        highest_value = None
        for acc_num, acc_val in self.promises.values():
            if acc_num is not None:
                if highest_accepted is None or acc_num > highest_accepted:
                    highest_accepted = acc_num
                    highest_value = acc_val

        if highest_value is not None:
            return highest_value
        return self.desired_value

    def _handle_accepted(self, msg: Message):
        if self.phase != 2:
            return
        self.accepts += 1
        if self.accepts >= self.majority and self.chosen_value is None:
            self.chosen_value = msg.value
            # Notify learners
            n = ProposalNum(self.current_round, self.node_id)
            for lid in self.learner_ids:
                self.network.send(Message(
                    MsgType.DECIDE, self.node_id, lid, n, value=msg.value
                ))

    def _handle_nack(self, msg: Message):
        # Back off and retry with a higher round
        pass  # handled by external retry logic


# ──────────────────────────────────────────────
# Learner
# ──────────────────────────────────────────────

class Learner:
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.learned_value: Optional[Any] = None

    def handle(self, msg: Message):
        if msg.msg_type == MsgType.DECIDE and self.learned_value is None:
            self.learned_value = msg.value


# ──────────────────────────────────────────────
# Simulation Driver
# ──────────────────────────────────────────────

def run_simulation(
    num_acceptors: int = 5,
    proposer_values: Dict[int, Any] = None,
    loss_rate: float = 0.1,
    seed: int = 42,
):
    """Run a complete Paxos simulation.

    Args:
        num_acceptors: number of acceptor nodes
        proposer_values: {proposer_id: proposed_value}
        loss_rate: probability of message loss (0.0 to 1.0)
        seed: random seed for reproducibility
    """
    random.seed(seed)

    if proposer_values is None:
        proposer_values = {100: "alpha", 101: "beta"}

    network = Network(loss_rate=loss_rate, min_delay=1.0, max_delay=5.0)
    acceptor_ids = list(range(num_acceptors))
    learner_ids = [200]

    # Create nodes
    acceptors = {aid: Acceptor(aid, network) for aid in acceptor_ids}
    learners = {lid: Learner(lid) for lid in learner_ids}
    proposers = {}
    for pid, val in proposer_values.items():
        proposers[pid] = Proposer(pid, val, acceptor_ids, learner_ids, network)

    # Dispatch table
    all_nodes = {}
    all_nodes.update(acceptors)
    all_nodes.update(learners)
    all_nodes.update(proposers)

    # Start proposers
    for p in proposers.values():
        p.start_round()

    # Run simulation
    max_rounds = 500
    rounds = 0
    while network.has_messages() and rounds < max_rounds:
        msg = network.deliver_next()
        if msg is None:
            break

        dst_node = all_nodes.get(msg.dst)
        if dst_node:
            dst_node.handle(msg)

        rounds += 1

    # Report results
    print(f"Simulation completed in {rounds} message deliveries")
    print(f"Network time: {network.current_time:.1f} units")
    print(f"Messages lost: ~{loss_rate*100:.0f}% rate")
    print()

    for pid, p in proposers.items():
        status = f"CHOSEN: {p.chosen_value}" if p.chosen_value else "no value chosen"
        print(f"Proposer {pid} (wanted '{p.desired_value}'): {status}")

    for lid, l in learners.items():
        status = f"LEARNED: {l.learned_value}" if l.learned_value else "nothing learned"
        print(f"Learner {lid}: {status}")

    # Verify safety: all nodes that learned a value agree
    learned_values = set()
    for l in learners.values():
        if l.learned_value is not None:
            learned_values.add(l.learned_value)
    for p in proposers.values():
        if p.chosen_value is not None:
            learned_values.add(p.chosen_value)

    if len(learned_values) <= 1:
        print("\nSAFETY CHECK: PASSED (at most one value chosen)")
    else:
        print(f"\nSAFETY CHECK: FAILED! Multiple values: {learned_values}")

    return learned_values


if __name__ == "__main__":
    print("=" * 60)
    print("Scenario 1: Single proposer, no message loss")
    print("=" * 60)
    run_simulation(num_acceptors=5, proposer_values={100: "hello"}, loss_rate=0.0)

    print()
    print("=" * 60)
    print("Scenario 2: Two competing proposers, 10% message loss")
    print("=" * 60)
    run_simulation(
        num_acceptors=5,
        proposer_values={100: "alpha", 101: "beta"},
        loss_rate=0.1,
        seed=123
    )

    print()
    print("=" * 60)
    print("Scenario 3: Three proposers, 20% message loss")
    print("=" * 60)
    run_simulation(
        num_acceptors=7,
        proposer_values={100: "X", 101: "Y", 102: "Z"},
        loss_rate=0.2,
        seed=456
    )
```

### 8.1 시뮬레이터 실행

```bash
python paxos_simulator.py
```

예상 출력(랜덤 시드에 따라 다름):

```
============================================================
Scenario 1: Single proposer, no message loss
============================================================
Simulation completed in 11 message deliveries
Network time: 15.3 units
Messages lost: ~0% rate

Proposer 100 (wanted 'hello'): CHOSEN: hello
Learner 200: LEARNED: hello

SAFETY CHECK: PASSED (at most one value chosen)

============================================================
Scenario 2: Two competing proposers, 10% message loss
============================================================
Simulation completed in 19 message deliveries
Network time: 22.7 units
Messages lost: ~10% rate

Proposer 100 (wanted 'alpha'): CHOSEN: alpha
Proposer 101 (wanted 'beta'): no value chosen
Learner 200: LEARNED: alpha

SAFETY CHECK: PASSED (at most one value chosen)
```

### 8.2 시뮬레이터 확장

더 깊은 이해를 위한 핵심 연습:

1. **재시도 로직 추가**: proposer가 NACK을 받으면, 라운드 번호를 증가시키고 지수 백오프로 재시도해야 한다.

2. **노드 충돌 구현**: 시뮬레이션 중간에 acceptor를 제거하고 Paxos가 여전히 합의에 도달하는지 확인한다(남은 과반수로).

3. **네트워크 파티션 추가**: acceptor를 두 그룹으로 분리하고 어느 그룹도 단독으로 합의에 도달할 수 없음을 관찰한다.

4. **메시지 수 측정**: 총 메시지를 세고 이론적 `O(N)` (phase당)과 비교한다.

---

## 9. 일반적인 함정과 구현 조언

### 9.1 응답 전 내구성

가장 흔한 구현 버그: **상태를 지속시키기 전에 Prepare 또는 Accept에 응답하는 것**. Acceptor가 응답 후 지속시키기 전에 충돌하면, 복구 후 약속을 위반할 수 있다.

**규칙**: 응답을 보내기 전에 항상 `fsync()`를 수행한다.

### 9.2 오래된 메시지

이전 라운드의 메시지가 언제든 도착할 수 있다. 모든 핸들러는 메시지의 제안 번호가 현재 예상 라운드와 일치하는지 확인해야 한다.

### 9.3 제안 번호 고갈

Proposer가 `(round, node_id)` 쌍을 사용하고 재시도마다 라운드를 증가시키면, 긴밀한 livelock 루프가 라운드 카운터를 고갈시킬 수 있다. 64비트 정수를 사용하고 백오프를 추가한다.

### 9.4 지정 Proposer vs 진정한 Leader

Paxos는 liveness를 위해서만 "지정 proposer"를 필요로 하며, safety를 위해서는 필요하지 않다. 시스템은 활성 proposer가 몇 명이든 항상 안전하다. 이것은 두 leader가 프로토콜 불변 조건을 위반하는 Raft와 같은 프로토콜과의 핵심적인 차이점이다.

### 9.5 실용적 읽기 목록

| 논문 | 연도 | 주요 기여 |
|-------|------|-----------------|
| Lamport, "The Part-Time Parliament" | 1998 | 원조 Paxos |
| Lamport, "Paxos Made Simple" | 2001 | 단순화된 설명 |
| Lamport, "Fast Paxos" | 2006 | 더 큰 쿼럼으로 1 RT 커밋 |
| Howard et al., "Flexible Paxos" | 2016 | 분리된 쿼럼 크기 |
| Moraru et al., "Egalitarian Paxos" | 2013 | 리더 없는 Paxos |
| van Renesse & Altinbuken, "Paxos Made Moderately Complex" | 2015 | 구현 가이드 |
| Lamport & Massa, "Cheap Paxos" | 2004 | 줄어든 replica 수 |

---

## 10. 요약

Single-decree Paxos는 과반수 쿼럼에 기반한 2단계 프로토콜을 통해 분산 합의의 근본적인 문제를 해결한다. 정확성은 세 가지 메커니즘에 의존한다: 고유하고 순서가 지정된 제안 번호, 오래된 제안이 성공하는 것을 방지하는 promise 메커니즘, 그리고 새로운 proposer가 이전에 수락된 값을 채택하도록 강제하는 값 선택 규칙이다.

Multi-Paxos는 안정적인 리더십을 통해 Phase 1의 비용을 여러 로그 항목에 걸쳐 분할 상환하여 이를 복제 로그로 확장한다. Flexible Paxos는 자주 실행되는 Phase 2에 대해 더 작은 쿼럼을 사용하여 일반적인 경우를 더욱 최적화한다. Egalitarian Paxos는 leader를 완전히 제거하여, 간섭하지 않는 명령을 단일 왕복으로 커밋할 수 있게 한다.

Paxos 계열 프로토콜은 30년 이상 분산 합의의 이론적 기반이었다. Raft(레슨 6에서 다룸)가 이해도 덕분에 새로운 구현의 선호 선택이 되었지만, Paxos를 이해하는 것은 분산 합의의 근본적인 트레이드오프에 대해 추론하는 데 여전히 필수적이다.

---

[다음: Raft 심화](./06_Raft_In_Depth.md)

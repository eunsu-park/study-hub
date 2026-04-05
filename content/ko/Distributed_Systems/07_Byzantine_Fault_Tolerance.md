# 레슨 7: Byzantine Fault Tolerance

[개요](./00_Overview.md) | [이전: Raft 심화](./06_Raft_In_Depth.md) | [다음: 분산 트랜잭션](./08_Distributed_Transactions.md)

---

## 학습 목표

- Byzantine Generals Problem을 이해하고 n ≤ 3f 노드에서 합의의 불가능성을 증명한다
- 정상 운영, view 변경, checkpoint를 포함한 PBFT 프로토콜을 추적한다
- PBFT를 현대 선형 복잡도 BFT 프로토콜(HotStuff, Tendermint)과 비교한다
- BFT 합의와 블록체인 시스템 간의 연결을 분석한다
- 3단계 커밋 프로세스를 보여주는 간소화된 PBFT 시뮬레이터를 구현한다

---

## 1. Byzantine Generals Problem

### 1.1 문제 정의

1982년, Lamport, Shostak, Pease는 **Byzantine Generals Problem**을 공식화했다: 군대의 부대를 지휘하는 장군들이 공통 전투 계획(공격 또는 후퇴)에 합의해야 한다. 일부 장군은 서로 다른 장군들에게 상충되는 메시지를 보내는 **배신자**일 수 있다.

이 문제는 분산 컴퓨팅의 근본적인 도전을 포착한다: **일부 참여자가 임의로 행동할 수 있을 때 합의에 도달하는 것** — 잘못된 값 전송, 다른 피어에게 다른 값 전송, 또는 전혀 응답하지 않는 것.

```
     ┌─────────────┐
     │ General 1   │──── "Attack!" ────▶ General 2
     │ (Traitor!)  │
     │             │──── "Retreat!" ───▶ General 3
     └─────────────┘

General 2 thinks: "Attack"
General 3 thinks: "Retreat"
Result: Inconsistent action → disaster
```

### 1.2 형식적 정의

- **n**명의 장군, 그 중 최대 **f**명이 **Byzantine** (결함/악의적)
- 모든 충성스러운 장군은 동일한 값에 합의해야 한다 (**Agreement**)
- 모든 충성스러운 장군이 같은 값을 제안하면, 그 값이 선택되어야 한다 (**Validity**)
- 프로토콜은 종료되어야 한다 (**Termination**)

### 1.3 n ≤ 3f에서의 불가능성

**정리** (Lamport, Shostak, Pease, 1982): *한 명이라도 배신자인 경우 3명 이하의 장군에 대해 Byzantine Generals Problem을 해결할 수 있는 프로토콜은 존재하지 않는다.*

**증명 (n = 3, f = 1)**:

프로토콜이 존재한다고 가정한다. 세 장군 G1, G2, G3을 고려하고, G3이 배신자이다.

```
Scenario A: G1 proposes "Attack", G2 proposes "Attack"
  G3 (traitor) tells G1: "Attack" and tells G2: "Retreat"

  G1 sees: G1="Attack", G2="Attack", G3="Attack" → decides "Attack"
  G2 sees: G1="Attack", G2="Attack", G3="Retreat" → ???

Scenario B: G1 proposes "Attack", G2 proposes "Retreat"
  G3 (traitor) tells G1: "Attack" and tells G2: "Retreat"

  G1 sees: G1="Attack", G2=???, G3="Attack"
  G2 sees: G1=???, G2="Retreat", G3="Retreat"
```

G2는 Scenario A(G3이 자신의 값에 대해 거짓말하는 배신자)와 G1이 배신자인 다른 시나리오를 구분할 수 없다. 두 경우 모두 G2는 상충되는 정보를 받고 진실을 판별할 수 없다.

**일반화**: 임의의 `n ≤ 3f`에 대해 동일한 논증이 적용된다. `f`명의 배신자가 조율하여 서로 다른 `f`명의 충성 장군 집합이 모순되는 정보를 보게 할 수 있다.

**핵심 결과**: Byzantine fault tolerance는 **n ≥ 3f + 1** 노드를 필요로 한다.

| 전체 노드 (n) | 최대 Byzantine 결함 (f) | 필요한 정직 노드 | 예시 |
|-----------------|------------------------|-----------------|---------|
| 4 | 1 | 3 | 최소 BFT 시스템 |
| 7 | 2 | 5 | |
| 10 | 3 | 7 | |
| 3f + 1 | f | 2f + 1 | 일반 공식 |

### 1.4 Oral Messages vs Signed Messages

Lamport의 원래 논문은 두 모델을 구분했다:

**Oral Messages (OM)**: 메시지는 위조될 수 있다. 배신자는 다른 장군이 보내지 않은 메시지를 보냈다고 주장할 수 있다. 이는 `n ≥ 3f + 1`을 필요로 한다.

**Signed Messages (SM)**: 메시지는 암호학적으로 서명되어 위조할 수 없다. 이를 통해 `n ≥ 2f + 1`로 Byzantine agreement를 해결할 수 있다(단, 기본 알고리즘에서 지수적 메시지 복잡도).

```python
from dataclasses import dataclass, field
from typing import Optional, List, Set, Dict, Tuple
import hashlib
import json

@dataclass
class SignedMessage:
    """A message with a chain of cryptographic signatures."""
    value: str
    signatures: List[Tuple[int, str]] = field(default_factory=list)

    def sign(self, node_id: int, private_key: str) -> 'SignedMessage':
        """Add a signature to the message."""
        content = json.dumps({
            'value': self.value,
            'prior_signatures': [(nid, sig) for nid, sig in self.signatures]
        })
        signature = hashlib.sha256(
            (content + private_key).encode()
        ).hexdigest()[:16]

        new_msg = SignedMessage(
            value=self.value,
            signatures=self.signatures + [(node_id, signature)]
        )
        return new_msg

    def verify(self, node_id: int, public_key: str) -> bool:
        """Verify that a specific node signed this message."""
        for nid, sig in self.signatures:
            if nid == node_id:
                return True  # simplified verification
        return False

    @property
    def signer_chain(self) -> List[int]:
        """Return the chain of signers."""
        return [nid for nid, _ in self.signatures]
```

### 1.5 왜 n ≥ 3f + 1인가?

직관은 쿼럼 교차에 기반한다:

- `f`개의 Byzantine 결함을 허용하려면, 두 쿼럼이 최소 `f + 1`개의 정직 노드에서 겹치는 크기 `q`의 쿼럼이 필요하다.
- `n`개의 전체 노드에서, 쿼럼은 `q = ⌈(2n + f + 1) / 3⌉` 노드의 집합이다.
- `n = 3f + 1`에서, 쿼럼 크기는 `2f + 1`이다.
- `3f + 1` 노드에서 크기 `2f + 1`의 두 쿼럼은 최소 `f + 1`개 노드에서 겹친다.
- `f + 1`개 노드 중 최대 `f`개가 Byzantine이므로, 최소 하나의 겹치는 노드가 정직하다.

---

## 2. PBFT: Practical Byzantine Fault Tolerance

### 2.1 개요

Castro와 Liskov는 1999년에 PBFT를 발표하여 최초의 **실용적** BFT 프로토콜을 제공했다. PBFT 이전에는 BFT 프로토콜이 지수적 메시지 복잡도를 가지거나 동기 네트워크를 요구했다. PBFT는 다음을 달성한다:

- **O(n²)** 요청당 메시지 복잡도
- **비동기** 운영 (안전성을 위한 타이밍 가정 없음)
- 약한 동기 가정(최종적 메시지 전달)으로 **Liveness**
- `n = 3f + 1` 노드로 `f`개의 Byzantine 결함 허용

### 2.2 시스템 모델

- **n = 3f + 1** replica, 0부터 n-1까지 번호 부여
- 하나의 replica가 **primary** (leader): replica `p = v mod n` 여기서 `v`는 view 번호
- 클라이언트는 primary에 요청을 보냄
- 모든 replica는 결정론적 상태 머신을 유지

### 2.3 정상 운영: 3단계 프로토콜

PBFT는 세 단계를 사용한다: **pre-prepare**, **prepare**, **commit**.

```
Client    Primary(0)   Replica 1    Replica 2    Replica 3
  │           │             │            │            │
  │──Request──▶             │            │            │
  │           │             │            │            │
  │           │──PrePrepare─▶            │            │
  │           │──PrePrepare──────────────▶            │
  │           │──PrePrepare──────────────────────────▶│
  │           │             │            │            │
  │           │◀──Prepare───│            │            │
  │           │   Prepare───▶────────────▶────────────▶
  │           │◀─────────────────Prepare─│            │
  │           │   ◀──────────────Prepare─▶────────────▶
  │           │◀─────────────────────────────Prepare──│
  │           │   ◀──────────────────────────Prepare──▶
  │           │             │            │            │
  │           │◀──Commit────│            │            │
  │           │   Commit────▶────────────▶────────────▶
  │           │◀─────────────────Commit──│            │
  │           │   ◀──────────────Commit──▶────────────▶
  │           │◀─────────────────────────────Commit───│
  │           │   ◀──────────────────────────Commit───▶
  │           │             │            │            │
  │◀──Reply───│             │            │            │
  │◀────────────────Reply───│            │            │
  │◀─────────────────────────────Reply───│            │
  │◀──────────────────────────────────────────Reply───│
```

### 2.4 단계 세부사항

**Pre-prepare** (primary → 모든 replica):
- Primary가 view `v`에서 요청에 시퀀스 번호 `n`을 할당
- 메시지: `⟨PRE-PREPARE, v, n, d⟩` 여기서 `d`는 요청의 다이제스트
- Replica는 다음 경우 수락: view 일치, 시퀀스 번호가 유효 범위 내, 동일한 `(v, n)`에 대해 다른 pre-prepare를 수락하지 않은 경우

**Prepare** (각 replica → 모든 replica):
- Pre-prepare를 수락한 후, 각 replica가 prepare 메시지를 브로드캐스트
- 메시지: `⟨PREPARE, v, n, d, i⟩` 여기서 `i`는 replica ID
- Replica가 pre-prepare와 `2f`개의 일치하는 prepare를 가지면 **prepared** 상태
- Prepared는 "이 요청이 view `v`에서 시퀀스 번호 `n`에 할당되었음을 인증합니다"를 의미

**Commit** (각 replica → 모든 replica):
- Prepared가 된 후, 각 replica가 commit 메시지를 브로드캐스트
- 메시지: `⟨COMMIT, v, n, d, i⟩`
- Replica가 `2f + 1`개의 일치하는 commit을 가지면 **committed-local** 상태
- committed-local 후, replica가 요청을 실행하고 클라이언트에 응답을 보냄

```python
from enum import Enum
from collections import defaultdict


class Phase(Enum):
    IDLE = "idle"
    PRE_PREPARED = "pre-prepared"
    PREPARED = "prepared"
    COMMITTED = "committed"
    EXECUTED = "executed"


@dataclass
class PBFTMessage:
    msg_type: str           # "pre-prepare", "prepare", "commit", "reply"
    view: int
    sequence: int
    digest: str             # hash of the request
    sender: int
    request: Optional[str] = None  # original request (only in pre-prepare)


class PBFTReplica:
    """A single PBFT replica."""

    def __init__(self, replica_id: int, n_replicas: int):
        self.id = replica_id
        self.n = n_replicas
        self.f = (n_replicas - 1) // 3
        self.view = 0

        # Message logs
        self.pre_prepares: Dict[Tuple[int, int], PBFTMessage] = {}  # (v, seq) -> msg
        self.prepares: Dict[Tuple[int, int], Set[int]] = defaultdict(set)  # (v, seq) -> {sender_ids}
        self.commits: Dict[Tuple[int, int], Set[int]] = defaultdict(set)   # (v, seq) -> {sender_ids}

        # State
        self.phase: Dict[Tuple[int, int], Phase] = {}  # (v, seq) -> phase
        self.executed: List[str] = []
        self.outgoing: List[PBFTMessage] = []  # messages to send

    @property
    def is_primary(self) -> bool:
        return self.id == self.view % self.n

    def handle_request(self, request: str, seq: int):
        """Primary handles a client request by sending pre-prepare."""
        if not self.is_primary:
            return

        digest = hashlib.sha256(request.encode()).hexdigest()[:16]
        msg = PBFTMessage("pre-prepare", self.view, seq, digest, self.id, request)
        self.pre_prepares[(self.view, seq)] = msg
        self.phase[(self.view, seq)] = Phase.PRE_PREPARED

        # Broadcast pre-prepare to all replicas
        for i in range(self.n):
            if i != self.id:
                self.outgoing.append(PBFTMessage(
                    "pre-prepare", self.view, seq, digest, self.id, request
                ))

    def handle_pre_prepare(self, msg: PBFTMessage):
        """Backup handles pre-prepare from primary."""
        key = (msg.view, msg.sequence)

        # Validate
        if msg.view != self.view:
            return
        if msg.sender != self.view % self.n:
            return  # not from primary
        if key in self.pre_prepares:
            return  # already have a pre-prepare for this slot

        # Verify digest
        expected_digest = hashlib.sha256(msg.request.encode()).hexdigest()[:16]
        if msg.digest != expected_digest:
            return  # digest mismatch (Byzantine primary)

        # Accept pre-prepare
        self.pre_prepares[key] = msg
        self.phase[key] = Phase.PRE_PREPARED

        # Send prepare to all replicas
        for i in range(self.n):
            if i != self.id:
                self.outgoing.append(PBFTMessage(
                    "prepare", self.view, msg.sequence, msg.digest, self.id
                ))

    def handle_prepare(self, msg: PBFTMessage):
        """Handle prepare message from another replica."""
        key = (msg.view, msg.sequence)

        if msg.view != self.view:
            return

        self.prepares[key].add(msg.sender)

        # Check if we have enough prepares (2f) plus the pre-prepare
        if (key in self.pre_prepares and
                len(self.prepares[key]) >= 2 * self.f and
                self.phase.get(key) == Phase.PRE_PREPARED):

            self.phase[key] = Phase.PREPARED

            # Send commit to all replicas
            for i in range(self.n):
                if i != self.id:
                    self.outgoing.append(PBFTMessage(
                        "commit", self.view, msg.sequence, msg.digest, self.id
                    ))

    def handle_commit(self, msg: PBFTMessage):
        """Handle commit message from another replica."""
        key = (msg.view, msg.sequence)

        if msg.view != self.view:
            return

        self.commits[key].add(msg.sender)

        # Check if we have enough commits (2f + 1)
        if (len(self.commits[key]) >= 2 * self.f + 1 and
                self.phase.get(key) in (Phase.PREPARED, Phase.PRE_PREPARED)):

            self.phase[key] = Phase.COMMITTED
            # Execute the request
            if key in self.pre_prepares:
                self.executed.append(self.pre_prepares[key].request)
                self.phase[key] = Phase.EXECUTED
```

### 2.5 왜 3단계인가?

2단계(pre-prepare + prepare)만으로는 충분하지 않다. replica가 다른 replica들도 "prepared" 상태에 도달했는지 확신할 수 없기 때문이다. commit 단계는 **최소 2f + 1 replica가 2f + 1 replica가 순서에 합의했음을 안다**는 것을 보장한다.

commit 단계 없이는, view 변경 중에 어떤 요청이 실제로 순서가 정해졌는지 vs 단지 임시로 시퀀스 번호가 할당된 것인지 판별할 수 없다.

| 단계 | 목적 | 필요한 쿼럼 |
|-------|---------|--------------|
| Pre-prepare | Primary가 시퀀스 번호 할당 | 1 (primary만) |
| Prepare | Replica가 view에서 시퀀스에 합의 | 2f + 1 (primary 포함) |
| Commit | Replica가 순서가 안정적임을 인증 | 2f + 1 |

### 2.6 메시지 복잡도

각 클라이언트 요청에 대해:

```
Pre-prepare:  primary → n-1 replicas               = n - 1 messages
Prepare:      each of n replicas → n-1 others       = n(n - 1) messages
Commit:       each of n replicas → n-1 others       = n(n - 1) messages
Reply:        each replica → client (client needs f+1) = f + 1 messages

Total: (n-1) + n(n-1) + n(n-1) + (f+1) ≈ 2n² messages = O(n²)
```

`n = 4` (1 결함 허용): 요청당 ~32 메시지.
`n = 7` (2 결함 허용): 요청당 ~98 메시지.
`n = 100` (33 결함 허용): 요청당 ~20,000 메시지.

**O(n²) 병목**은 PBFT를 대규모 replica 집합에서 비실용적으로 만든다.

### 2.7 View Change 프로토콜

Primary가 결함이 의심되면(예: pre-prepare를 보내지 않음), replica들이 **view change**를 시작하여 새 primary를 선출한다.

```
View Change Protocol:
1. Replica detects primary failure (timeout)
2. Replica broadcasts ⟨VIEW-CHANGE, v+1, ...prepared_proofs...⟩
3. New primary (replica v+1 mod n) collects 2f VIEW-CHANGE messages
4. New primary broadcasts ⟨NEW-VIEW, v+1, ...view_change_msgs..., ...pre-prepares...⟩
5. Replicas verify NEW-VIEW and adopt new view

The prepared_proofs included in VIEW-CHANGE messages contain
evidence of which requests were prepared in the old view.
The new primary must include pre-prepares for all prepared
requests in its NEW-VIEW message.
```

```python
@dataclass
class ViewChangeMessage:
    new_view: int
    sender: int
    prepared_proofs: List[Tuple[int, int, str]]  # (view, seq, digest) for each prepared request
    checkpoint_seq: int  # latest stable checkpoint


class ViewChanger:
    """Handles PBFT view change protocol."""

    def __init__(self, replica_id, n_replicas):
        self.id = replica_id
        self.n = n_replicas
        self.f = (n_replicas - 1) // 3
        self.view_change_msgs: Dict[int, List[ViewChangeMessage]] = defaultdict(list)

    def initiate_view_change(self, new_view, prepared_proofs, checkpoint_seq):
        """Start a view change when primary is suspected faulty."""
        msg = ViewChangeMessage(
            new_view=new_view,
            sender=self.id,
            prepared_proofs=prepared_proofs,
            checkpoint_seq=checkpoint_seq
        )
        return msg  # broadcast to all

    def handle_view_change(self, msg: ViewChangeMessage):
        """Collect view change messages (new primary only)."""
        self.view_change_msgs[msg.new_view].append(msg)

        if len(self.view_change_msgs[msg.new_view]) >= 2 * self.f:
            return self._compute_new_view(msg.new_view)
        return None

    def _compute_new_view(self, new_view):
        """New primary computes the set of pre-prepares for the new view.

        For each sequence number that was prepared in any old view,
        the new primary must re-propose it. For sequence numbers
        that were not prepared, the new primary proposes a no-op.
        """
        msgs = self.view_change_msgs[new_view]

        # Find the range of sequence numbers to re-propose
        min_seq = min(m.checkpoint_seq for m in msgs) + 1
        max_seq = max(
            max((seq for _, seq, _ in m.prepared_proofs), default=0)
            for m in msgs
        )

        re_proposals = {}
        for seq in range(min_seq, max_seq + 1):
            # Find the prepared proof with the highest view for this seq
            best = None
            for m in msgs:
                for v, s, d in m.prepared_proofs:
                    if s == seq and (best is None or v > best[0]):
                        best = (v, d)

            if best is not None:
                re_proposals[seq] = best[1]  # re-propose with same digest
            else:
                re_proposals[seq] = "NOP"  # no-op

        return re_proposals
```

### 2.8 Checkpoint과 가비지 컬렉션

PBFT replica는 주기적으로 **checkpoint**를 수행하고 오래된 로그 항목을 가비지 컬렉션한다:

1. 매 `K`번의 요청마다(예: K=100), replica가 상태의 checkpoint를 수행
2. `⟨CHECKPOINT, n, d, i⟩`를 브로드캐스트 여기서 `n`은 시퀀스 번호, `d`는 상태 다이제스트
3. Replica가 `2f + 1`개의 일치하는 checkpoint 메시지를 수집하면, checkpoint가 **안정**
4. 시퀀스 번호 ≤ `n`인 모든 로그 항목과 메시지를 폐기 가능

```python
class CheckpointManager:
    """Manages PBFT checkpoints for garbage collection."""

    CHECKPOINT_INTERVAL = 100

    def __init__(self, replica_id, n_replicas):
        self.id = replica_id
        self.f = (n_replicas - 1) // 3
        self.checkpoint_proofs: Dict[int, Set[int]] = defaultdict(set)
        self.stable_checkpoint_seq = 0

    def maybe_checkpoint(self, last_executed_seq, state_digest):
        """Check if it's time for a checkpoint."""
        if last_executed_seq % self.CHECKPOINT_INTERVAL != 0:
            return None

        # This node checkpoints; broadcast to others
        self.checkpoint_proofs[last_executed_seq].add(self.id)
        return {
            'type': 'checkpoint',
            'seq': last_executed_seq,
            'digest': state_digest,
            'sender': self.id
        }

    def handle_checkpoint(self, seq, digest, sender):
        """Process checkpoint message from another replica."""
        self.checkpoint_proofs[seq].add(sender)

        if len(self.checkpoint_proofs[seq]) >= 2 * self.f + 1:
            self.stable_checkpoint_seq = seq
            # Garbage collect old entries
            old_seqs = [s for s in self.checkpoint_proofs if s < seq]
            for s in old_seqs:
                del self.checkpoint_proofs[s]
            return True  # checkpoint is stable
        return False
```

---

## 3. BFT-SMaRt

BFT-SMaRt는 고성능 오픈 소스 Java BFT 상태 머신 복제 구현이다(Bessani, Sousa, Alchieri, 2014). 주요 특성:

| 특성 | 설명 |
|---------|------------|
| 언어 | Java |
| 프로토콜 | 최적화된 PBFT 기반 |
| 처리량 | ~80K ops/sec (4 replica, LAN) |
| 배치 처리 | 합의 인스턴스당 여러 요청 그룹화 |
| Leader 기반 | 정상 케이스에서 메시지 복잡도 감소 |
| View change | 증명 가능하게 올바른 view change 프로토콜 |
| 재구성 | 동적 멤버십 변경 지원 |

BFT-SMaRt는 EBSI(European Blockchain Services Infrastructure)를 포함한 연구 및 상업 애플리케이션에서 사용되는 몇 안 되는 **프로덕션 품질** BFT 구현 중 하나로 주목할 만하다.

---

## 4. HotStuff

### 4.1 동기: 선형 메시지 복잡도

PBFT의 O(n²) 메시지 복잡도는 확장성을 제한한다. HotStuff(Yin, Malkhi, Reiter, Gueta, Abraham, 2019)는 **threshold signature**를 사용하여 이를 **O(n)**으로 줄인다.

### 4.2 핵심 통찰: Threshold Signature

각 replica가 다른 모든 replica에 브로드캐스트하는 대신(O(n²)), HotStuff는 **별 토폴로지**를 사용한다: 모든 통신이 leader를 거친다.

```
PBFT (n² messages):                    HotStuff (n messages):
  R1 ←→ R2                              R1 ──→ Leader ──→ R1
  R1 ←→ R3                              R2 ──→ Leader ──→ R2
  R1 ←→ R4                              R3 ──→ Leader ──→ R3
  R2 ←→ R3                              R4 ──→ Leader ──→ R4
  R2 ←→ R4
  R3 ←→ R4
  Total: n(n-1) = 12                    Total: 2n = 8
```

Leader가 `2f + 1`개의 부분 서명을 수집하고 **threshold signature**(쿼럼을 나타내는 단일 서명)로 결합한다. 이 threshold signature는 어떤 replica든 검증할 수 있다.

### 4.3 3단계 프로토콜

HotStuff는 PBFT와 같이 3단계를 사용하지만 각 단계에서 선형 통신을 한다:

```
Phase 1: PREPARE
  Leader → all: ⟨PREPARE, node, QC_prev⟩
  all → Leader: ⟨vote, partial_sig⟩
  Leader: combine 2f+1 partial sigs → prepareQC

Phase 2: PRE-COMMIT
  Leader → all: ⟨PRE-COMMIT, node, prepareQC⟩
  all → Leader: ⟨vote, partial_sig⟩
  Leader: combine → precommitQC

Phase 3: COMMIT
  Leader → all: ⟨COMMIT, node, precommitQC⟩
  all → Leader: ⟨vote, partial_sig⟩
  Leader: combine → commitQC

Phase 4: DECIDE
  Leader → all: ⟨DECIDE, node, commitQC⟩
  All replicas execute the request
```

```python
@dataclass
class QuorumCertificate:
    """A threshold signature proving 2f+1 replicas voted for a proposal."""
    view: int
    node_hash: str
    combined_signature: str  # threshold signature from 2f+1 partial sigs

    def verify(self, threshold_key) -> bool:
        """Verify the threshold signature (abstracted)."""
        return True  # simplified


@dataclass
class HotStuffNode:
    """A proposal node in the HotStuff protocol (not a physical node)."""
    parent_hash: str
    command: str
    view: int
    justify: Optional[QuorumCertificate] = None  # QC that justifies this node

    @property
    def hash(self) -> str:
        content = f"{self.parent_hash}:{self.command}:{self.view}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class HotStuffReplica:
    """A HotStuff replica implementing the three-phase protocol."""

    def __init__(self, replica_id: int, n_replicas: int):
        self.id = replica_id
        self.n = n_replicas
        self.f = (n_replicas - 1) // 3
        self.view = 0
        self.locked_qc: Optional[QuorumCertificate] = None
        self.prepare_qc: Optional[QuorumCertificate] = None
        self.partial_sigs: Dict[str, List[str]] = defaultdict(list)
        self.executed: List[str] = []

    @property
    def is_leader(self) -> bool:
        return self.id == self.view % self.n

    def on_propose(self, command: str, parent: HotStuffNode) -> HotStuffNode:
        """Leader creates a new proposal."""
        node = HotStuffNode(
            parent_hash=parent.hash,
            command=command,
            view=self.view,
            justify=self.prepare_qc
        )
        return node

    def on_receive_proposal(self, node: HotStuffNode) -> Optional[str]:
        """Replica votes on a proposal (returns partial signature if valid).

        Safety rule: only vote if:
        1. The node extends from the locked QC, OR
        2. The node's justify QC has a higher view than our locked QC
        """
        if not self._is_safe(node):
            return None

        # Generate partial signature (simplified)
        partial_sig = f"sig_{self.id}_{node.hash}"
        return partial_sig

    def _is_safe(self, node: HotStuffNode) -> bool:
        """Check the safety rule for voting."""
        if self.locked_qc is None:
            return True

        # Safe if node extends the locked node
        # (simplified: check justify view)
        if node.justify is not None and node.justify.view >= self.locked_qc.view:
            return True

        return False

    def on_receive_qc(self, qc: QuorumCertificate, phase: str):
        """Process a quorum certificate from the leader."""
        if phase == "prepare":
            self.prepare_qc = qc
        elif phase == "pre-commit":
            self.locked_qc = qc  # lock on pre-commit QC
        elif phase == "commit":
            # Execute the command
            pass
```

### 4.4 처리량을 위한 파이프라이닝

HotStuff는 **파이프라이닝**을 가능하게 한다: 하나의 합의 인스턴스의 각 단계가 후속 인스턴스의 단계와 겹친다. Leader는 다음 제안을 시작하기 전에 세 단계가 모두 완료될 때까지 기다리지 않는다.

```
View 1: PREPARE(cmd1)  → prepareQC(cmd1)
View 2: PRE-COMMIT(cmd1) + PREPARE(cmd2)  → precommitQC(cmd1) + prepareQC(cmd2)
View 3: COMMIT(cmd1) + PRE-COMMIT(cmd2) + PREPARE(cmd3) → ...
View 4: DECIDE(cmd1) + COMMIT(cmd2) + PRE-COMMIT(cmd3) + PREPARE(cmd4)

Each view does work for multiple commands simultaneously.
Effective latency: 1 round trip per command (amortized).
```

이는 CPU 명령어 파이프라이닝과 유사하다: 각 개별 명령은 여전히 3단계가 걸리지만, 처리량은 왕복당 1명령에 접근한다.

### 4.5 HotStuff에서의 View Change

HotStuff의 view change는 PBFT의 것보다 극적으로 단순하다:

```
HotStuff View Change:
1. Replica times out → sends ⟨NEW-VIEW, v+1, prepareQC⟩ to new leader
2. New leader collects 2f+1 NEW-VIEW messages
3. New leader picks the highest prepareQC (highQC)
4. New leader proposes extending from highQC

That's it. No complex proof collection. The QC mechanism
ensures safety automatically.
```

단순성은 **3단계 구조**에서 비롯된다: 명령이 커밋될 때(commitQC를 가질 때), 정보가 QC 체인에 "내재화"되어 view 변경 중에 손실될 수 없다.

### 4.6 실제 HotStuff: Meta의 Diem/Libra

Meta(구 Facebook)는 Diem 블록체인(원래 Libra)의 합의 프로토콜로 HotStuff를 선택했다. 그들의 구현 **DiemBFT**는 HotStuff를 다음으로 확장한다:

- 평판 기반 leader 로테이션
- 낙관적 fast path (모든 replica가 동의할 때 3단계 대신 2단계)
- view 동기화를 위한 Pacemaker

---

## 5. Tendermint BFT

### 5.1 개요

Tendermint(Buchman, 2016; 이후 Cosmos 블록체인에서 사용)는 블록체인 애플리케이션을 위해 특별히 설계된 BFT 합의 프로토콜이다.

### 5.2 프로토콜: Propose → Prevote → Precommit

```
Round r:
  1. PROPOSE: designated proposer broadcasts a block
  2. PREVOTE: each validator broadcasts prevote for the block (or nil)
  3. PRECOMMIT: if validator sees 2f+1 prevotes, broadcasts precommit (or nil)
  4. COMMIT: if validator sees 2f+1 precommits, commits the block

If any step times out, move to round r+1 with a different proposer.
```

### 5.3 Lock 메커니즘

Tendermint는 라운드 간 안전성 위반을 방지하기 위해 **lock 규칙**을 사용한다:

```python
class TendermintValidator:
    """Simplified Tendermint validator."""

    def __init__(self, validator_id, n_validators):
        self.id = validator_id
        self.n = n_validators
        self.f = (n_validators - 1) // 3
        self.locked_round = -1
        self.locked_value = None
        self.valid_round = -1
        self.valid_value = None

    def prevote(self, proposed_value, propose_round, valid_round_in_proposal):
        """Decide what to prevote.

        Lock rule: if we are locked on a value, we can only prevote
        for that value OR for a value that has a valid_round higher
        than our locked_round.
        """
        if self.locked_round == -1:
            # Not locked; prevote for proposal
            return proposed_value

        if proposed_value == self.locked_value:
            # Proposal matches our lock
            return proposed_value

        if valid_round_in_proposal > self.locked_round:
            # Proposal has evidence from a later round; unlock
            return proposed_value

        # Locked on a different value; prevote nil
        return None

    def precommit(self, value, prevote_count):
        """Decide what to precommit.

        If we see 2f+1 prevotes for a value, lock on it and precommit.
        """
        if prevote_count >= 2 * self.f + 1 and value is not None:
            self.locked_round = self.current_round
            self.locked_value = value
            return value

        return None  # precommit nil

    def on_commit(self, value, precommit_count):
        """Commit if we see 2f+1 precommits."""
        if precommit_count >= 2 * self.f + 1:
            return value  # committed!
        return None
```

### 5.4 Tendermint vs PBFT

| 속성 | PBFT | Tendermint |
|----------|------|-----------|
| 단계 | 3 (pre-prepare, prepare, commit) | 3 (propose, prevote, precommit) |
| 메시지 복잡도 | O(n²) | O(n²) |
| Leader 로테이션 | view 변경 시 (장애) | 매 라운드 (라운드 로빈) |
| Locking | prepared 인증서에 암시적 | 명시적 lock 규칙 |
| 설계 대상 | 일반 SMR | 블록체인 (블록 단위) |
| 구현 | 연구 프로토타입 | 프로덕션 (Cosmos, Binance Chain) |

---

## 6. BFT 프로토콜 비교

### 6.1 프로토콜 비교 표

| 속성 | PBFT | HotStuff | Tendermint | BFT-SMaRt |
|----------|------|----------|-----------|-----------|
| 연도 | 1999 | 2019 | 2016 | 2014 |
| 메시지 복잡도 | O(n²) | O(n) | O(n²) | O(n²) |
| 통신 | All-to-all | 별형 (leader 경유) | All-to-all | All-to-all |
| 단계 (정상) | 3 | 3 (파이프라인) | 3 | 3 |
| 지연 시간 (메시지) | 5 | 7 (파이프라인 3) | 5 | 5 |
| View change 비용 | O(n³) | O(n) | O(n²) | O(n²) |
| Threshold signature | 아니오 | 예 | 아니오 | 아니오 |
| Responsiveness | 예 | 예 | 아니오 (타임아웃 기반) | 예 |
| 프로덕션 사용 | 제한적 | Diem, Aptos | Cosmos, BSC | EBSI |

### 6.2 지연 시간 분석

```
PBFT (n=4, f=1):
  Client → Primary: 1 hop
  Pre-prepare → all: 1 hop
  Prepare → all: 1 hop
  Commit → all: 1 hop
  Reply → Client: 1 hop
  Total: 5 message delays

HotStuff (n=4, f=1):
  Client → Leader: 1 hop
  PREPARE → all + votes → Leader: 2 hops
  PRE-COMMIT → all + votes → Leader: 2 hops
  COMMIT → all + votes → Leader: 2 hops
  DECIDE → all: 1 hop
  Total: 7 message delays (but pipelined → 2 hops effective)
```

### 6.3 처리량 비교

LAN에서의 대략적 처리량 (4 replica, 배치 처리):

| 프로토콜 | 처리량 (ops/sec) | 지연 시간 (ms) |
|----------|---------------------|-------------|
| PBFT | 50,000 - 100,000 | 1 - 5 |
| HotStuff | 30,000 - 80,000 | 2 - 10 |
| Tendermint | 1,000 - 10,000 | 1,000 - 6,000 |
| BFT-SMaRt | 60,000 - 120,000 | 1 - 5 |

참고: Tendermint의 낮은 처리량은 의도적이다 — 라운드 간 필수 타임아웃과 함께 한 번에 하나의 블록을 커밋한다.

---

## 7. BFT vs Crash-Fault-Tolerant 프로토콜 사용 시점

### 7.1 결정 프레임워크

```
Do you control all nodes?
├── Yes → Are nodes in a trusted environment?
│   ├── Yes → Crash-fault tolerance (Raft/Paxos) is sufficient
│   └── No (e.g., edge computing) → Consider BFT
└── No → BFT required (e.g., multi-organization consortium)

Is performance critical?
├── Yes → Prefer Raft/Paxos (lower overhead)
└── No → BFT is acceptable

How many nodes?
├── < 10 → PBFT or BFT-SMaRt
├── 10 - 100 → HotStuff (linear complexity)
└── > 100 → Consider PoS/PoW hybrid (blockchain-style)
```

### 7.2 Byzantine 내성의 비용

| 측면 | Crash-Fault (Raft) | Byzantine (PBFT) | 비율 |
|--------|-------------------|------------------|-------|
| f 결함당 replica | 2f + 1 | 3f + 1 | 1.5배 |
| 연산당 메시지 | O(n) | O(n²) | O(n)배 |
| 암호학적 오버헤드 | 없음 | 모든 메시지에 서명 | 상당함 |
| View change 복잡도 | 단순 | 복잡 | 훨씬 어려움 |
| 구현 난이도 | 보통 | 매우 어려움 | 훨씬 어려움 |

---

## 8. 블록체인과의 연결

### 8.1 블록체인 합의 지형

```
Consensus Protocols
├── Crash-Fault Tolerant (CFT)
│   ├── Paxos
│   └── Raft
│
├── Byzantine-Fault Tolerant (BFT)
│   ├── Classical BFT (permissioned)
│   │   ├── PBFT
│   │   ├── HotStuff → Diem/Aptos
│   │   └── Tendermint → Cosmos
│   │
│   └── Sybil-Resistant (permissionless)
│       ├── Proof of Work (PoW) → Bitcoin, Ethereum (pre-merge)
│       ├── Proof of Stake (PoS) → Ethereum 2.0, Solana
│       └── BFT + PoS Hybrids → Algorand, Avalanche
```

### 8.2 BFT로서의 Proof of Work

Bitcoin의 Proof of Work는 Byzantine 합의의 변형을 해결한다:

- **Sybil 저항**: 새 "신원"(마이닝 파워) 생성에 실제 자원(전기)이 필요
- **확률적 최종성**: 블록은 ~6번의 확인 후 최종으로 간주(~60분)
- **결함 허용**: 최대 50% 악의적 해시 파워 허용(33%가 아님)
- **확장성**: 수천 명의 익명 참여자와 작동

```
BFT:     Deterministic, fast finality, small groups, known participants
PoW:     Probabilistic, slow finality, large groups, anonymous participants
```

### 8.3 Proof of Stake와 BFT 하이브리드

현대 PoS 블록체인은 종종 선택된 validator 집합 간에 BFT 스타일 합의를 사용한다:

```python
class PoSBFTHybrid:
    """Conceptual model of a PoS + BFT hybrid system."""

    def __init__(self):
        self.validators = {}  # address → stake
        self.committee_size = 100

    def select_committee(self, randomness_seed):
        """Select a committee weighted by stake.

        Higher stake → higher probability of selection.
        """
        total_stake = sum(self.validators.values())
        committee = []

        for address, stake in self.validators.items():
            probability = stake / total_stake
            if self._is_selected(address, randomness_seed, probability):
                committee.append(address)

            if len(committee) >= self.committee_size:
                break

        return committee

    def run_bft_consensus(self, committee, block):
        """Run BFT (e.g., HotStuff) among the selected committee.

        The committee is small enough for BFT's O(n) or O(n²)
        message complexity to be practical.
        """
        # HotStuff with committee as replicas
        pass

    def _is_selected(self, address, seed, probability):
        """Verifiable random function for committee selection."""
        h = hashlib.sha256(f"{address}{seed}".encode()).hexdigest()
        threshold = int(probability * (2**256))
        return int(h, 16) < threshold
```

| 시스템 | 합의 | BFT 구성 요소 | 최종성 |
|--------|----------|---------------|----------|
| Ethereum 2.0 | Gasper (PoS) | Casper FFG | ~15분 |
| Cosmos | Tendermint (PoS) | Tendermint BFT | ~6초 |
| Algorand | Pure PoS | BA* (BFT 변형) | ~4초 |
| Avalanche | Snowball | Metastable BFT | ~2초 |
| Aptos | DiemBFT (PoS) | HotStuff 변형 | ~1초 |

---

## 9. 코드: 간소화된 PBFT 시뮬레이터

```python
"""
Simplified PBFT Simulator

Demonstrates the three-phase commit process:
pre-prepare → prepare → commit

Supports configurable Byzantine nodes that can:
- Not respond (crash fault)
- Send conflicting messages (equivocation)
"""

import hashlib
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple
from collections import defaultdict
from enum import Enum


class MessageType(Enum):
    REQUEST = "REQUEST"
    PRE_PREPARE = "PRE-PREPARE"
    PREPARE = "PREPARE"
    COMMIT = "COMMIT"
    REPLY = "REPLY"


class ReplicaStatus(Enum):
    IDLE = "idle"
    PRE_PREPARED = "pre-prepared"
    PREPARED = "prepared"
    COMMITTED = "committed"
    EXECUTED = "executed"


@dataclass
class PBFTMsg:
    msg_type: MessageType
    view: int
    seq: int
    digest: str
    sender: int
    request: Optional[str] = None


class Replica:
    """A PBFT replica that processes messages and tracks protocol state."""

    def __init__(self, replica_id: int, n_replicas: int, is_byzantine: bool = False):
        self.id = replica_id
        self.n = n_replicas
        self.f = (n_replicas - 1) // 3
        self.view = 0
        self.is_byzantine = is_byzantine

        # Protocol state per (view, seq)
        self.status: Dict[Tuple[int, int], ReplicaStatus] = {}
        self.pre_prepare_log: Dict[Tuple[int, int], PBFTMsg] = {}
        self.prepare_log: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        self.commit_log: Dict[Tuple[int, int], Set[int]] = defaultdict(set)

        # Results
        self.executed_requests: List[str] = []
        self.message_count = 0

    @property
    def is_primary(self) -> bool:
        return self.id == self.view % self.n

    def receive(self, msg: PBFTMsg) -> List[PBFTMsg]:
        """Process a message and return outgoing messages."""
        self.message_count += 1

        if self.is_byzantine:
            return self._byzantine_behavior(msg)

        if msg.msg_type == MessageType.REQUEST:
            return self._handle_request(msg)
        elif msg.msg_type == MessageType.PRE_PREPARE:
            return self._handle_pre_prepare(msg)
        elif msg.msg_type == MessageType.PREPARE:
            return self._handle_prepare(msg)
        elif msg.msg_type == MessageType.COMMIT:
            return self._handle_commit(msg)

        return []

    def _handle_request(self, msg: PBFTMsg) -> List[PBFTMsg]:
        """Primary handles client request."""
        if not self.is_primary:
            return []

        key = (msg.view, msg.seq)
        self.pre_prepare_log[key] = msg
        self.status[key] = ReplicaStatus.PRE_PREPARED

        # Broadcast PRE-PREPARE to all backups
        outgoing = []
        for i in range(self.n):
            if i != self.id:
                outgoing.append(PBFTMsg(
                    MessageType.PRE_PREPARE, msg.view, msg.seq,
                    msg.digest, self.id, msg.request
                ))
        return outgoing

    def _handle_pre_prepare(self, msg: PBFTMsg) -> List[PBFTMsg]:
        """Backup handles PRE-PREPARE from primary."""
        key = (msg.view, msg.seq)

        # Validation
        if msg.view != self.view:
            return []
        if msg.sender != self.view % self.n:
            return []
        if key in self.pre_prepare_log:
            if self.pre_prepare_log[key].digest != msg.digest:
                return []  # conflicting pre-prepare

        # Accept and verify digest
        expected_digest = hashlib.sha256(
            (msg.request or "").encode()
        ).hexdigest()[:16]
        if msg.digest != expected_digest:
            return []

        self.pre_prepare_log[key] = msg
        self.status[key] = ReplicaStatus.PRE_PREPARED

        # Broadcast PREPARE to all replicas
        outgoing = []
        for i in range(self.n):
            if i != self.id:
                outgoing.append(PBFTMsg(
                    MessageType.PREPARE, msg.view, msg.seq,
                    msg.digest, self.id
                ))
        return outgoing

    def _handle_prepare(self, msg: PBFTMsg) -> List[PBFTMsg]:
        """Handle PREPARE message."""
        key = (msg.view, msg.seq)

        if msg.view != self.view:
            return []

        self.prepare_log[key].add(msg.sender)

        # Check if prepared: pre-prepare + 2f prepares
        if (key in self.pre_prepare_log and
                len(self.prepare_log[key]) >= 2 * self.f and
                self.status.get(key) == ReplicaStatus.PRE_PREPARED):

            self.status[key] = ReplicaStatus.PREPARED

            # Broadcast COMMIT to all replicas
            outgoing = []
            for i in range(self.n):
                if i != self.id:
                    outgoing.append(PBFTMsg(
                        MessageType.COMMIT, msg.view, msg.seq,
                        msg.digest, self.id
                    ))
            return outgoing

        return []

    def _handle_commit(self, msg: PBFTMsg) -> List[PBFTMsg]:
        """Handle COMMIT message."""
        key = (msg.view, msg.seq)

        if msg.view != self.view:
            return []

        self.commit_log[key].add(msg.sender)

        # Check if committed-local: 2f+1 commits
        if (len(self.commit_log[key]) >= 2 * self.f + 1 and
                self.status.get(key) in (
                    ReplicaStatus.PREPARED, ReplicaStatus.PRE_PREPARED)):

            self.status[key] = ReplicaStatus.COMMITTED

            # Execute the request
            if key in self.pre_prepare_log:
                request = self.pre_prepare_log[key].request
                if request:
                    self.executed_requests.append(request)
                self.status[key] = ReplicaStatus.EXECUTED

                # Send REPLY to client
                return [PBFTMsg(
                    MessageType.REPLY, msg.view, msg.seq,
                    msg.digest, self.id, request
                )]

        return []

    def _byzantine_behavior(self, msg: PBFTMsg) -> List[PBFTMsg]:
        """Byzantine replica: silently drop all messages."""
        return []


class PBFTSimulator:
    """Simulate PBFT consensus among a set of replicas."""

    def __init__(self, n_replicas: int = 4, n_byzantine: int = 0):
        assert n_replicas >= 3 * n_byzantine + 1, (
            f"Need n >= 3f+1: {n_replicas} < {3*n_byzantine+1}"
        )
        self.n = n_replicas
        self.f = (n_replicas - 1) // 3

        # Create replicas (Byzantine nodes are the last ones)
        self.replicas: Dict[int, Replica] = {}
        for i in range(n_replicas):
            is_byz = i >= n_replicas - n_byzantine
            self.replicas[i] = Replica(i, n_replicas, is_byz)

        self.message_queue: List[Tuple[int, PBFTMsg]] = []
        self.total_messages = 0
        self.replies: List[PBFTMsg] = []

    def submit_request(self, request: str, seq: int):
        """Client submits a request to the primary."""
        digest = hashlib.sha256(request.encode()).hexdigest()[:16]
        primary_id = 0  # view 0 → primary is replica 0

        req_msg = PBFTMsg(
            MessageType.REQUEST, 0, seq, digest, -1, request
        )

        # Send to primary
        self.message_queue.append((primary_id, req_msg))

    def run(self, max_rounds: int = 100) -> bool:
        """Process messages until consensus or max rounds."""
        rounds = 0

        while self.message_queue and rounds < max_rounds:
            rounds += 1
            # Process all current messages
            current_batch = self.message_queue[:]
            self.message_queue = []

            for dst_id, msg in current_batch:
                replica = self.replicas.get(dst_id)
                if replica is None:
                    continue

                outgoing = replica.receive(msg)
                self.total_messages += 1

                for out_msg in outgoing:
                    if out_msg.msg_type == MessageType.REPLY:
                        self.replies.append(out_msg)
                    else:
                        # Broadcast: determine destinations
                        for rid in range(self.n):
                            if rid != out_msg.sender:
                                self.message_queue.append((rid, out_msg))

        return len(self.replies) > 0

    def print_results(self, request: str):
        """Print simulation results."""
        print(f"\nRequest: \"{request}\"")
        print(f"Total messages exchanged: {self.total_messages}")
        print(f"Replies received: {len(self.replies)}")

        # Check agreement
        executed_values = set()
        for rid, replica in self.replicas.items():
            if replica.is_byzantine:
                print(f"  Replica {rid}: BYZANTINE (dropped all messages)")
            else:
                status = "executed" if replica.executed_requests else "no execution"
                print(f"  Replica {rid}: {status} "
                      f"(msgs processed: {replica.message_count})")
                for req in replica.executed_requests:
                    executed_values.add(req)

        if len(executed_values) == 0:
            print("\nResult: NO CONSENSUS (insufficient honest replicas or rounds)")
        elif len(executed_values) == 1:
            print(f"\nResult: CONSENSUS REACHED on \"{list(executed_values)[0]}\"")
        else:
            print(f"\nResult: SAFETY VIOLATION! Multiple values: {executed_values}")


def main():
    # Scenario 1: 4 replicas, 0 Byzantine (ideal case)
    print("=" * 60)
    print("Scenario 1: 4 replicas, 0 Byzantine faults")
    print("=" * 60)
    sim1 = PBFTSimulator(n_replicas=4, n_byzantine=0)
    sim1.submit_request("SET x=42", seq=1)
    sim1.run()
    sim1.print_results("SET x=42")

    # Scenario 2: 4 replicas, 1 Byzantine (maximum tolerated)
    print("\n" + "=" * 60)
    print("Scenario 2: 4 replicas, 1 Byzantine fault")
    print("=" * 60)
    sim2 = PBFTSimulator(n_replicas=4, n_byzantine=1)
    sim2.submit_request("SET y=100", seq=1)
    sim2.run()
    sim2.print_results("SET y=100")

    # Scenario 3: 7 replicas, 2 Byzantine
    print("\n" + "=" * 60)
    print("Scenario 3: 7 replicas, 2 Byzantine faults")
    print("=" * 60)
    sim3 = PBFTSimulator(n_replicas=7, n_byzantine=2)
    sim3.submit_request("TRANSFER 50 FROM A TO B", seq=1)
    sim3.run()
    sim3.print_results("TRANSFER 50 FROM A TO B")

    # Scenario 4: 4 replicas, primary is honest but 1 backup is Byzantine
    print("\n" + "=" * 60)
    print("Scenario 4: Message count analysis (4 replicas)")
    print("=" * 60)
    sim4 = PBFTSimulator(n_replicas=4, n_byzantine=0)
    sim4.submit_request("INCREMENT counter", seq=1)
    sim4.run()
    sim4.print_results("INCREMENT counter")
    print(f"\nTheoretical message count: ~2n² = {2 * 4**2}")
    print(f"Actual messages exchanged: {sim4.total_messages}")


if __name__ == "__main__":
    main()
```

### 9.1 예상 출력

```
============================================================
Scenario 1: 4 replicas, 0 Byzantine faults
============================================================

Request: "SET x=42"
Total messages exchanged: 49
Replies received: 4
  Replica 0: executed (msgs processed: 13)
  Replica 1: executed (msgs processed: 12)
  Replica 2: executed (msgs processed: 12)
  Replica 3: executed (msgs processed: 12)

Result: CONSENSUS REACHED on "SET x=42"

============================================================
Scenario 2: 4 replicas, 1 Byzantine fault
============================================================

Request: "SET y=100"
Total messages exchanged: 37
Replies received: 3
  Replica 0: executed (msgs processed: 10)
  Replica 1: executed (msgs processed: 10)
  Replica 2: executed (msgs processed: 10)
  Replica 3: BYZANTINE (dropped all messages)

Result: CONSENSUS REACHED on "SET y=100"
```

### 9.2 시뮬레이터 확장

1. **Byzantine equivocation**: 메시지를 드롭하는 대신, Byzantine replica가 서로 다른 replica에 다른 다이제스트를 보내게 한다. 정직한 replica가 여전히 합의에 도달하는지 확인한다.

2. **View change**: View change 프로토콜을 구현한다. Pre-prepare를 보내지 않는 Byzantine primary를 시뮬레이션하여 view change를 트리거한다.

3. **다중 요청**: 여러 순차 요청을 제출하고 모든 정직한 replica가 동일한 순서로 실행하는지 확인한다.

4. **성능 측정**: `n`의 함수로 메시지 복잡도를 측정하고 이론적 O(n²)와 비교한다.

---

## 10. 요약

Byzantine fault tolerance는 분산 시스템에서 가장 어려운 장애 유형을 다룬다: 악의적으로 행동할 수 있는 노드. 기본적인 불가능성 결과 — 합의는 `f`개의 Byzantine 결함을 허용하기 위해 최소 `3f + 1` 노드가 필요하다 — 는 모든 BFT 프로토콜의 이론적 기준선을 설정한다.

PBFT는 O(n²) 메시지 복잡도와 순서 지정(pre-prepare + prepare)과 커밋(commit)을 분리하는 3단계 프로토콜로 BFT를 실용적으로 만들었다. HotStuff는 threshold signature와 별형 통신 토폴로지를 사용하여 phase당 O(n) 메시지로 확장성을 향상시켰다. Tendermint는 명시적 lock 규칙과 라운드 로빈 리더십으로 BFT를 블록체인에 적응시켰다.

Crash-fault tolerance(Raft/Paxos)와 Byzantine fault tolerance 중 선택은 신뢰 모델에 따라 달라진다: 모든 노드가 동일한 관리 도메인에 있을 때 CFT로 충분하고, 다조직 컨소시엄이나 공개 블록체인처럼 노드가 적대적일 수 있을 때 BFT가 필요하다.

---

[다음: 분산 트랜잭션](./08_Distributed_Transactions.md)

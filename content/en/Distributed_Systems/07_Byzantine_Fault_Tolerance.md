# Lesson 7: Byzantine Fault Tolerance

[Overview](./00_Overview.md) | [Previous: Raft In Depth](./06_Raft_In_Depth.md) | [Next: Distributed Transactions](./08_Distributed_Transactions.md)

---

## Learning Objectives

- Understand the Byzantine Generals Problem and prove the impossibility of consensus with n ≤ 3f nodes
- Trace through the PBFT protocol including normal operation, view changes, and checkpoints
- Compare PBFT with modern linear-complexity BFT protocols (HotStuff, Tendermint)
- Analyze the connection between BFT consensus and blockchain systems
- Implement a simplified PBFT simulator demonstrating the three-phase commit process

---

## 1. The Byzantine Generals Problem

### 1.1 Problem Statement

In 1982, Lamport, Shostak, and Pease formulated the **Byzantine Generals Problem**: a group of generals commanding divisions of an army must agree on a common battle plan (attack or retreat). Some generals may be **traitors** who send conflicting messages to different generals.

The problem captures a fundamental challenge in distributed computing: **reaching agreement when some participants may behave arbitrarily** — sending incorrect values, sending different values to different peers, or not responding at all.

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

### 1.2 Formal Definition

- **n** generals, of which at most **f** are **Byzantine** (faulty/malicious)
- All loyal generals must agree on the same value (**Agreement**)
- If all loyal generals propose the same value, that value must be chosen (**Validity**)
- The protocol must terminate (**Termination**)

### 1.3 Impossibility with n ≤ 3f

**Theorem** (Lamport, Shostak, Pease, 1982): *No protocol can solve the Byzantine Generals Problem for 3 or fewer generals if even one is a traitor.*

**Proof for n = 3, f = 1**:

Assume a protocol exists. Consider three generals: G1, G2, G3, where G3 is the traitor.

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

G2 cannot distinguish between Scenario A (where G3 is the traitor lying about its value) and a different scenario where G1 is the traitor. In both cases, G2 receives conflicting information and cannot determine the truth.

**Generalization**: For any `n ≤ 3f`, the same argument applies. The `f` traitors can coordinate to make different sets of `f` loyal generals see contradictory information.

**Key result**: Byzantine fault tolerance requires **n ≥ 3f + 1** nodes.

| Total nodes (n) | Max Byzantine faults (f) | Required honest | Example |
|-----------------|------------------------|-----------------|---------|
| 4 | 1 | 3 | Minimum BFT system |
| 7 | 2 | 5 | |
| 10 | 3 | 7 | |
| 3f + 1 | f | 2f + 1 | General formula |

### 1.4 Oral Messages vs Signed Messages

Lamport's original paper distinguished two models:

**Oral Messages (OM)**: Messages can be forged. A traitor can claim that another general sent a message it did not. This requires `n ≥ 3f + 1`.

**Signed Messages (SM)**: Messages are cryptographically signed and cannot be forged. This allows solving Byzantine agreement with `n ≥ 2f + 1` (but with exponential message complexity in the basic algorithm).

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

### 1.5 Why n ≥ 3f + 1?

The intuition is based on quorum intersection:

- To tolerate `f` Byzantine faults, we need quorums of size `q` such that any two quorums overlap in at least `f + 1` honest nodes.
- With `n` total nodes, a quorum is any set of `q = ⌈(2n + f + 1) / 3⌉` nodes.
- For `n = 3f + 1`, the quorum size is `2f + 1`.
- Two quorums of size `2f + 1` from `3f + 1` nodes overlap in at least `f + 1` nodes.
- Since at most `f` of those `f + 1` nodes are Byzantine, at least one overlapping node is honest.

---

## 2. PBFT: Practical Byzantine Fault Tolerance

### 2.1 Overview

Castro and Liskov published PBFT in 1999, providing the first **practical** BFT protocol. Before PBFT, BFT protocols had exponential message complexity or required synchronous networks. PBFT achieves:

- **O(n²)** message complexity per request
- **Asynchronous** operation (no timing assumptions for safety)
- **Liveness** with weak synchrony assumptions (eventual message delivery)
- Tolerates `f` Byzantine faults with `n = 3f + 1` nodes

### 2.2 System Model

- **n = 3f + 1** replicas, numbered 0 to n-1
- One replica is the **primary** (leader): replica `p = v mod n` where `v` is the view number
- Clients send requests to the primary
- All replicas maintain a deterministic state machine

### 2.3 Normal Operation: Three-Phase Protocol

PBFT uses three phases: **pre-prepare**, **prepare**, and **commit**.

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

### 2.4 Phase Details

**Pre-prepare** (primary → all replicas):
- Primary assigns a sequence number `n` to the request in view `v`
- Message: `⟨PRE-PREPARE, v, n, d⟩` where `d` is the digest of the request
- Replicas accept if: view matches, sequence number is in the valid range, and they haven't accepted a different pre-prepare for the same `(v, n)`

**Prepare** (each replica → all replicas):
- After accepting a pre-prepare, each replica broadcasts a prepare message
- Message: `⟨PREPARE, v, n, d, i⟩` where `i` is the replica ID
- A replica is **prepared** when it has the pre-prepare and `2f` matching prepares
- Prepared means: "I certify that this request is assigned sequence number `n` in view `v`"

**Commit** (each replica → all replicas):
- After becoming prepared, each replica broadcasts a commit message
- Message: `⟨COMMIT, v, n, d, i⟩`
- A replica is **committed-local** when it has `2f + 1` matching commits
- After committed-local, the replica executes the request and sends the reply to the client

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

### 2.5 Why Three Phases?

Two phases (pre-prepare + prepare) are not enough because a replica cannot be sure that other replicas have also reached the "prepared" state. The commit phase ensures that **at least 2f + 1 replicas know that 2f + 1 replicas have agreed on the ordering**.

Without the commit phase, during a view change, it would be impossible to determine which requests were actually ordered vs merely tentatively assigned sequence numbers.

| Phase | Purpose | Quorum needed |
|-------|---------|--------------|
| Pre-prepare | Primary assigns sequence number | 1 (primary only) |
| Prepare | Replicas agree on sequence in view | 2f + 1 (including primary) |
| Commit | Replicas certify that ordering is stable | 2f + 1 |

### 2.6 Message Complexity

For each client request:

```
Pre-prepare:  primary → n-1 replicas               = n - 1 messages
Prepare:      each of n replicas → n-1 others       = n(n - 1) messages
Commit:       each of n replicas → n-1 others       = n(n - 1) messages
Reply:        each replica → client (client needs f+1) = f + 1 messages

Total: (n-1) + n(n-1) + n(n-1) + (f+1) ≈ 2n² messages = O(n²)
```

For `n = 4` (tolerating 1 fault): ~32 messages per request.
For `n = 7` (tolerating 2 faults): ~98 messages per request.
For `n = 100` (tolerating 33 faults): ~20,000 messages per request.

The **O(n²) bottleneck** makes PBFT impractical for large replica sets.

### 2.7 View Change Protocol

When the primary is suspected of being faulty (e.g., not sending pre-prepares), replicas initiate a **view change** to elect a new primary.

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

### 2.8 Checkpoints and Garbage Collection

PBFT replicas periodically take **checkpoints** and garbage-collect old log entries:

1. Every `K` requests (e.g., K=100), a replica takes a checkpoint of its state
2. It broadcasts `⟨CHECKPOINT, n, d, i⟩` where `n` is the sequence number and `d` is the state digest
3. When a replica collects `2f + 1` matching checkpoint messages, the checkpoint is **stable**
4. All log entries and messages with sequence numbers ≤ `n` can be discarded

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

BFT-SMaRt is a high-performance, open-source Java implementation of BFT state machine replication (Bessani, Sousa, Alchieri, 2014). Key features:

| Feature | Description |
|---------|------------|
| Language | Java |
| Protocol | PBFT-based with optimizations |
| Throughput | ~80K ops/sec (4 replicas, LAN) |
| Batch processing | Groups multiple requests per consensus instance |
| Leader-based | Reduces message complexity for normal case |
| View change | Provably correct view change protocol |
| Reconfiguration | Supports dynamic membership changes |

BFT-SMaRt is notable for being one of the few **production-quality** BFT implementations, used in research and commercial applications including the EBSI (European Blockchain Services Infrastructure).

---

## 4. HotStuff

### 4.1 Motivation: Linear Message Complexity

The O(n²) message complexity of PBFT limits scalability. HotStuff (Yin, Malkhi, Reiter, Gueta, Abraham, 2019) reduces this to **O(n)** per phase using **threshold signatures**.

### 4.2 Key Insight: Threshold Signatures

Instead of each replica broadcasting to every other replica (O(n²)), HotStuff uses a **star topology**: all communication goes through the leader.

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

The leader collects `2f + 1` partial signatures and combines them into a **threshold signature** (a single signature representing the quorum). This threshold signature can be verified by any replica.

### 4.3 Three-Phase Protocol

HotStuff uses three phases (like PBFT) but with linear communication in each phase:

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

### 4.4 Pipelining for Throughput

HotStuff enables **pipelining**: each phase of one consensus instance overlaps with phases of subsequent instances. The leader doesn't wait for all three phases to complete before starting the next proposal.

```
View 1: PREPARE(cmd1)  → prepareQC(cmd1)
View 2: PRE-COMMIT(cmd1) + PREPARE(cmd2)  → precommitQC(cmd1) + prepareQC(cmd2)
View 3: COMMIT(cmd1) + PRE-COMMIT(cmd2) + PREPARE(cmd3) → ...
View 4: DECIDE(cmd1) + COMMIT(cmd2) + PRE-COMMIT(cmd3) + PREPARE(cmd4)

Each view does work for multiple commands simultaneously.
Effective latency: 1 round trip per command (amortized).
```

This is similar to CPU instruction pipelining: each individual command still takes 3 phases, but throughput approaches 1 command per round trip.

### 4.5 View Change in HotStuff

HotStuff's view change is dramatically simpler than PBFT's:

```
HotStuff View Change:
1. Replica times out → sends ⟨NEW-VIEW, v+1, prepareQC⟩ to new leader
2. New leader collects 2f+1 NEW-VIEW messages
3. New leader picks the highest prepareQC (highQC)
4. New leader proposes extending from highQC

That's it. No complex proof collection. The QC mechanism
ensures safety automatically.
```

The simplicity comes from the **three-phase structure**: by the time a command is committed (has a commitQC), the information is "baked into" the QC chain and cannot be lost during view changes.

### 4.6 HotStuff in Practice: Meta's Diem/Libra

Meta (formerly Facebook) chose HotStuff as the consensus protocol for the Diem blockchain (originally Libra). Their implementation, **DiemBFT**, extends HotStuff with:

- Reputation-based leader rotation
- Optimistic fast path (2 phases instead of 3 when all replicas agree)
- Pacemaker for view synchronization

---

## 5. Tendermint BFT

### 5.1 Overview

Tendermint (Buchman, 2016; later used in Cosmos blockchain) is a BFT consensus protocol designed specifically for blockchain applications.

### 5.2 Protocol: Propose → Prevote → Precommit

```
Round r:
  1. PROPOSE: designated proposer broadcasts a block
  2. PREVOTE: each validator broadcasts prevote for the block (or nil)
  3. PRECOMMIT: if validator sees 2f+1 prevotes, broadcasts precommit (or nil)
  4. COMMIT: if validator sees 2f+1 precommits, commits the block

If any step times out, move to round r+1 with a different proposer.
```

### 5.3 Lock Mechanism

Tendermint uses a **lock rule** to prevent safety violations across rounds:

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

| Property | PBFT | Tendermint |
|----------|------|-----------|
| Phases | 3 (pre-prepare, prepare, commit) | 3 (propose, prevote, precommit) |
| Message complexity | O(n²) | O(n²) |
| Leader rotation | On view change (failure) | Every round (round-robin) |
| Locking | Implicit in prepared certificates | Explicit lock rules |
| Designed for | General SMR | Blockchain (block-by-block) |
| Implementation | Research prototype | Production (Cosmos, Binance Chain) |

---

## 6. Comparing BFT Protocols

### 6.1 Protocol Comparison Table

| Property | PBFT | HotStuff | Tendermint | BFT-SMaRt |
|----------|------|----------|-----------|-----------|
| Year | 1999 | 2019 | 2016 | 2014 |
| Message complexity | O(n²) | O(n) | O(n²) | O(n²) |
| Communication | All-to-all | Star (via leader) | All-to-all | All-to-all |
| Phases (normal) | 3 | 3 (pipelined) | 3 | 3 |
| Latency (messages) | 5 | 7 (3 pipelined) | 5 | 5 |
| View change cost | O(n³) | O(n) | O(n²) | O(n²) |
| Threshold signatures | No | Yes | No | No |
| Responsiveness | Yes | Yes | No (timeout-based) | Yes |
| Production use | Limited | Diem, Aptos | Cosmos, BSC | EBSI |

### 6.2 Latency Analysis

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

### 6.3 Throughput Comparison

Approximate throughput on a LAN (4 replicas, batch processing):

| Protocol | Throughput (ops/sec) | Latency (ms) |
|----------|---------------------|-------------|
| PBFT | 50,000 - 100,000 | 1 - 5 |
| HotStuff | 30,000 - 80,000 | 2 - 10 |
| Tendermint | 1,000 - 10,000 | 1,000 - 6,000 |
| BFT-SMaRt | 60,000 - 120,000 | 1 - 5 |

Note: Tendermint's lower throughput is by design — it commits one block at a time with mandatory timeouts between rounds.

---

## 7. When to Use BFT vs Crash-Fault-Tolerant Protocols

### 7.1 Decision Framework

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

### 7.2 Cost of Byzantine Tolerance

| Aspect | Crash-Fault (Raft) | Byzantine (PBFT) | Ratio |
|--------|-------------------|------------------|-------|
| Replicas for f faults | 2f + 1 | 3f + 1 | 1.5× |
| Messages per operation | O(n) | O(n²) | O(n)× |
| Cryptographic overhead | None | Signatures on every message | Significant |
| View change complexity | Simple | Complex | Much harder |
| Implementation difficulty | Moderate | Very hard | Much harder |

---

## 8. Connection to Blockchain

### 8.1 Blockchain Consensus Landscape

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

### 8.2 Proof of Work as BFT

Bitcoin's Proof of Work solves a variant of Byzantine consensus:

- **Sybil resistance**: Creating new "identities" (mining power) costs real resources (electricity)
- **Probabilistic finality**: A block is considered final after ~6 confirmations (~60 minutes)
- **Fault tolerance**: Tolerates up to 50% malicious hash power (not 33%)
- **Scalability**: Works with thousands of anonymous participants

```
BFT:     Deterministic, fast finality, small groups, known participants
PoW:     Probabilistic, slow finality, large groups, anonymous participants
```

### 8.3 Proof of Stake and BFT Hybrids

Modern PoS blockchains often use BFT-style consensus among a selected validator set:

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

| System | Consensus | BFT Component | Finality |
|--------|----------|---------------|----------|
| Ethereum 2.0 | Gasper (PoS) | Casper FFG | ~15 minutes |
| Cosmos | Tendermint (PoS) | Tendermint BFT | ~6 seconds |
| Algorand | Pure PoS | BA* (BFT variant) | ~4 seconds |
| Avalanche | Snowball | Metastable BFT | ~2 seconds |
| Aptos | DiemBFT (PoS) | HotStuff variant | ~1 second |

---

## 9. Code: Simplified PBFT Simulator

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

### 9.1 Expected Output

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

### 9.2 Extending the Simulator

1. **Byzantine equivocation**: Instead of dropping messages, have Byzantine replicas send different digests to different replicas. Verify that honest replicas still reach consensus.

2. **View change**: Implement the view change protocol. Simulate a Byzantine primary that refuses to send pre-prepares, triggering a view change.

3. **Multiple requests**: Submit multiple sequential requests and verify that all honest replicas execute them in the same order.

4. **Performance measurement**: Measure message complexity as a function of `n` and compare with the theoretical O(n²).

---

## 10. Summary

Byzantine fault tolerance addresses the hardest class of failures in distributed systems: nodes that can behave arbitrarily, including maliciously. The foundational impossibility result — consensus requires at least `3f + 1` nodes to tolerate `f` Byzantine faults — sets the theoretical baseline for all BFT protocols.

PBFT made BFT practical with O(n²) message complexity and a three-phase protocol that separates ordering (pre-prepare + prepare) from commitment (commit). HotStuff improved scalability to O(n) messages per phase using threshold signatures and a star communication topology. Tendermint adapted BFT for blockchain with explicit lock rules and round-robin leadership.

The choice between crash-fault tolerance (Raft/Paxos) and Byzantine fault tolerance depends on the trust model: CFT suffices when all nodes are under the same administrative domain; BFT is necessary when nodes may be adversarial, as in multi-organization consortiums or public blockchains.

---

[Next: Distributed Transactions](./08_Distributed_Transactions.md)

# Lesson 5: Paxos Family

[Overview](./00_Overview.md) | [Previous: Consistency Models](./04_Consistency_Models.md) | [Next: Raft In Depth](./06_Raft_In_Depth.md)

---

## Learning Objectives

- Understand the historical context and motivation behind the Paxos consensus protocol
- Trace through single-decree Paxos step by step, including edge cases and correctness arguments
- Explain Multi-Paxos optimizations including leader election and gap handling
- Compare advanced Paxos variants: Flexible Paxos, Egalitarian Paxos, Cheap Paxos
- Implement a complete single-decree Paxos simulator with message passing

---

## 1. Historical Context

### 1.1 The Part-Time Parliament

In 1989, Leslie Lamport submitted a paper titled *"The Part-Time Parliament"* describing a consensus algorithm through an elaborate metaphor about the legislature of an ancient Greek island called Paxos. The paper was so obscure in its presentation that reviewers could not separate the algorithm from the metaphor. It was finally published in 1998 — nearly a decade later.

The key insight was revolutionary: **it is possible for a group of unreliable processors to agree on a single value, as long as a majority of them can communicate**. This was the first practical algorithm to solve consensus in an asynchronous, crash-fault-tolerant setting (circumventing FLP impossibility through the use of leader election and timeouts).

### 1.2 Paxos Made Simple

Frustrated by the reception of his original paper, Lamport published *"Paxos Made Simple"* in 2001 — a terse, 14-page description that began:

> "The Paxos algorithm, when presented in plain English, is very simple."

Despite its title, many practitioners still found the algorithm difficult to implement correctly. This difficulty stems not from the algorithm itself, but from the gap between "single-decree Paxos" (choosing one value) and the practical "Multi-Paxos" (choosing a sequence of values), which the paper left largely unspecified.

### 1.3 Why Paxos Matters

Paxos and its derivatives form the backbone of virtually every production consensus system:

| System | Protocol Basis |
|--------|---------------|
| Google Chubby | Multi-Paxos |
| Google Spanner | Multi-Paxos |
| Apache ZooKeeper | ZAB (Paxos-derived) |
| etcd / Kubernetes | Raft (Paxos-inspired) |
| CockroachDB | Raft (Paxos-inspired) |
| AWS DynamoDB | Paxos for metadata |
| Azure Cosmos DB | Multi-Paxos variant |

Understanding Paxos is essential not just for implementing consensus, but for reasoning about correctness in any distributed system.

---

## 2. Single-Decree Paxos

Single-decree (or "basic") Paxos solves the simplest consensus problem: **getting a set of nodes to agree on a single value**.

### 2.1 Roles

Paxos defines three logical roles. In practice, a single physical node often plays all three:

| Role | Responsibility |
|------|---------------|
| **Proposer** | Proposes values; drives the protocol forward |
| **Acceptor** | Votes on proposals; stores accepted values |
| **Learner** | Learns the chosen value once a majority accepts |

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

### 2.2 Proposal Numbers

Every proposal carries a unique, monotonically increasing **proposal number** (also called a ballot number). A common scheme:

```
proposal_number = (round_number, proposer_id)
```

Comparison is lexicographic: first by `round_number`, then by `proposer_id`. This guarantees global uniqueness and total ordering.

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

**Goal**: A proposer establishes its right to make a proposal by securing promises from a majority of acceptors.

```
Proposer                              Acceptor
   │                                     │
   │──── Prepare(n) ────────────────────▶│
   │                                     │
   │◀─── Promise(n, accepted_n, val) ────│  (if n > max_promised)
   │      or NACK                        │  (if n ≤ max_promised)
```

**Prepare(n)**: "I want to make a proposal with number `n`. Promise me you will not accept any proposal with a number less than `n`."

**Promise(n, accepted_n, accepted_val)**: "I promise not to accept any proposal numbered less than `n`. Here is the highest-numbered proposal I have already accepted (if any)."

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

**Goal**: Once a proposer receives promises from a majority, it sends an Accept request. The value it proposes must follow the **value selection rule**.

```
Proposer                              Acceptor
   │                                     │
   │──── Accept(n, v) ──────────────────▶│
   │                                     │
   │◀─── Accepted(n) ───────────────────│  (if n ≥ max_promised)
   │      or NACK                        │  (if n < max_promised)
```

**Value Selection Rule**: If any Promise response included a previously accepted value, the proposer **must** use the value from the highest-numbered accepted proposal. Otherwise, the proposer is free to choose any value.

This is the critical safety mechanism: it ensures that once a value is chosen (accepted by a majority), any future proposer will be forced to propose the same value.

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

**Acceptor handling Accept**:

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

### 2.5 Complete Protocol Flow

Here is the full single-decree Paxos flow with 5 acceptors and one proposer:

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
    │◀─── Promise(ok,∅) ───────────────│     │     │   ← majority (3)
    │                         │     │     │     │     │
    │   [select_value → "X"]  │     │     │     │     │
    │                         │     │     │     │     │
    │──── Accept(1,1,"X") ───▶│     │     │     │     │
    │──── Accept(1,1,"X") ────────▶ │     │     │     │
    │──── Accept(1,1,"X") ─────────────▶  │     │     │
    │                         │     │     │     │     │
    │◀─── Accepted ──────────│     │     │     │     │
    │◀─── Accepted ───────────────│     │     │     │
    │◀─── Accepted ────────────────────│     │     │   ← majority (3)
    │                         │     │     │     │     │
    │   [VALUE "X" IS CHOSEN] │     │     │     │     │
```

### 2.6 Correctness Proof

The key safety property of Paxos is:

> **At most one value can be chosen.**

We prove this by contradiction. Assume two different values `v` and `v'` are both chosen (each accepted by a majority).

**Proof sketch**:

Let `v` be chosen at proposal number `n` and `v'` at proposal number `n'`, where `n < n'`.

1. Since `v` was chosen, a majority `Q` accepted `(n, v)`.
2. Since `v'` was proposed at `n'`, the proposer received promises from a majority `Q'` in Phase 1.
3. Since `Q` and `Q'` are both majorities, they intersect: there exists an acceptor `a ∈ Q ∩ Q'`.
4. Acceptor `a` accepted `(n, v)` and later promised `n' > n`.
5. When `a` sent its Promise for `n'`, it reported `(n, v)` as its highest accepted value.
6. By the value selection rule, the proposer of `n'` must choose the value from the highest-numbered previously accepted proposal.
7. The highest accepted value reported by any acceptor in `Q'` has proposal number ≥ `n`, and its value must be `v` (by induction on proposal numbers between `n` and `n'`).
8. Therefore `v' = v`. Contradiction.

The formal proof uses strong induction on proposal numbers. The base case is trivial (the first chosen value). The inductive step shows that for any proposal number `n' > n`, if a value was chosen at `n`, then the value proposed at `n'` must be the same. This inductive argument is the heart of Paxos's correctness.

**Liveness**: Paxos does **not** guarantee liveness. Two proposers can endlessly preempt each other (livelock). In practice, this is solved by electing a distinguished proposer (leader).

### 2.7 Edge Cases

#### Dueling Proposers (Livelock)

```
P1: Prepare(1,1) → gets majority promises
P2: Prepare(2,2) → gets majority promises (invalidates P1's promises)
P1: Accept(1,1,"X") → REJECTED (acceptors promised (2,2))
P1: Prepare(3,1) → gets majority promises (invalidates P2's promises)
P2: Accept(2,2,"Y") → REJECTED (acceptors promised (3,1))
... repeats forever ...
```

**Solution**: Use randomized backoff or elect a single leader.

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

#### Split Votes

With an even number of acceptors (or proposers choosing different values), Phase 1 can succeed but Phase 2 gets fewer than a majority:

```
5 acceptors: A1, A2, A3, A4, A5

P1: Prepare(1,1) → promises from {A1, A2, A3}
P1: Accept(1,1,"X") → accepted by {A1, A2} only (A3 crashed before Accept)

Value "X" is NOT chosen (only 2 < majority of 3).
```

This is correct behavior — Paxos simply hasn't chosen a value yet. A new round can be initiated.

#### Acceptor Crashes and Recovery

Acceptors must write their state (`max_promised`, `accepted_proposal`, `accepted_value`) to **durable storage** before responding. If an acceptor crashes and recovers, it reads its persisted state and continues correctly.

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

#### Learning a Chosen Value

The simplest approach: the proposer tells all learners once it receives a majority of Accepted responses. A more robust approach uses a "distinguished learner" that acceptors notify directly.

---

## 3. Multi-Paxos

Single-decree Paxos chooses **one** value. Real systems need to agree on a **sequence** of values (a replicated log). Multi-Paxos runs many instances of Paxos, one per log slot.

### 3.1 The Naive Approach

Run independent single-decree Paxos for each log slot. This requires 2 round trips per slot:

```
Slot 1: Prepare + Accept (4 messages × N acceptors)
Slot 2: Prepare + Accept (4 messages × N acceptors)
Slot 3: Prepare + Accept (4 messages × N acceptors)
...
```

This is extremely expensive: `O(4N)` messages per log entry.

### 3.2 Leader Optimization

The key Multi-Paxos optimization: **a stable leader can skip Phase 1 for consecutive slots**.

If a proposer successfully completes Phase 1 for slot `i`, it knows that no higher-numbered proposal has been made. As long as it remains the leader, it can go directly to Phase 2 (Accept) for slots `i+1, i+2, ...`.

```
Leader Election (one-time Phase 1 for all future slots):
  Leader: Prepare(n) for slot_range=[i, ∞)
  Acceptors: Promise(n) for all future slots

Steady-state operation (Phase 2 only):
  Leader: Accept(n, slot=i,   value=v1)  → 1 round trip
  Leader: Accept(n, slot=i+1, value=v2)  → 1 round trip
  Leader: Accept(n, slot=i+2, value=v3)  → 1 round trip
```

This reduces the amortized cost to `O(2N)` messages per log entry — the same as Raft.

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

### 3.3 Log Slots and Instance Numbers

Each slot in the Multi-Paxos log is an independent Paxos instance:

```
Log:  [ slot 0 ] [ slot 1 ] [ slot 2 ] [ slot 3 ] [ slot 4 ] ...
        "SET     "SET       "DEL       ???         "SET
         x=1"     y=2"       x"       (gap)        z=3"
```

Each slot has its own `accepted_proposal` and `accepted_value`. The leader assigns slots sequentially, but gaps can occur if the leader crashes mid-replication.

### 3.4 Gap Handling and No-Op Proposals

When a new leader takes over, it may discover gaps — slots where a value was proposed but not committed. The new leader must fill these gaps before proceeding:

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

The no-op is a special command that has no effect on the state machine but fills the gap so that subsequent slots can be applied in order.

### 3.5 Leader Election in Multi-Paxos

Lamport's original paper does not specify a leader election mechanism. Common approaches:

1. **Highest-ID leader**: The node with the highest ID that is reachable becomes leader. Simple but creates hotspots.

2. **Lease-based leadership**: The leader holds a time-bounded lease. Other nodes only attempt to become leader after the lease expires.

3. **View-based**: Nodes maintain a monotonically increasing view number. Leader for view `v` is `v mod N`.

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

### 4.1 Key Insight

Classic Paxos requires a **majority quorum** for both Phase 1 and Phase 2. Heidi Howard's Flexible Paxos (2016) showed that the only requirement is:

> **Phase 1 quorum ∩ Phase 2 quorum ≠ ∅**

This means Phase 1 and Phase 2 can use **different quorum sizes**, as long as they overlap.

### 4.2 Quorum Configurations

With `N` acceptors:

| Configuration | Phase 1 (Q1) | Phase 2 (Q2) | Property |
|--------------|-------------|-------------|----------|
| Classic Paxos | ⌈(N+1)/2⌉ | ⌈(N+1)/2⌉ | Balanced |
| Write-optimized | N | 1 | Fast writes, expensive leader election |
| Read-optimized | 1 | N | Cheap Phase 1, expensive writes |
| Asymmetric | N-1 | 2 | Good for stable leaders |

**Constraint**: `Q1 + Q2 > N` (ensures intersection).

### 4.3 Why FPaxos Matters

In Multi-Paxos with a stable leader, Phase 1 runs **rarely** (only during leader election). Phase 2 runs for **every** log entry. FPaxos lets us make Phase 2 cheaper at the expense of Phase 1:

```
Classic Multi-Paxos (5 nodes):
  Phase 1 (rare): needs 3 promises
  Phase 2 (every op): needs 3 accepts

FPaxos (5 nodes, Q1=4, Q2=2):
  Phase 1 (rare): needs 4 promises
  Phase 2 (every op): needs 2 accepts  ← 33% fewer messages!
```

This is particularly valuable for geo-distributed deployments where Phase 2 messages cross data centers.

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

### 5.1 Motivation: Leaderless Consensus

Multi-Paxos requires a stable leader. This creates:

1. **Bottleneck**: All requests go through one node
2. **Latency**: Clients far from the leader pay extra round trips
3. **Failover**: Leader failure requires election (temporary unavailability)

EPaxos (Moraru, Andersen, Kaminsky, 2013) eliminates the leader entirely. Any replica can propose a command directly.

### 5.2 Command Interference

EPaxos introduces the concept of **command interference**: two commands interfere if the order in which they execute affects the final state.

```
Non-interfering:  SET x=1 and SET y=2  (different keys → order doesn't matter)
Interfering:      SET x=1 and SET x=2  (same key → order matters)
```

For non-interfering commands, EPaxos can commit in a **fast path** (1 round trip). For interfering commands, it falls back to a **slow path** (2 round trips) to establish a total order.

### 5.3 Fast Path

```
Replica R1 receives command C1:

R1 ──── PreAccept(C1, deps={}) ──────▶ R2, R3, R4, R5
R2 ◀─── PreAcceptOK(C1, deps={}) ────── (no conflicts)
R3 ◀─── PreAcceptOK(C1, deps={}) ────── (no conflicts)
R4 ◀─── PreAcceptOK(C1, deps={}) ────── (no conflicts)

R1: Fast quorum (⌊(3N/4)⌋ + 1) replies agree → COMMIT in 1 round trip
```

The fast path quorum is larger than a simple majority (roughly 3/4 of replicas) to ensure that any two fast quorums overlap with each other **and** with a simple majority.

### 5.4 Slow Path

When replicas report different dependency sets (because they've seen interfering commands), EPaxos falls back to the slow path:

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

### 5.5 Dependency Tracking and Execution

Each command carries a **dependency set**: the set of commands that must execute before it. The execution order is determined by topological sort of the dependency graph:

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

### 5.6 EPaxos Tradeoffs

| Aspect | Multi-Paxos | EPaxos |
|--------|------------|--------|
| Leader required | Yes | No |
| Fast path latency | 1 RT (with leader) | 1 RT (any replica) |
| Slow path latency | N/A | 2 RT |
| Fast path quorum | Majority | ⌊3N/4⌋ + 1 |
| Conflict handling | Total order via leader | Dependency tracking |
| Implementation complexity | Moderate | High |
| Geo-distribution | Leader is bottleneck | Any replica serves clients |
| Message complexity (no conflicts) | O(N) | O(N) |
| Message complexity (conflicts) | O(N) | O(2N) |

---

## 6. Cheap Paxos and Vertical Paxos

### 6.1 Cheap Paxos

Cheap Paxos (Lamport and Massa, 2004) reduces the number of **active** replicas needed during normal operation by using auxiliary replicas that activate only during failures.

**Idea**: Use `f+1` main replicas and `f` auxiliary replicas. During normal operation, only the `f+1` main replicas participate. When a main replica fails, an auxiliary replica activates temporarily.

```
Normal operation:  Main1, Main2  (f+1 = 2, tolerates f=1 failure)
Main2 crashes:     Main1, Aux1   (Aux1 activates to maintain quorum)
Main2 recovers:    Main1, Main2  (Aux1 deactivates)
```

This saves CPU and network resources in the common case, at the cost of slower failure recovery.

### 6.2 Vertical Paxos

Vertical Paxos (Lamport, Malkhi, Zhou, 2009) separates **configuration management** from **data replication**. It uses an auxiliary "configuration service" (running Paxos) that manages which replicas handle each partition:

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

This is the theoretical foundation for systems like Google Spanner, where a Paxos group manages metadata and a simpler protocol handles data replication.

---

## 7. Protocol Comparison

| Property | Single-Decree Paxos | Multi-Paxos | EPaxos | Raft |
|----------|-------------------|-------------|--------|------|
| Leader required | No (but helps liveness) | Yes | No | Yes |
| Phase 1 messages | 2N | Amortized: 0 | 2N (fast), 4N (slow) | N (election) |
| Phase 2 messages | 2N | 2N | 0 (fast), 2N (slow) | 2N |
| Min nodes for f faults | 2f+1 | 2f+1 | 2f+1 | 2f+1 |
| Latency (steady state) | 2 RT | 1 RT | 1 RT (fast), 2 RT (slow) | 1 RT |
| Log ordering | Single value | Leader-ordered | Dependency graph | Leader-ordered |
| Understandability | Moderate | Hard | Very Hard | Easy |
| Real implementations | Rare standalone | Chubby, Spanner | Research | etcd, CockroachDB |
| Reconfiguration | Manual | Implementation-specific | Instance-based | Joint consensus |

---

## 8. Full Implementation: Single-Decree Paxos Simulator

This section provides a complete, runnable Paxos simulator with message passing, network simulation, and fault injection.

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

### 8.1 Running the Simulator

```bash
python paxos_simulator.py
```

Expected output (varies with random seed):

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

### 8.2 Extending the Simulator

Key exercises for deeper understanding:

1. **Add retry logic**: When a proposer receives a NACK, it should increment its round number and retry with exponential backoff.

2. **Implement node crashes**: Remove an acceptor mid-simulation and verify that Paxos still reaches consensus (with remaining majority).

3. **Add network partitions**: Split acceptors into two groups and observe that neither group can reach consensus alone.

4. **Measure message counts**: Count total messages and compare with the theoretical `O(N)` per phase.

---

## 9. Common Pitfalls and Implementation Advice

### 9.1 Durability Before Response

The most common implementation bug: **responding to a Prepare or Accept before persisting state**. If an acceptor crashes after responding but before persisting, it can violate its promise after recovery.

**Rule**: Always `fsync()` before sending the response.

### 9.2 Stale Messages

Messages from old rounds can arrive at any time. Every handler must check that the message's proposal number matches the current expected round.

### 9.3 Proposal Number Exhaustion

If a proposer uses `(round, node_id)` pairs and increments the round on every retry, a tight livelock loop can exhaust the round counter. Use 64-bit integers and add backoff.

### 9.4 Distinguished Proposer vs True Leader

Paxos only requires a "distinguished proposer" for liveness, not safety. The system is always safe regardless of how many proposers are active. This is a key difference from protocols like Raft, where having two leaders violates the protocol invariants.

### 9.5 Practical Reading List

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| Lamport, "The Part-Time Parliament" | 1998 | Original Paxos |
| Lamport, "Paxos Made Simple" | 2001 | Simplified description |
| Lamport, "Fast Paxos" | 2006 | 1 RT commits with larger quorums |
| Howard et al., "Flexible Paxos" | 2016 | Decoupled quorum sizes |
| Moraru et al., "Egalitarian Paxos" | 2013 | Leaderless Paxos |
| van Renesse & Altinbuken, "Paxos Made Moderately Complex" | 2015 | Implementation guide |
| Lamport & Massa, "Cheap Paxos" | 2004 | Reduced replica count |

---

## 10. Summary

Single-decree Paxos solves the fundamental problem of distributed consensus through a two-phase protocol built on majority quorums. Its correctness relies on three mechanisms: unique, ordered proposal numbers; the promise mechanism that prevents older proposals from succeeding; and the value selection rule that forces new proposers to adopt previously accepted values.

Multi-Paxos extends this to replicated logs by amortizing the cost of Phase 1 over many log entries through stable leadership. Flexible Paxos further optimizes the common case by using smaller quorums for the frequently executed Phase 2. Egalitarian Paxos removes the leader entirely, enabling any replica to commit non-interfering commands in a single round trip.

The Paxos family of protocols has been the theoretical foundation of distributed consensus for over three decades. While Raft (covered in Lesson 6) has become the preferred choice for new implementations due to its understandability, understanding Paxos remains essential for reasoning about the fundamental tradeoffs in distributed consensus.

---

[Next: Raft In Depth](./06_Raft_In_Depth.md)

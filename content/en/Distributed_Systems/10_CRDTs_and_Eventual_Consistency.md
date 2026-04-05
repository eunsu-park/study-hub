# Lesson 10: CRDTs and Eventual Consistency

[Overview](./00_Overview.md) | [Previous: Replication Strategies](./09_Replication_Strategies.md) | [Next: Partitioning and Sharding](./11_Partitioning_and_Sharding.md)

---

## Learning Objectives

- Formalize eventual consistency, convergence, and strong eventual consistency (SEC) as distinct guarantees
- Understand the mathematical foundations of CRDTs: join-semilattices, commutativity, associativity, and idempotence
- Implement state-based CRDTs (G-Counter, PN-Counter, G-Set, 2P-Set, LWW-Register, OR-Set) from scratch
- Compare state-based vs operation-based CRDTs and analyze their trade-offs
- Evaluate CRDT limitations and identify practical applications in production systems

---

## 1. The Problem: Concurrent Updates Without Coordination

In a replicated system with multiple writers (multi-leader or leaderless), concurrent updates to the same data are inevitable. From Lesson 9, we know this creates conflicts. The traditional solutions — consensus protocols or single-leader serialization — add latency and reduce availability.

**Can we design data structures that automatically converge without coordination?**

This is the central question that CRDTs (Conflict-free Replicated Data Types) answer.

### 1.1 A Motivating Example

Consider a collaborative document with a "like counter." Three replicas independently process increments:

```
Time ──────────────────────────────────────────────►

Replica A:  counter=0  ──► +1 ──► counter=1
Replica B:  counter=0  ──► +1 ──► +1 ──► counter=2
Replica C:  counter=0  ──► +1 ──► counter=1

Merging: What should the final count be?
  - Simple addition: 1 + 2 + 1 = 4?  (WRONG — double-counts the initial 0)
  - Max: max(1, 2, 1) = 2?           (WRONG — loses A and C increments)
  - ???: The correct answer is 4 total increments
```

A naive integer counter cannot be correctly merged. We need a data structure that *records enough information* to merge without losing updates or double-counting.

### 1.2 Why Not Just Use Consensus?

| Approach | Latency | Availability | Complexity |
|---|---|---|---|
| Consensus (Paxos/Raft) | High (multiple round trips) | Requires majority | High |
| Single-leader | Medium (leader round trip) | Leader is SPOF | Medium |
| CRDTs | None (local operations) | Always available | Medium (design-time) |

CRDTs shift complexity from **runtime coordination** to **data structure design**. Operations are applied locally with zero network round trips. Replicas synchronize asynchronously and are *mathematically guaranteed* to converge.

---

## 2. Formalizing Eventual Consistency

Before diving into CRDTs, we must precisely define what "eventual consistency" means. The term is often used loosely; here we distinguish three levels.

### 2.1 Basic Eventual Consistency

**Definition**: If no new updates are made, *eventually* all replicas will converge to the same value.

This is a weak guarantee:
- It says nothing about *how long* convergence takes
- During convergence, replicas may return different values
- It does not guarantee *which* value they converge to

### 2.2 Strong Convergence

**Definition**: Replicas that have received the *same set of updates* are in the *same state*, regardless of the order in which updates were received.

This is strictly stronger: it guarantees convergence is *deterministic* and depends only on the *set* of updates, not their ordering.

### 2.3 Strong Eventual Consistency (SEC)

**Definition**: A system provides SEC if:
1. **Eventual delivery**: Every update applied at one correct replica is eventually applied at all correct replicas
2. **Strong convergence**: Replicas that have received the same set of updates are in the same state
3. **Termination**: All operations complete in finite time (no blocking)

SEC is the guarantee provided by CRDTs. It is stronger than basic eventual consistency but weaker than linearizability or sequential consistency.

```
Strength hierarchy:

  Linearizability (strongest)
       │
  Sequential Consistency
       │
  Causal Consistency
       │
  Strong Eventual Consistency (SEC)  ← CRDTs guarantee this
       │
  Basic Eventual Consistency
       │
  No consistency (weakest)
```

---

## 3. Mathematical Foundations

CRDTs achieve SEC by constraining the merge operation to be a **join** in a **join-semilattice**.

### 3.1 Join-Semilattice

A **join-semilattice** is a set S with a binary operation ⊔ (join/merge) that satisfies:

| Property | Definition | Intuition |
|---|---|---|
| **Commutativity** | a ⊔ b = b ⊔ a | Order of merging doesn't matter |
| **Associativity** | (a ⊔ b) ⊔ c = a ⊔ (b ⊔ c) | Grouping of merges doesn't matter |
| **Idempotence** | a ⊔ a = a | Merging the same state twice has no effect |

These three properties together guarantee:
- Any permutation of merge operations produces the same result (commutativity + associativity)
- Duplicate messages are harmless (idempotence)
- No coordination is needed — just merge whenever you receive a state from another replica

### 3.2 Partial Order

A join-semilattice induces a partial order ≤ defined as:

$$a \leq b \iff a \sqcup b = b$$

Intuitively: a ≤ b means "b contains all information in a (and possibly more)." The merge operation always moves *upward* in this partial order — states monotonically grow.

```
Example: Set union as join-semilattice

      {a,b,c}        ← top (after all merges)
      ╱   │   ╲
  {a,b} {a,c} {b,c}
    ╱╲   ╱╲   ╱╲
  {a}  {b}  {c}
    ╲   │   ╱
       {}              ← bottom (initial state)

Merge = set union: {a,b} ⊔ {b,c} = {a,b,c}
```

### 3.3 Why This Guarantees Convergence

**Theorem** (Shapiro et al., 2011): If the state of each replica forms a join-semilattice and the merge operation is the lattice join, then the system achieves Strong Eventual Consistency.

**Proof sketch**:
1. All replicas start at the bottom element
2. Each local update moves the state upward in the lattice
3. Merge with a remote state is also a join, moving upward
4. Commutativity + associativity ensure the same final state regardless of merge order
5. Idempotence ensures duplicate messages are harmless
6. Once all replicas have received all updates, they are all at the same lattice element

---

## 4. State-Based CRDTs (CvRDTs)

State-based CRDTs (Convergent Replicated Data Types) work by sending the **full state** to other replicas, which merge it using the lattice join.

```
Replica A                    Replica B
    │                            │
    │  local operations          │  local operations
    │  ──────────────            │  ──────────────
    │                            │
    │──── send full state ──────►│
    │                            │  merge(local_state, received_state)
    │◄── send full state ────────│
    │  merge(local_state,        │
    │        received_state)     │
```

### 4.1 G-Counter (Grow-Only Counter)

The simplest CRDT. Each node maintains its own counter in a vector. The global count is the sum of all entries.

**State**: Vector of per-node counts `{node_id: count}`

**Operations**:
- `increment(node_id)`: Increment this node's entry
- `value()`: Sum of all entries
- `merge(other)`: Element-wise maximum

```python
from __future__ import annotations
from typing import Any


class GCounter:
    """
    Grow-Only Counter (G-Counter).

    Each replica has its own slot in a vector. Increment only
    affects the local slot. Merge takes element-wise maximum.

    Lattice: the partial order is component-wise ≤
    Join: component-wise max
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.counts: dict[str, int] = {}

    def increment(self, amount: int = 1) -> None:
        """Increment this node's counter. Only grows (amount must be >= 0)."""
        assert amount >= 0, "G-Counter can only grow"
        self.counts[self.node_id] = self.counts.get(self.node_id, 0) + amount

    def value(self) -> int:
        """The counter value is the sum of all node counts."""
        return sum(self.counts.values())

    def merge(self, other: GCounter) -> GCounter:
        """
        Merge with another G-Counter.
        Takes element-wise maximum — satisfies semilattice properties.
        """
        result = GCounter(self.node_id)
        all_nodes = set(self.counts.keys()) | set(other.counts.keys())
        for node in all_nodes:
            result.counts[node] = max(
                self.counts.get(node, 0),
                other.counts.get(node, 0),
            )
        return result

    def __repr__(self) -> str:
        return f"GCounter(value={self.value()}, counts={self.counts})"


def demo_g_counter():
    """Demonstrate G-Counter convergence."""
    # Three replicas
    a = GCounter("A")
    b = GCounter("B")
    c = GCounter("C")

    # Independent increments
    a.increment(3)  # A sees 3 clicks
    b.increment(5)  # B sees 5 clicks
    c.increment(2)  # C sees 2 clicks

    print(f"Before merge: A={a.value()}, B={b.value()}, C={c.value()}")

    # Merge A and B
    ab = a.merge(b)
    print(f"A merge B = {ab.value()}")  # 3 + 5 = 8

    # Merge result with C
    abc = ab.merge(c)
    print(f"(A merge B) merge C = {abc.value()}")  # 3 + 5 + 2 = 10

    # Verify commutativity: C merge (B merge A)
    ba = b.merge(a)
    cba = c.merge(ba)
    print(f"C merge (B merge A) = {cba.value()}")  # also 10

    # Verify idempotence: merging with self
    abc2 = abc.merge(abc)
    print(f"Idempotent: {abc.value()} == {abc2.value()}")  # True
```

**Why element-wise max?** If we used addition, a merge would double-count. If we used just `max`, we'd lose independent increments. Element-wise max on per-node counters gives the correct semantics: each node's contributions are counted exactly once.

### 4.2 PN-Counter (Positive-Negative Counter)

A G-Counter can only grow. To support decrements, we use *two* G-Counters: one for increments (P) and one for decrements (N). The value is P - N.

```python
class PNCounter:
    """
    Positive-Negative Counter.

    Two G-Counters: P for increments, N for decrements.
    Value = P.value() - N.value()

    Supports both increment and decrement while maintaining
    CRDT convergence properties.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.p = GCounter(node_id)  # positive (increment) counter
        self.n = GCounter(node_id)  # negative (decrement) counter

    def increment(self, amount: int = 1) -> None:
        """Increment the counter."""
        self.p.increment(amount)

    def decrement(self, amount: int = 1) -> None:
        """Decrement the counter."""
        self.n.increment(amount)

    def value(self) -> int:
        """Net value = increments - decrements."""
        return self.p.value() - self.n.value()

    def merge(self, other: PNCounter) -> PNCounter:
        """Merge two PN-Counters by merging P and N independently."""
        result = PNCounter(self.node_id)
        result.p = self.p.merge(other.p)
        result.n = self.n.merge(other.n)
        return result

    def __repr__(self) -> str:
        return f"PNCounter(value={self.value()}, P={self.p.value()}, N={self.n.value()})"


def demo_pn_counter():
    """Demonstrate PN-Counter with increments and decrements."""
    a = PNCounter("A")
    b = PNCounter("B")

    # A increments 10 times, decrements 3 times
    a.increment(10)
    a.decrement(3)

    # B increments 5 times, decrements 1 time
    b.increment(5)
    b.decrement(1)

    print(f"A = {a.value()} (P={a.p.value()}, N={a.n.value()})")  # 10-3=7
    print(f"B = {b.value()} (P={b.p.value()}, N={b.n.value()})")  # 5-1=4

    # Merge: total value should be (10+5) - (3+1) = 11
    merged = a.merge(b)
    print(f"Merged = {merged.value()}")  # 11
```

### 4.3 G-Set (Grow-Only Set)

A set that only supports adding elements — never removing.

**Lattice**: Set union is the join.

```python
class GSet:
    """
    Grow-Only Set.

    Elements can only be added, never removed.
    Merge = set union (trivially a join-semilattice).
    """

    def __init__(self):
        self.elements: set[Any] = set()

    def add(self, element: Any) -> None:
        """Add an element. Irreversible."""
        self.elements.add(element)

    def lookup(self, element: Any) -> bool:
        """Check membership."""
        return element in self.elements

    def merge(self, other: GSet) -> GSet:
        """Merge = set union."""
        result = GSet()
        result.elements = self.elements | other.elements
        return result

    def value(self) -> set:
        return frozenset(self.elements)

    def __repr__(self) -> str:
        return f"GSet({self.elements})"
```

### 4.4 2P-Set (Two-Phase Set)

Supports both add and remove, but once removed, an element can **never be re-added** (tombstone is permanent).

```python
class TwoPSet:
    """
    Two-Phase Set (2P-Set).

    Two G-Sets: one for additions (A), one for removals (R).
    An element is in the set iff it is in A but not in R.

    Limitation: once removed, an element cannot be re-added.
    """

    def __init__(self):
        self.add_set = GSet()      # additions
        self.remove_set = GSet()   # removals (tombstones)

    def add(self, element: Any) -> None:
        """Add an element (only effective if not previously removed)."""
        self.add_set.add(element)

    def remove(self, element: Any) -> bool:
        """
        Remove an element. Must have been added first.
        Once removed, cannot be re-added (permanent tombstone).
        """
        if not self.add_set.lookup(element):
            return False  # can only remove elements that exist
        self.remove_set.add(element)
        return True

    def lookup(self, element: Any) -> bool:
        """Element is present iff in add_set and NOT in remove_set."""
        return self.add_set.lookup(element) and not self.remove_set.lookup(element)

    def value(self) -> set:
        return self.add_set.elements - self.remove_set.elements

    def merge(self, other: TwoPSet) -> TwoPSet:
        """Merge by independently merging add and remove sets."""
        result = TwoPSet()
        result.add_set = self.add_set.merge(other.add_set)
        result.remove_set = self.remove_set.merge(other.remove_set)
        return result

    def __repr__(self) -> str:
        return f"2PSet(value={self.value()})"
```

### 4.5 LWW-Register (Last-Writer-Wins Register)

Stores a single value with a timestamp. On merge, the value with the higher timestamp wins.

```python
import time as _time
from dataclasses import dataclass


@dataclass
class LWWRegister:
    """
    Last-Writer-Wins Register.

    Each update is tagged with a timestamp. Merge keeps the
    value with the highest timestamp.

    Lattice: ordered by timestamp (higher = greater in lattice)
    Join: max by timestamp

    WARNING: Relies on clock accuracy. Clock skew can cause
    "newer" writes to be silently overwritten.
    """

    value: Any = None
    timestamp: float = 0.0
    node_id: str = ""

    def update(self, value: Any, node_id: str,
               timestamp: float | None = None) -> None:
        """Update the register with a new value and timestamp."""
        ts = timestamp if timestamp is not None else _time.time()
        if ts > self.timestamp or (ts == self.timestamp and node_id > self.node_id):
            self.value = value
            self.timestamp = ts
            self.node_id = node_id

    def merge(self, other: LWWRegister) -> LWWRegister:
        """Merge: keep the value with the higher timestamp."""
        if other.timestamp > self.timestamp:
            return LWWRegister(other.value, other.timestamp, other.node_id)
        if other.timestamp == self.timestamp:
            # Deterministic tie-breaking: higher node_id wins
            if other.node_id > self.node_id:
                return LWWRegister(other.value, other.timestamp, other.node_id)
        return LWWRegister(self.value, self.timestamp, self.node_id)

    def __repr__(self) -> str:
        return f"LWWRegister(value={self.value}, ts={self.timestamp:.6f})"


def demo_lww_register():
    """Demonstrate LWW-Register with concurrent writes."""
    reg_a = LWWRegister()
    reg_b = LWWRegister()

    # Concurrent updates with explicit timestamps
    reg_a.update("Alice", "A", timestamp=100.0)
    reg_b.update("Bob", "B", timestamp=100.5)  # later timestamp

    # Merge: Bob wins (higher timestamp)
    merged = reg_a.merge(reg_b)
    print(f"Merged: {merged}")  # Bob

    # The problem: if clocks are skewed, older data can win
    reg_c = LWWRegister()
    reg_c.update("Charlie", "C", timestamp=200.0)  # skewed clock far ahead

    reg_d = LWWRegister()
    reg_d.update("Diana", "D", timestamp=101.0)  # real time, but lower

    merged2 = reg_c.merge(reg_d)
    print(f"Skewed: {merged2}")  # Charlie wins despite being "wrong"
```

### 4.6 MV-Register (Multi-Value Register)

Instead of discarding concurrent writes (LWW), an MV-Register preserves all concurrent values. The application can then resolve the conflict.

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class VectorClock:
    """Immutable vector clock for causality tracking."""
    clock: tuple[tuple[str, int], ...] = ()

    def _to_dict(self) -> dict[str, int]:
        return dict(self.clock)

    @classmethod
    def from_dict(cls, d: dict[str, int]) -> VectorClock:
        return cls(tuple(sorted(d.items())))

    def increment(self, node_id: str) -> VectorClock:
        d = self._to_dict()
        d[node_id] = d.get(node_id, 0) + 1
        return VectorClock.from_dict(d)

    def merge(self, other: VectorClock) -> VectorClock:
        d = self._to_dict()
        for node, count in other._to_dict().items():
            d[node] = max(d.get(node, 0), count)
        return VectorClock.from_dict(d)

    def __le__(self, other: VectorClock) -> bool:
        """a <= b iff all components of a are <= corresponding components of b."""
        d_self = self._to_dict()
        d_other = other._to_dict()
        for node, count in d_self.items():
            if count > d_other.get(node, 0):
                return False
        return True

    def concurrent_with(self, other: VectorClock) -> bool:
        """Two clocks are concurrent iff neither dominates the other."""
        return not (self <= other) and not (other <= self)


class MVRegister:
    """
    Multi-Value Register.

    Preserves all concurrent values instead of arbitrarily picking one.
    Uses vector clocks to determine causality.
    Clients see all concurrent values and can resolve conflicts.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        # List of (value, vector_clock) pairs
        self.values: list[tuple[Any, VectorClock]] = []

    def update(self, value: Any) -> None:
        """
        Update with a new value. Supersedes all values causally
        dominated by this node's current knowledge.
        """
        # Merge all current vector clocks and increment
        merged_vc = VectorClock()
        for _, vc in self.values:
            merged_vc = merged_vc.merge(vc)
        new_vc = merged_vc.increment(self.node_id)

        # New value supersedes all existing values
        self.values = [(value, new_vc)]

    def read(self) -> list[Any]:
        """Return all concurrent values."""
        return [v for v, _ in self.values]

    def merge(self, other: MVRegister) -> MVRegister:
        """
        Merge: keep values that are not dominated by any value
        in the other register.
        """
        result = MVRegister(self.node_id)
        all_entries = self.values + other.values

        # Keep only entries not dominated by another entry
        surviving = []
        for val, vc in all_entries:
            dominated = False
            for other_val, other_vc in all_entries:
                if vc <= other_vc and vc != other_vc:
                    dominated = True
                    break
            if not dominated:
                # Deduplicate
                if (val, vc) not in surviving:
                    surviving.append((val, vc))
        result.values = surviving
        return result

    def __repr__(self) -> str:
        return f"MVRegister(values={self.read()})"


def demo_mv_register():
    """Demonstrate MV-Register preserving concurrent writes."""
    reg_a = MVRegister("A")
    reg_b = MVRegister("B")

    # Sequential update on A
    reg_a.update("version-1")
    reg_a.update("version-2")  # supersedes version-1

    # Concurrent update on B (doesn't know about A's updates)
    reg_b.update("version-X")

    print(f"A: {reg_a.read()}")  # ['version-2']
    print(f"B: {reg_b.read()}")  # ['version-X']

    # Merge: both concurrent values preserved
    merged = reg_a.merge(reg_b)
    print(f"Merged: {merged.read()}")  # ['version-2', 'version-X']
    # Application must resolve: e.g., pick one or merge semantically
```

### 4.7 OR-Set (Observed-Remove Set)

The OR-Set is one of the most practical CRDTs. Unlike the 2P-Set, it allows re-adding elements after removal. Each addition is tagged with a unique identifier; removal only affects the *observed* tags.

```python
import uuid
from typing import Any


class ORSet:
    """
    Observed-Remove Set (OR-Set).

    Each add() generates a unique tag. remove() removes all
    currently observed tags for an element. A concurrent add()
    on another replica (with a different tag) survives the remove.

    This is the most practical set CRDT: supports add, remove,
    and re-add with intuitive semantics.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        # Map: element -> set of unique tags
        self.elements: dict[Any, set[str]] = {}
        # Tombstones: tags that have been removed
        self.tombstones: set[str] = set()

    def add(self, element: Any) -> str:
        """
        Add an element with a unique tag.
        Returns the tag (useful for testing).
        """
        tag = f"{self.node_id}:{uuid.uuid4().hex[:8]}"
        if element not in self.elements:
            self.elements[element] = set()
        self.elements[element].add(tag)
        return tag

    def remove(self, element: Any) -> bool:
        """
        Remove an element by tombstoning all its currently observed tags.
        Concurrent adds (with different tags) will survive this remove.
        """
        if element not in self.elements:
            return False
        tags = self.elements[element]
        if not tags:
            return False
        # Tombstone all observed tags
        self.tombstones.update(tags)
        self.elements[element] = set()
        return True

    def lookup(self, element: Any) -> bool:
        """Element is present iff it has at least one non-tombstoned tag."""
        if element not in self.elements:
            return False
        live_tags = self.elements[element] - self.tombstones
        return len(live_tags) > 0

    def value(self) -> set:
        """Return the set of all present elements."""
        result = set()
        for element, tags in self.elements.items():
            live_tags = tags - self.tombstones
            if live_tags:
                result.add(element)
        return result

    def merge(self, other: ORSet) -> ORSet:
        """
        Merge two OR-Sets.

        For each element:
        - Union all tags from both replicas
        - Union all tombstones from both replicas
        - An element is present iff it has tags not in tombstones
        """
        result = ORSet(self.node_id)

        # Union all elements and their tags
        all_elements = set(self.elements.keys()) | set(other.elements.keys())
        for element in all_elements:
            tags_a = self.elements.get(element, set())
            tags_b = other.elements.get(element, set())
            result.elements[element] = tags_a | tags_b

        # Union all tombstones
        result.tombstones = self.tombstones | other.tombstones

        return result

    def __repr__(self) -> str:
        return f"ORSet({self.value()})"


def demo_or_set():
    """Demonstrate OR-Set with concurrent add and remove."""
    set_a = ORSet("A")
    set_b = ORSet("B")

    # A adds "milk"
    set_a.add("milk")
    print(f"A after add milk: {set_a.value()}")  # {'milk'}

    # Sync: B receives A's state
    synced = set_a.merge(set_b)
    set_b = synced  # B now has "milk"
    set_b.node_id = "B"

    # Concurrent operations:
    # A removes "milk"
    set_a.remove("milk")
    print(f"A after remove milk: {set_a.value()}")  # set()

    # B concurrently adds "milk" again (different tag!)
    set_b.add("milk")
    print(f"B after concurrent add: {set_b.value()}")  # {'milk'}

    # Merge: B's add survives A's remove (add wins over concurrent remove)
    merged = set_a.merge(set_b)
    print(f"Merged (add wins): {merged.value()}")  # {'milk'}

    # This is the "add-wins" semantics of OR-Set:
    # A removed the tags it observed, but B added a new tag
    # that A's remove didn't know about.
```

---

## 5. Operation-Based CRDTs (CmRDTs)

Operation-based CRDTs (Commutative Replicated Data Types) take a different approach: instead of sending the full state, they send **operations** to other replicas.

### 5.1 Requirements

For op-based CRDTs to converge, they need:

1. **Commutative operations**: The result must be the same regardless of the order operations are applied
2. **Reliable causal broadcast**: The messaging layer must deliver every operation exactly once and in causal order

```
State-based:                    Operation-based:
  Send full state               Send operation (e.g., "increment by 1")
  Receiver merges                Receiver applies operation
  Idempotent (resend OK)         NOT idempotent (must deliver exactly once)
  Larger messages                Smaller messages
  Any network (even lossy)       Requires reliable causal broadcast
```

### 5.2 Op-Based Counter

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable


class OpType(Enum):
    INCREMENT = "increment"
    DECREMENT = "decrement"


@dataclass
class Operation:
    """An operation to be broadcast to all replicas."""
    op_type: OpType
    amount: int
    source_node: str
    sequence_num: int  # for causal ordering


class OpBasedCounter:
    """
    Operation-based counter CRDT.

    Operations (increment, decrement) are broadcast to all replicas.
    Operations must be delivered exactly once in causal order.

    Commutativity: increment(a) then increment(b) = increment(b) then increment(a)
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.value_: int = 0
        self._seq = 0
        self._delivered: set[tuple[str, int]] = set()  # dedup: (source, seq)
        self.peers: list[OpBasedCounter] = []

    def increment(self, amount: int = 1) -> None:
        """Local increment + broadcast to peers."""
        self.value_ += amount
        self._seq += 1
        op = Operation(OpType.INCREMENT, amount, self.node_id, self._seq)
        self._broadcast(op)

    def decrement(self, amount: int = 1) -> None:
        """Local decrement + broadcast to peers."""
        self.value_ -= amount
        self._seq += 1
        op = Operation(OpType.DECREMENT, amount, self.node_id, self._seq)
        self._broadcast(op)

    def receive(self, op: Operation) -> None:
        """
        Receive and apply an operation from a peer.
        Must be exactly-once delivery (dedup check).
        """
        key = (op.source_node, op.sequence_num)
        if key in self._delivered:
            return  # duplicate — ignore
        self._delivered.add(key)

        if op.op_type == OpType.INCREMENT:
            self.value_ += op.amount
        elif op.op_type == OpType.DECREMENT:
            self.value_ -= op.amount

    def _broadcast(self, op: Operation) -> None:
        """Reliable broadcast to all peers."""
        for peer in self.peers:
            peer.receive(op)

    def __repr__(self) -> str:
        return f"OpCounter({self.node_id}, value={self.value_})"


def demo_op_counter():
    """Demonstrate operation-based counter."""
    a = OpBasedCounter("A")
    b = OpBasedCounter("B")
    c = OpBasedCounter("C")

    # Wire up peers
    a.peers = [b, c]
    b.peers = [a, c]
    c.peers = [a, b]

    # Operations at different nodes
    a.increment(5)
    b.increment(3)
    c.decrement(1)

    # All replicas converge
    print(f"A={a.value_}, B={b.value_}, C={c.value_}")  # all 7
    assert a.value_ == b.value_ == c.value_ == 7
```

### 5.3 Op-Based vs State-Based: Trade-Offs

| Aspect | State-Based (CvRDT) | Operation-Based (CmRDT) |
|---|---|---|
| **Message size** | Full state (can be large) | Single operation (small) |
| **Message delivery** | Any (lossy, reordered OK) | Exactly-once, causal order required |
| **Idempotence** | Built-in (merge is idempotent) | NOT idempotent (dedup needed) |
| **Network requirement** | Unreliable (gossip works) | Reliable causal broadcast |
| **Implementation** | Simpler merge logic | Simpler per-operation logic |
| **Bandwidth** | High (scales with state size) | Low (scales with operation count) |

**In practice**: State-based CRDTs are more common because they have weaker network requirements. Delta-state CRDTs (Section 6) combine the best of both.

---

## 6. Delta-State CRDTs

Delta-state CRDTs address the main weakness of state-based CRDTs: sending the **full state** is expensive for large data structures.

### 6.1 Key Idea

Instead of sending the full state, send only the **delta** — the minimal state change since the last synchronization. The delta itself must be a valid join-semilattice element so it can be merged.

```
Full state sync:
  Replica A: {a:5, b:3, c:7} ──── sends 3 entries ────► Replica B

Delta sync:
  Replica A: only changed a:5→6 ── sends {a:6} ──────► Replica B
  Replica B: merge({a:6}) into local state
```

### 6.2 Delta G-Counter

```python
class DeltaGCounter:
    """
    Delta-state G-Counter.

    Tracks changes since last sync and sends only the delta.
    Reduces bandwidth from O(num_nodes) to O(changed_nodes) per sync.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.counts: dict[str, int] = {}
        self._pending_delta: dict[str, int] = {}  # unsent changes

    def increment(self, amount: int = 1) -> None:
        """Increment and record the delta."""
        old = self.counts.get(self.node_id, 0)
        new = old + amount
        self.counts[self.node_id] = new
        self._pending_delta[self.node_id] = new

    def value(self) -> int:
        return sum(self.counts.values())

    def get_delta(self) -> dict[str, int]:
        """Get the pending delta (changes since last sync) and clear it."""
        delta = dict(self._pending_delta)
        self._pending_delta.clear()
        return delta

    def merge_delta(self, delta: dict[str, int]) -> None:
        """Merge a delta from another replica."""
        for node, count in delta.items():
            self.counts[node] = max(self.counts.get(node, 0), count)

    def merge_full(self, other: DeltaGCounter) -> None:
        """Full state merge (fallback for new replicas or recovery)."""
        for node, count in other.counts.items():
            self.counts[node] = max(self.counts.get(node, 0), count)

    def __repr__(self) -> str:
        return f"DeltaGCounter(value={self.value()}, counts={self.counts})"


def demo_delta_counter():
    """Demonstrate delta-state synchronization."""
    a = DeltaGCounter("A")
    b = DeltaGCounter("B")

    # A increments several times
    a.increment(3)
    a.increment(2)

    # B increments
    b.increment(7)

    # Sync via deltas (not full state)
    delta_a = a.get_delta()  # {A: 5}
    delta_b = b.get_delta()  # {B: 7}

    print(f"Delta from A: {delta_a}")  # small!
    print(f"Delta from B: {delta_b}")  # small!

    a.merge_delta(delta_b)
    b.merge_delta(delta_a)

    print(f"A: {a.value()}, B: {b.value()}")  # both 12
```

### 6.3 Delta Dissemination Strategies

| Strategy | Description | Use Case |
|---|---|---|
| **Pairwise** | Each pair of replicas exchanges deltas | Small clusters |
| **Gossip** | Randomly select peers, exchange deltas | Large clusters |
| **Anti-entropy** | Periodic full-state sync as fallback | Recovery, new nodes |
| **Causal delta** | Attach causal metadata to deltas | When ordering matters |

---

## 7. Practical Applications

### 7.1 Collaborative Editing

**Automerge** and **Yjs** use CRDTs for real-time collaborative editing. The text is represented as a sequence CRDT where each character has a unique ID and position.

```
User A types "Hello":  H-e-l-l-o
User B types "Hi" concurrently:  H-i

CRDT merge produces interleaved but consistent result:
  Both replicas see the same merged document
  (exact result depends on CRDT algorithm and IDs)
```

### 7.2 Shopping Cart (Amazon)

Amazon's Dynamo paper described a shopping cart as an OR-Set:
- Adding an item = `add(item)` with unique tag
- Removing an item = `remove(item)` tombstones observed tags
- Concurrent add on different replicas: both items survive
- "Add wins" semantics: worst case, a removed item reappears (customer removes again)

```python
class ShoppingCart:
    """Shopping cart built on OR-Set CRDT."""

    def __init__(self, user_id: str, replica_id: str):
        self.user_id = user_id
        self._set = ORSet(replica_id)

    def add_item(self, item: str, quantity: int = 1) -> None:
        """Add an item to the cart."""
        self._set.add((item, quantity))

    def remove_item(self, item: str) -> None:
        """Remove all entries for an item."""
        to_remove = [
            (i, q) for i, q in self._set.value() if i == item
        ]
        for entry in to_remove:
            self._set.remove(entry)

    def contents(self) -> dict[str, int]:
        """Get cart contents as {item: total_quantity}."""
        result: dict[str, int] = {}
        for item, qty in self._set.value():
            result[item] = result.get(item, 0) + qty
        return result

    def merge(self, other: ShoppingCart) -> ShoppingCart:
        """Merge with another replica of the same cart."""
        result = ShoppingCart(self.user_id, self._set.node_id)
        result._set = self._set.merge(other._set)
        return result

    def __repr__(self) -> str:
        return f"Cart({self.contents()})"
```

### 7.3 Distributed Counters (Redis CRDT)

Redis Enterprise uses CRDTs for active-active geo-replication. Counters are implemented as PN-Counters, sets as OR-Sets, and strings as LWW-Registers.

```
Redis CRDT Type Mapping:
  Redis String   →  LWW-Register (or op-based with timestamp)
  Redis Counter  →  PN-Counter (per-replica increments)
  Redis Set      →  OR-Set (add-wins semantics)
  Redis Hash     →  Map of LWW-Registers (per-field LWW)
  Redis Sorted Set → OR-Set with LWW scores
```

### 7.4 DNS Health Checks (AWS Route 53)

AWS Route 53 uses CRDTs to aggregate health check results from multiple health checker nodes distributed globally. Each checker reports UP/DOWN independently; the aggregated result converges without coordination.

---

## 8. CRDT Limitations

CRDTs are powerful but not a universal solution. Understanding their limitations is crucial.

### 8.1 Monotonic Growth (Tombstone Overhead)

CRDTs require **monotonic** state growth in the lattice. Removals are implemented via tombstones, which are never garbage-collected (without coordination).

```
OR-Set after 1 million add/remove cycles:

  Live elements: 100
  Tombstones: 999,900  ← dominates memory!
```

**Mitigations**:
- Periodic garbage collection epochs (requires coordination — weakens CRDT properties)
- Causal stability: a tombstone can be GC'd once all replicas have observed it
- Application-level TTLs (e.g., tombstones older than 7 days are pruned)

### 8.2 Limited Expressiveness

Not all data structures can be expressed as CRDTs. CRDTs are fundamentally limited to operations that can be made commutative.

| Can Be a CRDT | Cannot (Easily) Be a CRDT |
|---|---|
| Counters | Bounded counters (e.g., "max 100") |
| Sets | Unique-element sets (e.g., "exactly one of X") |
| Registers | Move operations (e.g., "move item from set A to set B" atomically) |
| Sequences (text) | Transactions across multiple CRDTs |
| Flags (enable/disable) | Invariant-preserving operations |

### 8.3 Conflict Semantics May Surprise Users

- **LWW silently drops writes**: Users may not realize their changes were discarded
- **Add-wins (OR-Set)**: A removed item can reappear if a concurrent add exists
- **PN-Counter can go negative**: Decrementing below zero is possible if increments haven't propagated

### 8.4 Metadata Overhead

CRDTs carry metadata (vector clocks, unique tags, per-node counters) that grows with the number of replicas and operations.

```
G-Counter metadata: O(number of replicas)
OR-Set metadata: O(number of add operations ever performed)
MV-Register metadata: O(number of concurrent versions × vector clock size)
```

For systems with many replicas or high write rates, this overhead can be significant.

### 8.5 No Total Ordering

CRDTs provide convergence but NOT total ordering. If you need "the counter reached exactly 100 and then stopped," a CRDT cannot enforce this without coordination.

---

## 9. Comprehensive CRDT Comparison

```python
def crdt_comparison_table():
    """Print a comprehensive comparison of all CRDTs covered."""
    headers = ["CRDT", "Type", "Operations", "Merge", "Limitations"]
    rows = [
        ["G-Counter", "State", "increment", "element-wise max",
         "Cannot decrement"],
        ["PN-Counter", "State", "increment, decrement", "merge P and N G-Counters",
         "Can go negative; no bound"],
        ["G-Set", "State", "add", "set union",
         "Cannot remove elements"],
        ["2P-Set", "State", "add, remove (once)", "merge add and remove sets",
         "Cannot re-add removed elements"],
        ["LWW-Register", "State", "update (with timestamp)", "keep higher timestamp",
         "Silent data loss; clock-dependent"],
        ["MV-Register", "State", "update (with vector clock)", "keep concurrent values",
         "Multiple values need app resolution"],
        ["OR-Set", "State", "add, remove, re-add", "union tags + tombstones",
         "Tombstone overhead"],
        ["Op-Counter", "Op", "increment, decrement", "apply operations",
         "Requires reliable causal broadcast"],
    ]

    # Print formatted table
    col_widths = [max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "-+-".join("-" * w for w in col_widths)

    print(header_line)
    print(separator)
    for row in rows:
        print(" | ".join(cell.ljust(w) for cell, w in zip(row, col_widths)))
```

| CRDT | Type | Supported Operations | Merge Strategy | Primary Limitation |
|---|---|---|---|---|
| G-Counter | State | increment | Element-wise max | Cannot decrement |
| PN-Counter | State | increment, decrement | Merge P and N independently | Can go negative |
| G-Set | State | add | Set union | Cannot remove |
| 2P-Set | State | add, remove (once) | Merge add + remove sets | No re-add after remove |
| LWW-Register | State | update (timestamped) | Higher timestamp wins | Silent data loss |
| MV-Register | State | update (vector clock) | Keep concurrent values | App must resolve conflicts |
| OR-Set | State | add, remove, re-add | Union tags + tombstones | Tombstone growth |
| Delta-State | State | same as base CRDT | Merge deltas | Requires delta tracking |
| Op-Counter | Op-based | increment, decrement | Apply operations | Needs reliable delivery |

---

## 10. Full Demonstration

```python
def full_crdt_demo():
    """Comprehensive demonstration of all CRDTs with merge verification."""

    print("=" * 60)
    print("G-Counter Demo")
    print("=" * 60)
    demo_g_counter()

    print("\n" + "=" * 60)
    print("PN-Counter Demo")
    print("=" * 60)
    demo_pn_counter()

    print("\n" + "=" * 60)
    print("LWW-Register Demo")
    print("=" * 60)
    demo_lww_register()

    print("\n" + "=" * 60)
    print("MV-Register Demo")
    print("=" * 60)
    demo_mv_register()

    print("\n" + "=" * 60)
    print("OR-Set Demo")
    print("=" * 60)
    demo_or_set()

    print("\n" + "=" * 60)
    print("Op-Based Counter Demo")
    print("=" * 60)
    demo_op_counter()

    print("\n" + "=" * 60)
    print("Delta-State Counter Demo")
    print("=" * 60)
    demo_delta_counter()

    print("\n" + "=" * 60)
    print("CRDT Comparison")
    print("=" * 60)
    crdt_comparison_table()

    # Verify semilattice properties for G-Counter
    print("\n" + "=" * 60)
    print("Semilattice Property Verification (G-Counter)")
    print("=" * 60)

    a = GCounter("A")
    b = GCounter("B")
    c = GCounter("C")
    a.increment(3)
    b.increment(5)
    c.increment(7)

    # Commutativity: a ⊔ b = b ⊔ a
    ab = a.merge(b)
    ba = b.merge(a)
    assert ab.counts == ba.counts, "Commutativity violated!"
    print(f"Commutativity: a⊔b = b⊔a = {ab.value()} ✓")

    # Associativity: (a ⊔ b) ⊔ c = a ⊔ (b ⊔ c)
    ab_c = ab.merge(c)
    bc = b.merge(c)
    a_bc = a.merge(bc)
    assert ab_c.counts == a_bc.counts, "Associativity violated!"
    print(f"Associativity: (a⊔b)⊔c = a⊔(b⊔c) = {ab_c.value()} ✓")

    # Idempotence: a ⊔ a = a
    aa = a.merge(a)
    assert aa.counts == a.counts, "Idempotence violated!"
    print(f"Idempotence: a⊔a = a = {aa.value()} ✓")

    print("\nAll semilattice properties verified!")


if __name__ == "__main__":
    full_crdt_demo()
```

---

## 11. Summary

| Concept | Key Takeaway |
|---|---|
| **Eventual consistency** | Weak guarantee — replicas converge "eventually" with no timing bound |
| **Strong eventual consistency** | CRDTs guarantee convergence once the same updates are delivered |
| **Join-semilattice** | Commutative + associative + idempotent merge = order-independent convergence |
| **State-based CRDTs** | Send full state; merge via lattice join; tolerant of lossy networks |
| **Operation-based CRDTs** | Send operations; smaller messages; require reliable causal delivery |
| **Delta-state CRDTs** | Best of both: small deltas with semilattice merge |
| **OR-Set** | Most practical set CRDT: add, remove, re-add with "add-wins" semantics |
| **Limitations** | Tombstone overhead, limited expressiveness, no invariant enforcement |

### Design Guidelines

1. **Start with the simplest CRDT** that meets your needs (G-Counter before PN-Counter before OR-Set)
2. **Understand the conflict semantics** — LWW loses data silently; explain this to users or choose add-wins
3. **Plan for tombstone garbage collection** — without it, metadata grows without bound
4. **Consider delta-state CRDTs** for bandwidth-constrained environments
5. **CRDTs complement consensus** — use CRDTs for data that can tolerate eventual consistency; use Paxos/Raft for data that requires strong consistency

---

[Next: Partitioning and Sharding](./11_Partitioning_and_Sharding.md)

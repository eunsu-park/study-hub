# Lesson 2: Time, Clocks, and Ordering

[Overview](./00_Overview.md) | [Previous](./01_System_Models_and_Failure_Modes.md) | [Next](./03_FLP_Impossibility_and_Bounds.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why physical clocks are insufficient for ordering events in distributed systems and quantify sources of clock error
2. Define the happens-before relation and use it to determine causal ordering of events
3. Implement Lamport timestamps and understand their limitations for detecting concurrency
4. Implement vector clocks to capture the full causal history of events and detect concurrent operations
5. Describe Hybrid Logical Clocks (HLC) and Google Spanner's TrueTime as practical approaches to distributed time

---

## Table of Contents

1. [The Problem with Physical Time](#1-the-problem-with-physical-time)
2. [Happens-Before Relation](#2-happens-before-relation)
3. [Lamport Timestamps](#3-lamport-timestamps)
4. [Vector Clocks](#4-vector-clocks)
5. [Version Vectors vs Vector Clocks](#5-version-vectors-vs-vector-clocks)
6. [Hybrid Logical Clocks (HLC)](#6-hybrid-logical-clocks-hlc)
7. [TrueTime and Interval-Based Clocks](#7-truetime-and-interval-based-clocks)
8. [Comparison of Clock Mechanisms](#8-comparison-of-clock-mechanisms)
9. [Code: Full Implementations](#9-code-full-implementations)
10. [Exercises](#10-exercises)
11. [Summary](#11-summary)
12. [References](#12-references)

---

## 1. The Problem with Physical Time

### 1.1 Why We Need a Notion of "Order"

In a single-process program, the order of events is trivial: instruction `A` happens before instruction `B` if `A` appears earlier in the program. In a distributed system, there is no single instruction sequence. Events happen at different nodes, and we need a way to answer the question: "Did event `A` happen before event `B`?"

This question is critical for:

- **Database consistency**: Did the write happen before or after the read?
- **Distributed debugging**: Which log entry came first?
- **Conflict resolution**: Which update should win?
- **Causal messaging**: Did this reply come after the message it responds to?

### 1.2 Physical Clock Sources

Every computer has a physical clock, but these clocks are imperfect:

| Clock Source | Typical Accuracy | Drift Rate | Notes |
|-------------|-----------------|------------|-------|
| Quartz oscillator | ~10 ppm | ~1 sec/day | Standard PC hardware |
| NTP-synchronized | ~1-10 ms | Corrected periodically | Depends on network path |
| PTP (IEEE 1588) | ~1-100 μs | Hardware-assisted | Requires PTP-capable NICs |
| GPS receiver | ~10-100 ns | Atomic-clock quality | Requires antenna, sky visibility |
| Atomic clock (Cs/Rb) | ~1 ns | ~10⁻¹² s/s | Expensive, used in data centers |
| Google TrueTime | ~1-7 ms uncertainty | GPS + atomic | Custom infrastructure |

### 1.3 Sources of Clock Error

**Clock drift**: Quartz oscillators vibrate at a frequency that depends on temperature, voltage, and aging. A typical drift rate of 10 ppm (parts per million) means:

```
10 ppm = 10 × 10⁻⁶ = 0.00001

Over 1 day (86,400 seconds):
  drift = 86,400 × 0.00001 = 0.864 seconds

Over 1 week:
  drift ≈ 6 seconds

Two independent clocks can diverge by up to:
  2 × drift = 1.7 seconds/day
```

**Clock skew**: The instantaneous difference between two clocks at a given point in real time.

```
skew(A, B) = C_A(t) - C_B(t)

where C_A(t) is the reading of clock A at real time t
```

**NTP correction issues**:

1. **Step adjustment**: NTP may jump the clock forward or backward, causing timestamps to go backward (non-monotonic).
2. **Slew adjustment**: NTP may speed up or slow down the clock to gradually correct drift, changing the rate of time passage.
3. **Network asymmetry**: NTP assumes symmetric network delays, but real paths are often asymmetric, introducing systematic error.

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

**Leap seconds**: UTC occasionally adds (or theoretically removes) a second to stay aligned with Earth's rotation. This means `23:59:59` is followed by `23:59:60` before `00:00:00`. Software that assumes 60 seconds per minute or monotonically increasing timestamps can break.

### 1.4 Why Physical Clocks Cannot Order Events

Consider two events at different nodes:

```
Node A: event_a at physical time T_A = 100.003
Node B: event_b at physical time T_B = 100.005

Question: Did event_a happen before event_b?

If clock skew between A and B is ±5ms:
  Real time of event_a could be anywhere in [99.998, 100.008]
  Real time of event_b could be anywhere in [100.000, 100.010]

  These intervals OVERLAP, so we CANNOT determine the order.
```

This is not a limitation of our clocks -- it is a fundamental consequence of the finiteness of the speed of light and the uncertainty of network delays. Even with perfect clocks, two events at different locations that happen within the light-travel-time between them have no well-defined order (this is essentially the same insight as special relativity's simultaneity).

**Conclusion**: We need **logical clocks** that do not depend on physical time but instead capture the causal structure of events.

---

## 2. Happens-Before Relation

### 2.1 Lamport's Insight (1978)

Leslie Lamport's seminal 1978 paper "Time, Clocks, and the Ordering of Events in a Distributed System" introduced the **happens-before** relation, denoted `→`.

**Key insight**: We do not need to know what time events happened. We only need to know whether one event could have **causally influenced** another.

### 2.2 Formal Definition

Given a distributed system with processes `p₁, p₂, ..., pₙ`, the happens-before relation `→` is the smallest relation satisfying:

1. **Process order**: If `a` and `b` are events at the same process, and `a` occurs before `b` in that process's execution, then `a → b`.

2. **Message causality**: If `a` is the sending of a message by one process and `b` is the receipt of the same message by another process, then `a → b`.

3. **Transitivity**: If `a → b` and `b → c`, then `a → c`.

If neither `a → b` nor `b → a`, then `a` and `b` are **concurrent**, written `a ‖ b`.

### 2.3 Visualizing Happens-Before

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

### 2.4 Properties of Happens-Before

| Property | Holds? | Explanation |
|----------|--------|-------------|
| Irreflexive | Yes | ¬(a → a); an event does not happen before itself |
| Antisymmetric | Yes | a → b implies ¬(b → a) |
| Transitive | Yes | a → b ∧ b → c ⟹ a → c |
| Total order? | **No** | Concurrent events are incomparable |
| Partial order? | Yes (strict) | It is a strict partial order |

### 2.5 Causality vs. Happens-Before

The happens-before relation captures **potential causality**, not actual causality. If `a → b`, it means `a` *could have* influenced `b`, not that it *did* influence `b`.

```
a → b means:
  "Information could have flowed from a to b"
  NOT: "a caused b"

a ‖ b means:
  "Information could NOT have flowed between a and b"
  Therefore: "a and b are causally independent"
```

This distinction matters because:
- `a → b` is a necessary condition for `a` to cause `b`, but not sufficient.
- `a ‖ b` is sufficient to guarantee that `a` did NOT cause `b`.

---

## 3. Lamport Timestamps

### 3.1 The Algorithm

Each process `pᵢ` maintains a counter `Cᵢ` (initialized to 0). The rules are:

1. **Internal event**: Before any event at process `pᵢ`, increment the counter:
   ```
   Cᵢ = Cᵢ + 1
   ```

2. **Send event**: Before sending a message `m`, increment the counter and attach it to the message:
   ```
   Cᵢ = Cᵢ + 1
   send(m, Cᵢ)
   ```

3. **Receive event**: Upon receiving message `m` with timestamp `t`, update the counter:
   ```
   Cᵢ = max(Cᵢ, t) + 1
   ```

### 3.2 The Clock Condition

Lamport timestamps satisfy the **clock condition**:

```
If a → b, then C(a) < C(b)
```

**But the converse is NOT true**:

```
C(a) < C(b) does NOT imply a → b
```

This means Lamport timestamps can tell us "if `a` happened before `b`, then `C(a) < C(b)`," but if we observe `C(a) < C(b)`, we cannot conclude that `a → b`. The events might be concurrent.

### 3.3 Visualization

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

### 3.4 Total Ordering with Lamport Timestamps

To create a **total order** (useful for things like mutual exclusion), break ties using process IDs:

```
(C(a), pid_a) < (C(b), pid_b) iff:
  C(a) < C(b), or
  C(a) = C(b) and pid_a < pid_b
```

This gives a total order that is **consistent with** the happens-before relation but NOT identical to it.

### 3.5 Implementation

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

### 3.6 Limitations

| Limitation | Consequence |
|-----------|-------------|
| Cannot detect concurrency | `C(a) < C(b)` does not imply `a → b` |
| No causal history | Cannot determine the causal past of an event |
| Counter size | Grows unboundedly (but slowly in practice) |
| Tie-breaking is arbitrary | Total order may not match real-time order |

The inability to detect concurrency is the primary motivation for **vector clocks**.

---

## 4. Vector Clocks

### 4.1 Motivation

We want a clock mechanism where:

```
V(a) < V(b)  if and only if  a → b
```

This requires each process to track not just its own counter, but the **latest known counter of every process** in the system.

### 4.2 Definition

In a system of `n` processes `{p₁, p₂, ..., pₙ}`, each process `pᵢ` maintains a vector `Vᵢ` of `n` integers, where `Vᵢ[j]` represents `pᵢ`'s knowledge of `pⱼ`'s latest event counter.

**Rules**:

1. **Internal event at pᵢ**: Increment own component:
   ```
   Vᵢ[i] = Vᵢ[i] + 1
   ```

2. **Send event at pᵢ**: Increment own component and attach the vector to the message:
   ```
   Vᵢ[i] = Vᵢ[i] + 1
   send(m, Vᵢ)
   ```

3. **Receive event at pᵢ from pⱼ with attached vector Vₘ**: Take the element-wise maximum, then increment own component:
   ```
   Vᵢ[k] = max(Vᵢ[k], Vₘ[k])  for all k
   Vᵢ[i] = Vᵢ[i] + 1
   ```

### 4.3 Comparison Rules

Given two vector timestamps `V(a)` and `V(b)`:

```
V(a) ≤ V(b)   iff  ∀k: V(a)[k] ≤ V(b)[k]
V(a) < V(b)    iff  V(a) ≤ V(b) and V(a) ≠ V(b)
V(a) ‖ V(b)   iff  ¬(V(a) ≤ V(b)) and ¬(V(b) ≤ V(a))
```

**The fundamental theorem of vector clocks**:

```
a → b  ⟺  V(a) < V(b)
a ‖ b  ⟺  V(a) ‖ V(b)
```

This bidirectional implication is what makes vector clocks strictly more powerful than Lamport timestamps.

### 4.4 Visualization

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

### 4.5 Implementation

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

### 4.6 Detecting Causal Relationships

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

### 4.7 Scalability Concerns

| Dimension | Impact |
|-----------|--------|
| Space per message | O(n) integers, where n = number of processes |
| Space per event | O(n) integers stored |
| Comparison cost | O(n) per comparison |
| Adding a process | Must extend all vectors (coordination required) |
| Removing a process | Cannot simply shrink (entry may be needed for future comparisons) |

For systems with thousands of processes, vector clocks become impractical due to message size. Solutions:

- **Plausible clocks**: Approximate vector clocks with bounded size
- **Matrix clocks**: Track knowledge about knowledge (O(n^2) but enables garbage collection)
- **Hybrid approaches**: Use vector clocks for a small set of replicas, physical time for coarser ordering

---

## 5. Version Vectors vs Vector Clocks

These two concepts are frequently confused but serve different purposes.

### 5.1 Key Distinction

| Aspect | Vector Clocks | Version Vectors |
|--------|--------------|-----------------|
| Tracks | Events (send, receive, internal) | Updates to a data item |
| Incremented on | Every event | Only on writes to the data item |
| Purpose | Capture complete causal history | Detect conflicts between replicas |
| Size growth | One entry per process | One entry per replica |
| Used in | Causal broadcast, debugging | Multi-master replication (Dynamo) |

### 5.2 Version Vector Example

Consider a key-value store replicated on 3 nodes:

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

### 5.3 Dotted Version Vectors

Standard version vectors can produce **false conflicts** (sibling explosion). Dotted version vectors (Preguica et al., 2012) solve this by tracking the exact event (dot) that created each sibling:

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

## 6. Hybrid Logical Clocks (HLC)

### 6.1 Motivation

Vector clocks give us perfect causality tracking but do not correlate with physical time. Lamport timestamps are compact but lose concurrency information. **Hybrid Logical Clocks** (Kulkarni et al., 2014) combine the best of both worlds:

- Bounded size (constant, not O(n))
- Capture the happens-before relation (like Lamport clocks)
- Stay close to physical time (unlike pure logical clocks)
- Can be used for snapshot queries at a physical time

### 6.2 HLC Structure

An HLC timestamp is a pair `(l, c)` where:

- `l`: the maximum physical time seen so far (logical component that tracks physical time)
- `c`: a bounded counter for breaking ties when physical clocks are equal

The invariant is:

```
l ≥ pt (physical time)  — HLC is always ≥ physical time
l is bounded above by  pt + ε  where ε is the max clock skew
```

### 6.3 Algorithm

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

### 6.4 Properties

| Property | Guarantee |
|----------|-----------|
| Clock condition | `a → b ⟹ HLC(a) < HLC(b)` (same as Lamport) |
| Physical time closeness | `l - pt ≤ ε` where ε is max clock skew |
| Counter bound | `c ≤ n × ε × event_rate` in the worst case |
| Space per timestamp | O(1) — just (l, c, process_id) |
| Comparison | Lexicographic on (l, c, process_id) |

### 6.5 HLC vs Lamport vs Vector Clocks

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

### 6.6 Implementation

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

## 7. TrueTime and Interval-Based Clocks

### 7.1 Google Spanner's Approach

Google Spanner (Corbett et al., 2012) takes a radically different approach: instead of abandoning physical time, it **bounds the uncertainty** of physical time using specialized hardware.

**TrueTime API**:

```
TT.now()    → TTinterval: [earliest, latest]
TT.after(t) → bool: true if t is definitely in the past
TT.before(t)→ bool: true if t is definitely in the future
```

The key guarantee: the true absolute time `t_abs` is always within the returned interval:

```
earliest ≤ t_abs ≤ latest
uncertainty ε = (latest - earliest) / 2
```

### 7.2 How TrueTime Works

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

### 7.3 Commit-Wait Protocol

Spanner uses TrueTime to implement **externally consistent** (linearizable) transactions without locking:

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

**Cost**: The commit wait adds latency equal to `2ε` (twice the uncertainty) to every transaction. With typical `ε ≈ 3.5 ms`, this adds ~7 ms of latency. This is why Google invests heavily in reducing `ε` through better hardware.

### 7.4 Interval-Based Ordering

```
Transaction T1: commit timestamp s1, uncertainty [s1 - ε, s1 + ε]
Transaction T2: commit timestamp s2, uncertainty [s2 - ε, s2 + ε]

If s1 + ε < s2 - ε:
  T1 definitely committed before T2 in real time → ordered

If intervals overlap:
  Cannot determine real-time order
  But commit-wait ensures this case does not arise for causally related transactions
```

### 7.5 Simulating TrueTime

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

## 8. Comparison of Clock Mechanisms

### 8.1 Feature Comparison

| Feature | Physical | Lamport | Vector | HLC | TrueTime |
|---------|----------|---------|--------|-----|----------|
| Size | O(1) | O(1) | O(n) | O(1) | O(1) |
| a→b ⟹ C(a)<C(b) | No | Yes | Yes | Yes | Yes |
| C(a)<C(b) ⟹ a→b | No | No | Yes | No | No |
| Detect concurrency | No | No | Yes | No | No |
| Physical time | Yes | No | No | Yes | Yes (bounded) |
| Snapshot queries | Yes* | No | No | Yes | Yes |
| External consistency | No | No | No | No | Yes |
| Hardware required | Basic | None | None | Basic | GPS+Atomic |
| Practical for n>100 | Yes | Yes | No | Yes | Yes |

`*` Physical time snapshots are unreliable due to clock skew.

### 8.2 When to Use What

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

### 8.3 Usage in Real Systems

| System | Clock Mechanism | Why |
|--------|----------------|-----|
| Amazon DynamoDB | Version vectors | Detect write conflicts across replicas |
| Google Spanner | TrueTime | External consistency without distributed locking |
| CockroachDB | HLC | Spanner-like semantics without GPS hardware |
| Apache Kafka | Lamport-like | Monotonic offset ordering within partitions |
| Riak | Dotted version vectors | Accurate conflict detection |
| MongoDB | HLC (since 3.6) | Causal consistency sessions |
| etcd | Raft log index | Total order from consensus (implicit Lamport) |

---

## 9. Code: Full Implementations

### 9.1 Comprehensive Clock Comparison

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

### 9.2 Causal Broadcast Using Vector Clocks

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

### 9.3 Matrix Clocks for Garbage Collection

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

## 10. Exercises

### Exercise 1: Compute Lamport Timestamps

Given the following event trace, compute Lamport timestamps for each event:

```
Process A: a1(send to B), a2(internal), a3(receive from C)
Process B: b1(receive from A), b2(send to C), b3(internal)
Process C: c1(internal), c2(receive from B), c3(send to A)
```

### Exercise 2: Compute Vector Clocks

For the same event trace as Exercise 1, compute vector clocks for each event. Then determine which pairs of events are concurrent.

### Exercise 3: HLC Behavior Under Clock Skew

Process A has a physical clock running 10ms fast. Process B has a physical clock running 5ms slow. At real time `t = 1000`:

- A's physical clock reads 1010
- B's physical clock reads 995

A sends a message to B that arrives at real time `t = 1002` (B's clock reads 997).

1. What HLC timestamp does A assign to the send event?
2. What HLC timestamp does B assign to the receive event?
3. Is the HLC ordering consistent with the happens-before relation?

### Exercise 4: Version Vector Conflict Detection

Three replicas (R1, R2, R3) store key "user:profile". The following writes occur:

1. R1 writes value V1, version vector becomes [1, 0, 0]
2. R2 reads from R1, then writes V2: version vector [1, 1, 0]
3. R3 writes V3 without reading any other replica: version vector [0, 0, 1]
4. Client reads from R2 (gets V2, [1, 1, 0]) and from R3 (gets V3, [0, 0, 1])

Is there a conflict? How should it be resolved?

### Exercise 5: Implementation Challenge

Implement a message logging system that uses vector clocks to detect **anomalous** message orderings -- cases where messages are delivered out of causal order. The system should:

1. Track all send and receive events across 4 processes
2. Detect when a message is delivered before a causally preceding message
3. Report the anomaly with full vector clock details
4. Implement causal buffering to fix the anomalies

---

## 11. Summary

### Key Takeaways

1. **Physical clocks are unreliable** for ordering events in distributed systems due to drift, skew, NTP limitations, and the speed of light.

2. **Happens-before (→)** defines a strict partial order on events based on potential causality: process order and message send-receive pairs.

3. **Lamport timestamps** provide a compact (O(1)) clock satisfying `a → b ⟹ C(a) < C(b)`, but cannot detect concurrency.

4. **Vector clocks** provide the complete causal picture: `a → b ⟺ V(a) < V(b)`, but require O(n) space per timestamp.

5. **HLC** combines physical time awareness with Lamport-like properties in O(1) space, making it practical for large-scale systems.

6. **TrueTime** bounds clock uncertainty with hardware, enabling external consistency through the commit-wait protocol.

7. The choice of clock mechanism depends on your consistency requirements, system scale, and available hardware.

### Clock Mechanism Decision Framework

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

## 12. References

1. Lamport, L. (1978). "Time, Clocks, and the Ordering of Events in a Distributed System." *Communications of the ACM*, 21(7), 558-565.
2. Fidge, C. J. (1988). "Timestamps in Message-Passing Systems That Preserve the Partial Ordering." *Proceedings of the 11th Australian Computer Science Conference*.
3. Mattern, F. (1989). "Virtual Time and Global States of Distributed Systems." *Parallel and Distributed Algorithms*, 215-226.
4. Kulkarni, S., Demirbas, M., et al. (2014). "Logical Physical Clocks and Consistent Snapshots in Globally Distributed Databases." *OPODIS 2014*.
5. Corbett, J. C., et al. (2012). "Spanner: Google's Globally-Distributed Database." *OSDI 2012*, 261-264.
6. Preguica, N., Baquero, C., et al. (2012). "Brief Announcement: Efficient Causality Tracking in Distributed Storage Systems with Dotted Version Vectors." *PODC 2012*.
7. Schwarz, R. & Mattern, F. (1994). "Detecting Causal Relationships in Distributed Computations." *Distributed Computing*, 7(3), 149-174.
8. Mills, D. L. (2006). *Computer Network Time Synchronization: The Network Time Protocol*. CRC Press.

---

[Next: Lesson 03 - FLP Impossibility and Theoretical Bounds](./03_FLP_Impossibility_and_Bounds.md)

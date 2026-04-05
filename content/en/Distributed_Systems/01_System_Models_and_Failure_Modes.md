# Lesson 1: System Models and Failure Modes

[Overview](./00_Overview.md) | [Next](./02_Time_Clocks_and_Ordering.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between synchronous, asynchronous, and partially synchronous system models and map each to real-world deployments
2. Formally define crash-stop, crash-recovery, and Byzantine failure models and reason about their implications for protocol design
3. Classify distributed system properties as either safety or liveness guarantees
4. Explain why the FLP impossibility result constrains consensus protocol design
5. Analyze network link models (reliable, fair-loss, arbitrary) and their relationship to process failure assumptions

---

## Table of Contents

1. [Why System Models Matter](#1-why-system-models-matter)
2. [Timing Models](#2-timing-models)
3. [Network Link Models](#3-network-link-models)
4. [Process Failure Models](#4-process-failure-models)
5. [Safety and Liveness Properties](#5-safety-and-liveness-properties)
6. [Combining Process and Link Failures](#6-combining-process-and-link-failures)
7. [FLP Preview: Why Impossibility Results Matter](#7-flp-preview-why-impossibility-results-matter)
8. [Real-World System Model Mapping](#8-real-world-system-model-mapping)
9. [Code: Simulating Failure Modes](#9-code-simulating-failure-modes)
10. [Summary and Key Takeaways](#10-summary-and-key-takeaways)
11. [Practice Problems](#11-practice-problems)
12. [References](#12-references)

---

## 1. Why System Models Matter

### The Fundamental Challenge

A distributed system is a collection of independent computing nodes that communicate by passing messages over a network. Unlike a single machine, where components share memory and a single clock, a distributed system must contend with three fundamental uncertainties:

- **Unbounded delays**: Messages can take arbitrarily long to arrive.
- **Independent failures**: Any node can fail independently of others.
- **No global clock**: There is no shared notion of "now."

Without a precise model of what *can* go wrong, it is impossible to reason about whether an algorithm is correct. System models provide the formal framework for stating assumptions, proving correctness, and understanding the limits of what is achievable.

### The Role of Abstraction

Real networks and machines are messy. Packets get corrupted, kernels panic, SSDs silently flip bits, and cloud VMs get preempted without warning. System models abstract this messiness into clean categories so that:

1. **Algorithm designers** know exactly which failures their protocol must tolerate.
2. **Implementers** know which real-world scenarios map to each failure class.
3. **Operators** can choose the right algorithm for their deployment environment.

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

### The Model Hierarchy

System models form a hierarchy from weakest (hardest to build protocols for) to strongest (easiest but least realistic):

```
Strongest assumptions (easiest to design for)
    ▲
    │  Synchronous + Crash-stop + Reliable links
    │  Partially synchronous + Crash-recovery + Fair-loss links
    │  Asynchronous + Byzantine + Arbitrary links
    ▼
Weakest assumptions (hardest to design for, most general)
```

An algorithm proven correct under weaker assumptions automatically works under stronger ones. The converse is **not** true -- an algorithm that assumes synchrony may fail catastrophically in an asynchronous environment.

---

## 2. Timing Models

The timing model specifies what assumptions we make about message delivery delays and relative processing speeds.

### 2.1 Synchronous Model

**Definition**: In a synchronous system, there exist known upper bounds on:

- **Message delay**: Every message sent is delivered within a known bound `Δ` time units.
- **Processing time**: Every process completes a computation step within a known bound `Φ` time units.
- **Clock drift**: Every process has access to a local clock with bounded drift rate `ρ` relative to real time.

Formally, for any message `m` sent at real time `t`:

```
delivery_time(m) ≤ t + Δ
```

And for any computation step starting at real time `t`:

```
completion_time(step) ≤ t + Φ
```

**Properties**:

| Property | Guarantee |
|----------|-----------|
| Message delivery | Bounded by Δ |
| Processing speed | Bounded by Φ |
| Failure detection | Perfect (just wait Δ + Φ) |
| Consensus | Solvable deterministically |
| Timeout-based detection | Always correct |

**How failure detection works**: If process `p` sends a message to process `q` and does not receive a response within `2Δ + Φ` time, then `p` can **conclusively** determine that `q` has crashed. There are no false positives.

**Real-world approximation**: A tightly controlled cluster with dedicated network switches, bounded queue depths, and real-time operating systems approaches a synchronous model. Examples:

- Hard real-time embedded systems (e.g., avionics, automotive CAN bus)
- Dedicated InfiniBand clusters with bounded latency
- FPGA-based trading systems with deterministic timing

**Limitations**: Pure synchrony is almost never achievable in commodity distributed systems. A single garbage collection pause, network congestion event, or page fault can violate the timing bound.

### 2.2 Asynchronous Model

**Definition**: In an asynchronous system, **no** timing assumptions are made:

- Messages may take arbitrarily long to be delivered (but are eventually delivered if the link is fair-loss or reliable).
- Processes may take arbitrarily long to execute a step.
- There is no relationship between local clocks at different processes.

Formally:

```
∀ bound B, ∃ execution where delivery_time(m) > B
```

**Properties**:

| Property | Guarantee |
|----------|-----------|
| Message delivery | No bound (eventual if link is fair) |
| Processing speed | No bound |
| Failure detection | **Impossible** to distinguish slow from crashed |
| Consensus | **Impossible** deterministically (FLP) |
| Timeout-based detection | Always unreliable |

**The core problem**: In an asynchronous system, you **cannot** distinguish a crashed process from a very slow one. Any timeout you set may trigger a false positive (declaring a live process dead) or a false negative (waiting too long for a dead process).

**Real-world mapping**:

- Internet-scale systems where messages traverse multiple ISPs
- Systems with unbounded garbage collection pauses (e.g., JVM without real-time GC)
- Peer-to-peer networks with heterogeneous node capabilities

**Why study it**: The asynchronous model is the **gold standard** for theoretical results. If an algorithm works in the asynchronous model, it works everywhere. The FLP impossibility theorem (Lesson 03) shows that deterministic consensus is impossible in this model, motivating the search for practical workarounds.

### 2.3 Partially Synchronous Model

**Definition** (Dwork, Lynch, Stockmeyer 1988): A partially synchronous system satisfies timing bounds, but with a caveat. There are two equivalent formulations:

**Formulation 1 (Unknown bound)**: There exists a bound `Δ`, but its value is unknown to the processes. They must design protocols that work for any `Δ`, though they do not know what it is.

**Formulation 2 (Global Stabilization Time, GST)**: There exists a time `GST` (unknown to the processes) after which the system behaves synchronously with bound `Δ`. Before `GST`, the system is fully asynchronous.

```
Formally:  ∃ GST, Δ such that:
  ∀ messages m sent at time t ≥ GST:
    delivery_time(m) ≤ t + Δ
```

**Intuition**: The network may be arbitrarily bad for a while (partitions, congestion bursts, routing loops), but *eventually* it stabilizes and messages start arriving within a bounded time.

**Properties**:

| Property | Guarantee |
|----------|-----------|
| Message delivery | Bounded after GST |
| Processing speed | Bounded after GST |
| Failure detection | Eventually accurate |
| Consensus | **Solvable** (Paxos, Raft, PBFT) |
| Safety | Holds always (even before GST) |
| Liveness | Guaranteed only after GST |

**Why it matters**: Partial synchrony is the **sweet spot** for practical distributed systems. It captures the reality that networks are usually well-behaved but occasionally experience disruptions. Most real consensus protocols (Paxos, Raft, PBFT) are designed for partial synchrony:

- **Safety** is guaranteed regardless of timing (even during network partitions).
- **Liveness** (progress) is guaranteed only after the system stabilizes.

### 2.4 Comparison Table

| Dimension | Synchronous | Partially Synchronous | Asynchronous |
|-----------|-------------|----------------------|--------------|
| Delay bound | Known Δ | ∃Δ, unknown or after GST | None |
| Processing bound | Known Φ | ∃Φ, unknown or after GST | None |
| Failure detection | Perfect | Eventually perfect | Impossible |
| Consensus | Trivially solvable | Solvable (Paxos, Raft) | Impossible (FLP) |
| Real-world example | Hard real-time | Cloud data centers | The Internet |
| Algorithm complexity | Simple | Moderate | N/A (need randomization) |

### 2.5 The Timing Spectrum in Practice

```
    Hard Real-Time    LAN Cluster    Cloud Region     Internet      Tor Network
         │                │              │               │              │
    ◄────┼────────────────┼──────────────┼───────────────┼──────────────┼────►
    Synchronous    ≈Synchronous    Partial Sync      Async       Adversarial
                                                                   Async
```

Most production systems operate in the "partial synchrony" zone. The key design principle:

> **Design for safety under asynchrony; rely on partial synchrony only for liveness.**

---

## 3. Network Link Models

Communication between processes happens through network links. The link model specifies what can go wrong with message transmission.

### 3.1 Reliable Links

**Definition**: If a correct process `p` sends a message `m` to a correct process `q`, then `q` eventually delivers `m`. Moreover:

1. **No duplication**: `m` is delivered at most once.
2. **No creation**: If `q` delivers `m`, then `p` previously sent `m`.
3. **Reliable delivery**: If `p` is correct and sends `m` to correct `q`, then `q` eventually delivers `m`.

```
Process p ──── m ────► Process q     (always delivered, exactly once)
```

**How to build it**: Reliable links can be built on top of fair-loss links using sequence numbers and retransmission:

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

### 3.2 Fair-Loss Links

**Definition**: A fair-loss link makes three guarantees:

1. **Fair loss**: If a correct process `p` sends a message `m` to a correct process `q` infinitely often, then `q` delivers `m` infinitely often.
2. **Finite duplication**: If `p` sends `m` a finite number of times, then `q` delivers `m` a finite number of times.
3. **No creation**: Same as reliable links.

**Intuition**: Any single message may be lost, but if you keep retransmitting, it will eventually get through. This models UDP over a non-adversarial network.

```
Process p ──── m ────► Process q     (may be lost)
Process p ──── m ────► Process q     (may be lost)
Process p ──── m ────► Process q     (delivered!)
```

### 3.3 Arbitrary (Adversarial) Links

**Definition**: No guarantees at all. Messages can be:

- Lost
- Duplicated
- Reordered
- Modified (corrupted)
- Fabricated out of thin air (spoofed)

This models an adversarial network where an attacker has full control over the communication channel.

**Mitigation**: Use cryptographic techniques:

- **Integrity**: Message Authentication Codes (MACs) or digital signatures prevent modification and fabrication.
- **Confidentiality**: Encryption prevents eavesdropping.
- **Replay protection**: Nonces and sequence numbers prevent replay attacks.

With cryptographic protections, an arbitrary link can be reduced to a fair-loss link (assuming the adversary cannot break the cryptographic primitives).

### 3.4 Link Model Comparison

| Property | Reliable | Fair-Loss | Arbitrary |
|----------|----------|-----------|-----------|
| Message loss | No | Yes (finite) | Yes |
| Duplication | No | Yes (finite) | Yes (infinite) |
| Corruption | No | No | Yes |
| Fabrication | No | No | Yes |
| Reordering | Possible | Possible | Yes |
| Built from | Fair-loss + retransmit | Physical layer | Physical layer |
| Real-world | TCP (approx.) | UDP | Open Internet |

### 3.5 Network Partitions

A **network partition** occurs when the network splits into two or more groups of nodes that can communicate within each group but not between groups.

```
┌─────────────────┐         ┌─────────────────┐
│  Partition A     │   ✕✕✕   │  Partition B     │
│  Node 1          │ ◄─────► │  Node 3          │
│  Node 2          │  NO     │  Node 4          │
│                  │  COMM   │  Node 5          │
└─────────────────┘         └─────────────────┘
```

Partitions are a form of link failure, not process failure. All nodes are still alive and processing, but they cannot communicate across the partition boundary. This is precisely the scenario that the CAP theorem addresses (Lesson 04).

**Partial partitions** are also possible: node A can reach node B, node B can reach node C, but node A cannot reach node C directly.

---

## 4. Process Failure Models

A **process failure model** specifies how individual nodes can deviate from their correct behavior.

### 4.1 Crash-Stop Failures

**Definition**: A process executes its algorithm correctly until some point in time, at which it **permanently stops** executing. Once crashed, it never recovers.

Formally, using a simplified process algebra notation:

```
Process behavior: p ::= action.p | STOP

Correct process: always eventually takes the next action
Crashed process: transitions to STOP and remains there permanently

Timeline:
  ─────────────────────┬───────────────────
    correct execution  │  STOP (permanent)
                     crash
```

**Properties**:

- Before crashing, the process follows its algorithm perfectly.
- After crashing, the process sends no messages and takes no steps.
- Other processes cannot distinguish "crashed" from "very slow" in asynchronous models.
- The process does NOT recover (this is the key distinction from crash-recovery).

**Fault tolerance requirement**: A system of `n` processes can tolerate up to `f` crash-stop failures if `n ≥ 2f + 1` for most consensus protocols.

**Real-world mapping**:

- Process killed by the OS (OOM killer, segfault)
- Hardware failure without redundant storage (dead SSD, burnt motherboard)
- VM terminated by the cloud provider without restart

### 4.2 Crash-Recovery Failures

**Definition**: A process may crash and later **recover**, resuming execution. Upon recovery, the process loses all in-memory state but may have access to persistent (stable) storage that survives crashes.

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

**Stable storage abstraction**: The process can write to stable storage that persists across crashes. Upon recovery, it reads the stable state and resumes.

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

**Key insight**: In the crash-recovery model, the **persistence discipline** determines correctness. A process that responds to a vote request before persisting its vote could violate safety upon recovery.

**Fault tolerance**: With crash-recovery and stable storage, a process that crashes and recovers is treated as one that was "slow" rather than "dead." This means protocols can tolerate more failures, but at the cost of:
- Higher latency (every state change must be written to disk before responding)
- More complex recovery logic

### 4.3 Byzantine Failures

**Definition**: A process with a Byzantine failure can exhibit **arbitrary behavior**. It can:

- Send conflicting messages to different processes
- Lie about its state
- Collude with other Byzantine processes
- Remain silent (subsumes crash failures)
- Follow the protocol perfectly (the hardest case to detect)

Named after the **Byzantine Generals Problem** (Lamport, Shostak, Pease 1982).

```
Correct process: follows the protocol exactly
Byzantine process: ANY behavior is possible

Examples of Byzantine behavior:
  - Send "YES" to node A and "NO" to node B for the same proposal
  - Claim to have received a message that was never sent
  - Delay responses strategically to cause maximum disruption
  - Forge messages from other processes (without cryptography)
```

**Formal definition** using a process algebra-like notation:

```
Correct process:  p_correct ::= receive(m). f(m). send(f(m)). p_correct
Byzantine process: p_byz ::= receive(m). send(ANYTHING). p_byz
                          |  send(ANYTHING). p_byz        (no input needed)
                          |  STOP. p_byz                  (crash)
```

**Fault tolerance bounds**:

| Property | Required Nodes | Formula |
|----------|----------------|---------|
| Byzantine Agreement (unsigned messages) | n ≥ 3f + 1 | Cannot tolerate f ≥ n/3 |
| Byzantine Agreement (signed messages) | n ≥ 2f + 1 | Digital signatures help |
| Byzantine Broadcast | n ≥ 3f + 1 | Same as agreement |

The `3f + 1` bound is **tight** -- it is both necessary and sufficient for Byzantine agreement with oral (unsigned) messages.

**Intuition for 3f + 1**: Consider `f = 1` Byzantine node. We need `n ≥ 4`. With 3 nodes and 1 Byzantine, the two correct nodes cannot distinguish which of the other two is lying:

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

**Real-world causes of Byzantine behavior**:

- Software bugs that cause incorrect state transitions
- Hardware errors (bit flips, firmware bugs)
- Compromised nodes (hacked servers)
- Malicious participants (blockchain networks)
- Configuration errors causing split-brain behavior

### 4.4 Failure Model Hierarchy

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

A protocol that tolerates Byzantine failures automatically tolerates crash failures (since a crash is one possible Byzantine behavior). A protocol designed for crash-stop may catastrophically fail with Byzantine faults.

### 4.5 Failure Model Comparison Table

| Dimension | Crash-Stop | Crash-Recovery | Byzantine |
|-----------|------------|----------------|-----------|
| Behavior before failure | Correct | Correct | Correct (or not) |
| Failure action | Stop forever | Stop, then recover | Anything |
| State after failure | Lost | Stable storage preserved | Corrupted |
| Can lie? | No | No | Yes |
| Detection difficulty | Medium | Medium | Very hard |
| Min nodes for f faults | 2f + 1 | 2f + 1 (with storage) | 3f + 1 |
| Protocol complexity | Low | Medium | High |
| Message complexity | O(n) to O(n²) | O(n²) | O(n³) typical |
| Real-world example | OOM kill | Server reboot | Hacked node |
| Key protocols | Paxos, Raft | Paxos, Raft | PBFT, HotStuff |

---

## 5. Safety and Liveness Properties

Every correctness property of a distributed system can be classified as either a **safety** property or a **liveness** property. This classification, due to Alpern and Schneider (1985), is one of the most fundamental distinctions in distributed systems theory.

### 5.1 Safety Properties

**Definition**: A safety property states that "nothing bad happens." Formally, a property `P` is a safety property if and only if:

- Every violation of `P` has a **finite prefix** -- there is a specific point in the execution where the property was violated.
- Once violated, it cannot become un-violated (the bad thing already happened).

```
Safety property violation:
  ──────────────────────X────────────────
  correct execution    │ violation (irrecoverable)
                       │
                  finite prefix that
                  demonstrates violation
```

**Examples of safety properties**:

| Safety Property | "Bad thing" it prevents |
|----------------|------------------------|
| Agreement | Two correct processes decide different values |
| Validity | A decided value was never proposed |
| Integrity | A message is delivered more than once |
| Mutual exclusion | Two processes hold the lock simultaneously |
| Consistency (linearizability) | Operations appear out of real-time order |
| No data loss | Acknowledged write is lost |

**Key insight**: Safety properties can be violated in finite time but can never be "achieved" in finite time -- you must observe the entire execution to confirm that a safety property holds.

### 5.2 Liveness Properties

**Definition**: A liveness property states that "something good eventually happens." Formally:

- Every finite execution prefix can be extended to satisfy the property.
- No finite prefix can violate a liveness property (there is always hope).

```
Liveness property:
  ──────────────────────────────────?─────
  no progress so far...            │
                              but still hope!
                              (could be satisfied
                               in the future)
```

**Examples of liveness properties**:

| Liveness Property | "Good thing" that eventually happens |
|------------------|--------------------------------------|
| Termination | Every correct process eventually decides |
| Eventual delivery | Every sent message is eventually delivered |
| Progress | If a process requests a lock, it eventually gets it |
| Availability | Every request eventually receives a response |
| Eventual consistency | All replicas eventually converge |

### 5.3 The Safety-Liveness Decomposition Theorem

**Theorem** (Alpern & Schneider, 1985): Every correctness property of a distributed system can be expressed as the intersection of a safety property and a liveness property.

```
Any property P = Safety(P) ∩ Liveness(P)
```

**Example -- Consensus**:

```
Consensus = Safety ∩ Liveness

Safety component:
  - Agreement: No two correct processes decide differently
  - Validity: The decided value was proposed by some process

Liveness component:
  - Termination: Every correct process eventually decides
```

### 5.4 The Fundamental Trade-off

In asynchronous systems with failures, you **cannot** simultaneously guarantee both safety and liveness. This is the essence of the FLP impossibility result (Lesson 03).

The practical consequence for protocol design:

> **Always preserve safety. Sacrifice liveness when necessary.**

| Approach | Safety | Liveness | Example |
|----------|--------|----------|---------|
| Paxos/Raft | Always guaranteed | Guaranteed after GST | Most databases |
| Optimistic replication | Eventual (CRDTs) | Always guaranteed | Collaborative editing |
| Blockchain (PoW) | Probabilistic | Probabilistic | Bitcoin |

This means that a Paxos-based system may stop making progress during a network partition (sacrificing liveness), but it will **never** return an incorrect result (preserving safety). This is exactly the right trade-off for a database.

### 5.5 Formal Classification Exercise

Classify each of the following as safety (S), liveness (L), or neither (N):

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

## 6. Combining Process and Link Failures

Real systems experience combinations of process and link failures. The interaction between these failure modes determines what is achievable.

### 6.1 Failure Combination Matrix

| Process Model | Link Model | Achievable | Example System |
|---------------|------------|------------|----------------|
| Crash-stop | Reliable | Consensus (2f+1) | LAN cluster |
| Crash-stop | Fair-loss | Consensus (2f+1, with retransmit) | WAN replicas |
| Crash-recovery | Reliable | Consensus (2f+1, with stable storage) | Database replicas |
| Crash-recovery | Fair-loss | Consensus (harder, need persistent state + retransmit) | Cloud databases |
| Byzantine | Reliable | Consensus (3f+1) | Permissioned blockchain |
| Byzantine | Arbitrary | Consensus (3f+1, with crypto) | Public blockchain |

### 6.2 Equivalent Configurations

Some combinations are equivalent in terms of achievability:

**Claim**: A crash-stop process with fair-loss links is equivalent to a crash-stop process with reliable links (in terms of what problems are solvable).

**Proof sketch**: We can build a reliable link on top of a fair-loss link using retransmission (as shown in Section 3.1). Since the construction only requires the sending process to be alive (not crashed), it works for correct processes. Crashed processes do not need to send or receive anyway.

**Claim**: A Byzantine process with reliable links is **not** equivalent to a crash-stop process with arbitrary links.

**Reason**: Byzantine processes can actively send misleading messages, while arbitrary links can only corrupt or lose messages. A Byzantine process can engage in strategic deception that link-level corruption cannot model.

### 6.3 The Network Partition vs. Node Crash Ambiguity

In an asynchronous system with crash-stop failures and fair-loss links:

```
Scenario 1: Node B crashed
  A ────────► B (dead)     No response

Scenario 2: Network partition
  A ────✕───► B (alive)    No response (messages lost)

Scenario 3: Node B is slow
  A ────────► B (alive)    Response coming... eventually
```

From A's perspective, all three scenarios are **indistinguishable**. This is why:

- Failure detection is impossible in asynchronous systems.
- Timeouts create false positives (declaring a live node dead).
- The CAP theorem forces a choice between consistency and availability during partitions.

---

## 7. FLP Preview: Why Impossibility Results Matter

### 7.1 The Consensus Problem

The **consensus problem** requires a set of processes to agree on a single value:

1. **Agreement**: All correct processes decide the same value.
2. **Validity**: The decided value was proposed by some process.
3. **Termination**: Every correct process eventually decides.

### 7.2 The FLP Impossibility Result (Preview)

**Theorem** (Fischer, Lynch, Paterson 1985): In an asynchronous system with reliable links, there is **no** deterministic protocol that solves consensus if even **one** process may crash.

```
Asynchronous + Deterministic + Even 1 crash → Consensus IMPOSSIBLE

Formally:
  ¬∃ protocol P: (asynchronous ∧ deterministic ∧ f ≥ 1) → consensus
```

This is NOT saying:
- Consensus is impossible in practice (it is not)
- We should give up on distributed systems
- No useful work can be done

This IS saying:
- Every correct consensus protocol must use at least one of:
  - **Randomization** (e.g., Ben-Or's protocol)
  - **Timing assumptions** (e.g., partial synchrony in Paxos/Raft)
  - **Failure detectors** (e.g., Chandra-Toueg's oracle model)
- Any protocol that claims to solve consensus deterministically in a purely asynchronous system is **wrong**.

### 7.3 Why This Matters for Practitioners

| Protocol | How it circumvents FLP |
|----------|----------------------|
| Paxos | Assumes partial synchrony for liveness |
| Raft | Uses timeouts (partial synchrony) |
| PBFT | Assumes partial synchrony for view changes |
| Bitcoin PoW | Randomization (mining) + probabilistic safety |
| Ben-Or | Randomized coin flips |
| Chandra-Toueg | Assumes failure detector oracle (◇S) |

We will prove FLP rigorously in Lesson 03 and explore each circumvention strategy.

---

## 8. Real-World System Model Mapping

### 8.1 Cloud Provider Models

| System | Timing Model | Failure Model | Link Model | Notes |
|--------|-------------|---------------|------------|-------|
| AWS DynamoDB | Partial sync | Crash-recovery | Reliable (within AZ) | Sloppy quorums, hinted handoff |
| Google Spanner | Partial sync + TrueTime | Crash-recovery | Reliable (within region) | TrueTime provides bounded uncertainty |
| Azure Cosmos DB | Partial sync | Crash-recovery | Reliable | 5 consistency levels |
| Apache ZooKeeper | Partial sync | Crash-recovery | Fair-loss (TCP retry) | ZAB protocol |
| etcd | Partial sync | Crash-recovery | Reliable (TCP) | Raft-based |
| CockroachDB | Partial sync | Crash-recovery | Reliable | Raft + MVCC |

### 8.2 Blockchain Models

| System | Timing Model | Failure Model | Link Model | Notes |
|--------|-------------|---------------|------------|-------|
| Bitcoin | **Synchronous** (assumes max block propagation time) | **Byzantine** (up to 50% hash power) | Fair-loss (gossip) | Nakamoto consensus |
| Ethereum (PoS) | Partial sync | Byzantine (up to 1/3 validators) | Fair-loss | Casper FFG + LMD GHOST |
| Tendermint | Partial sync | Byzantine (up to 1/3) | Reliable (with retransmit) | BFT + DPoS |
| Hyperledger Fabric | Partial sync | Crash (CFT mode) or Byzantine (BFT mode) | Reliable | Pluggable consensus |

### 8.3 Key Observations

**AWS model**: AWS services typically assume crash-recovery failures within a data center and treat entire availability zone failures as crash-stop. Network links within an AZ are treated as reliable (thanks to redundant switching), while cross-region links are treated as fair-loss.

**Bitcoin model**: Bitcoin makes a **synchronous** assumption: blocks propagate to all nodes within a bounded time (roughly 10 minutes). If this assumption is violated (e.g., by network-level attacks), Bitcoin's safety guarantees weaken. The "6 confirmation" rule is a practical adaptation to the bounded-delay assumption.

**Google Spanner model**: Spanner's TrueTime API provides an interval `[earliest, latest]` for the current time, with the guarantee that the true time falls within this interval. This turns the clock uncertainty problem into a **bounded wait** problem -- transactions simply wait out the uncertainty interval to ensure global ordering.

---

## 9. Code: Simulating Failure Modes

### 9.1 Message-Passing Framework

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

### 9.2 Running a Failure Simulation

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

### 9.3 Sample Output Analysis

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

**Observation**: With crash-stop failures, the crashed node simply does not participate. The remaining 4 nodes (including the broadcaster) agree on the value. With `n = 5` and `f = 1`, we have `n = 5 ≥ 2(1) + 1 = 3`, so consensus is achievable.

### 9.4 Failure Detector Simulation

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

### 9.5 Demonstrating the Impossibility of Perfect Detection

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

## 10. Summary and Key Takeaways

### System Model Cheat Sheet

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

### Key Principles

1. **Model before you build**: Always explicitly state your system model assumptions before designing a protocol.
2. **Weaker assumptions = stronger guarantees**: Algorithms proven correct under weaker models work in more environments.
3. **Safety over liveness**: When you cannot have both (FLP), always preserve safety.
4. **Partial synchrony is the practical sweet spot**: It captures real-world behavior and enables consensus.
5. **Byzantine tolerance is expensive**: The jump from crash to Byzantine tolerance costs both nodes (3f+1 vs 2f+1) and messages (O(n^3) vs O(n^2)).

---

## 11. Practice Problems

### Problem 1: Model Classification

For each system below, identify the most appropriate timing model, failure model, and link model:

1. A cluster of 5 servers in the same rack, connected via a dedicated switch, running a database
2. A peer-to-peer file sharing network spanning 10,000 nodes worldwide
3. A blockchain network where any participant can join or leave
4. A real-time control system for an autonomous vehicle
5. Three data centers connected by leased lines, replicating a financial database

### Problem 2: Safety vs Liveness

Classify each property and explain your reasoning:

1. "No two ATMs dispense money for the same withdrawal"
2. "Every ATM withdrawal request is eventually processed"
3. "The bank balance is always non-negative"
4. "Every deposited check is eventually credited"
5. "A transferred amount is debited from the source before being credited to the destination"

### Problem 3: Failure Tolerance Calculation

A system has 7 nodes. Calculate the maximum number of faulty nodes `f` it can tolerate under:

1. Crash-stop failures (consensus requires n ≥ 2f + 1)
2. Byzantine failures with unsigned messages (requires n ≥ 3f + 1)
3. Byzantine failures with digital signatures (requires n ≥ 2f + 1)

### Problem 4: Code Challenge

Extend the simulation code from Section 9 to:

1. Implement a stubborn retransmission layer that converts fair-loss links to reliable links
2. Add a Byzantine failure mode where a faulty node sends the correct value to a majority and a wrong value to the minority (making detection harder)
3. Implement an eventually perfect failure detector that tracks its accuracy over time

### Problem 5: Real-World Analysis

Read the Jepsen analysis of a distributed database of your choice (https://jepsen.io/analyses). Answer:

1. What system model does the database claim to operate under?
2. What system model does it actually operate under (based on Jepsen's findings)?
3. What safety violations were discovered?
4. Were these violations caused by incorrect model assumptions or implementation bugs?

---

## 12. References

1. Fischer, M. J., Lynch, N. A., & Paterson, M. S. (1985). "Impossibility of Distributed Consensus with One Faulty Process." *Journal of the ACM*, 32(2), 374-382.
2. Lamport, L., Shostak, R., & Pease, M. (1982). "The Byzantine Generals Problem." *ACM Transactions on Programming Languages and Systems*, 4(3), 382-401.
3. Dwork, C., Lynch, N., & Stockmeyer, L. (1988). "Consensus in the Presence of Partial Synchrony." *Journal of the ACM*, 35(2), 288-323.
4. Alpern, B., & Schneider, F. B. (1985). "Defining Liveness." *Information Processing Letters*, 21(4), 181-185.
5. Chandra, T. D., & Toueg, S. (1996). "Unreliable Failure Detectors for Reliable Distributed Systems." *Journal of the ACM*, 43(2), 225-267.
6. Cachin, C., Guerraoui, R., & Rodrigues, L. (2011). *Introduction to Reliable and Secure Distributed Programming*. Springer.
7. Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly Media.

---

[Next: Lesson 02 - Time, Clocks, and Ordering](./02_Time_Clocks_and_Ordering.md)

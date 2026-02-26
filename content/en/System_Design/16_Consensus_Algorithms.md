# Consensus Algorithms

**Previous**: [Distributed Systems Concepts](./15_Distributed_Systems_Concepts.md) | **Next**: [Practical Design Examples 1](./17_Design_Example_1.md)

Difficulty: ⭐⭐⭐⭐

## Learning Objectives

1. Define the consensus problem and its three correctness properties (agreement, validity, termination)
2. Explain the roles (proposer, acceptor, learner) and two-phase protocol of Basic Paxos, including how it handles competing proposals
3. Describe Raft's leader election, log replication, and safety mechanisms, and explain why Raft was designed to be more understandable than Paxos
4. Distinguish crash fault tolerance (CFT) from Byzantine fault tolerance (BFT) and state the minimum node counts required for each
5. Compare practical consensus implementations (ZooKeeper ZAB, etcd Raft, Consul) and their use cases for coordination, configuration, and leader election
6. Trace through consensus protocol execution scenarios including leader failure, split vote, and network partition

---

Consensus is the beating heart of every reliable distributed system: it is how a database replication cluster agrees on which write committed first, how a Kubernetes control plane elects a leader, and how a blockchain validates transactions. Getting consensus wrong leads to split-brain scenarios, data loss, or silent corruption. This lesson demystifies Paxos and Raft -- algorithms that underpin tools you likely use every day -- so you can reason about their guarantees and limitations in production.

## Table of Contents

1. [Defining the Consensus Problem](#1-defining-the-consensus-problem)
2. [Paxos Algorithm](#2-paxos-algorithm)
3. [Raft Algorithm](#3-raft-algorithm)
4. [Byzantine Fault Tolerance](#4-byzantine-fault-tolerance)
5. [Practical Applications](#5-practical-applications)
6. [Practice Problems](#6-practice-problems)

---

## 1. Defining the Consensus Problem

### What is Consensus?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Consensus Problem                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Goal: N nodes agree on a single value                                 │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │   Node 1: "A"                                                   │   │
│  │   Node 2: "B"        ───►    All nodes: "A" (or "B")            │   │
│  │   Node 3: "A"                                                   │   │
│  │   Node 4: "C"                                                   │   │
│  │   Node 5: (failed)                                              │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Use Cases:                                                            │
│  - Leader election: Who is the leader?                                 │
│  - Distributed locks: Who acquired the lock?                           │
│  - Distributed DB: What order to commit transactions?                  │
│  - State machine replication: What command to execute?                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Properties of Consensus

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Required Properties of Consensus Algorithms         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Safety - Nothing bad happens                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Agreement:                                                      │   │
│  │  - All correct nodes agree on the same value                    │   │
│  │  - Two nodes must not decide on different values                │   │
│  │                                                                  │   │
│  │  Validity:                                                       │   │
│  │  - The decided value must be one that some node proposed        │   │
│  │  - Cannot select a value no one proposed                        │   │
│  │                                                                  │   │
│  │  Integrity:                                                      │   │
│  │  - Each node decides at most once                               │   │
│  │  - Once decided, cannot change                                  │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  2. Liveness - Good things eventually happen                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Termination:                                                    │   │
│  │  - All correct nodes eventually decide on a value               │   │
│  │  - Don't wait forever                                           │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  FLP Impossibility (1985):                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  It is impossible to guarantee both Safety and Liveness in     │   │
│  │  an asynchronous system while tolerating even a single         │   │
│  │  node failure.                                                   │   │
│  │                                                                  │   │
│  │  Real system responses:                                         │   │
│  │  - Sacrifice some Liveness (retry after timeout)                │   │
│  │  - Add synchrony assumptions (partial synchrony)               │   │
│  │  - Safety is always guaranteed                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

> **Intuition behind FLP**: In a fully asynchronous network, there is no upper bound on message delivery time. This means a slow node is indistinguishable from a crashed node — you can never be certain whether the message is still in transit or the sender has failed. Because of this fundamental ambiguity, no algorithm can guarantee it will always terminate (liveness) while also never deciding on conflicting values (safety). Real systems work around FLP by assuming *partial synchrony* — messages are eventually delivered within some unknown but finite bound — which allows practical algorithms like Paxos and Raft to work correctly in practice.

### Failure Models

Let **f** = the number of faults the system must tolerate.

| Model | Description | Required Nodes |
|-------|-------------|----------------|
| Crash Failure | Node stops, can recover | 2f + 1 |
| Omission Failure | Messages can be lost | 2f + 1 |
| Byzantine Failure | Malicious/arbitrary behavior | 3f + 1 |

**Quorum intuition**: With `2f + 1` nodes, even if `f` crash, the remaining `f + 1` still form a strict majority that can make progress. For Byzantine faults, `3f + 1` is needed because malicious nodes can send conflicting messages, so you need enough honest nodes to outvote the adversaries.

---

## 2. Paxos Algorithm

> **Analogy -- Meeting Minutes by Majority Vote**
>
> Imagine five colleagues in different offices trying to agree on a single meeting time via email (no phone, no chat). One person **proposes** "Tuesday 3 PM" and asks everyone to tentatively accept. If a majority (3 out of 5) reply "OK, I'll pencil that in," the proposer sends a **confirm** message, and the time is final. If two people simultaneously propose different times, the group resolves it by always deferring to the proposal with the higher sequence number -- exactly the mechanism Paxos and Raft use to break ties among competing leaders.

### Paxos Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Paxos Algorithm                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Invented by Leslie Lamport in 1989 (published 1998)                   │
│  The foundational algorithm for distributed consensus                   │
│                                                                         │
│  Roles:                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Proposer                                                       │   │
│  │  - Proposes values                                              │   │
│  │  - Handles client requests                                      │   │
│  │                                                                  │   │
│  │  Acceptor                                                       │   │
│  │  - Votes on proposals                                           │   │
│  │  - Decision made when majority accepts                          │   │
│  │  - Stores state                                                 │   │
│  │                                                                  │   │
│  │  Learner                                                        │   │
│  │  - Learns the decided value                                     │   │
│  │  - Read-only replica                                            │   │
│  │                                                                  │   │
│  │  (In practice, one node can play multiple roles)                │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Basic Paxos: Two Phases

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Basic Paxos: Phase 1 - Prepare                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Proposer                   Acceptors (majority needed)                │
│      │                    A1      A2      A3                           │
│      │                    │       │       │                            │
│      │───Prepare(n=1)────►│       │       │                            │
│      │───Prepare(n=1)────────────►│       │                            │
│      │───Prepare(n=1)────────────────────►│                            │
│      │                    │       │       │                            │
│      │                    │       │       │                            │
│      │◄──Promise(n=1)────│       │       │                            │
│      │◄──Promise(n=1)────────────│       │                            │
│      │◄──Promise(n=1)────────────────────│                            │
│      │                    │       │       │                            │
│                                                                         │
│  Prepare(n): "I intend to propose with proposal number n"              │
│  Promise(n): "I will ignore proposals smaller than n"                  │
│              + return any previously accepted value                    │
│                                                                         │
│  Acceptor Rules:                                                       │
│  - If received n > existing promised_n, send Promise                   │
│  - Otherwise ignore/reject                                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     Basic Paxos: Phase 2 - Accept                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  After receiving majority Promises:                                    │
│                                                                         │
│  Proposer                   Acceptors                                  │
│      │                    A1      A2      A3                           │
│      │                    │       │       │                            │
│      │──Accept(n=1,v="X")─►│       │       │                            │
│      │──Accept(n=1,v="X")─────────►│       │                            │
│      │──Accept(n=1,v="X")─────────────────►│                            │
│      │                    │       │       │                            │
│      │◄──Accepted(n=1)────│       │       │                            │
│      │◄──Accepted(n=1)────────────│       │                            │
│      │◄──Accepted(n=1)────────────────────│                            │
│      │                    │       │       │                            │
│      ▼                                                                  │
│  Majority Accepted → Value "X" decided!                                │
│                                                                         │
│  Accept(n, v): "Please accept value v with proposal number n"          │
│  Accepted(n): "Accepted"                                               │
│                                                                         │
│  Value Selection Rule:                                                 │
│  - If a previous value was received in Promise, use the one with      │
│    the highest n                                                       │
│  - Otherwise propose your own value                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Simplified single-decree Paxos pseudocode:**

```python
class PaxosProposer:
    def __init__(self, node_id, acceptors):
        self.node_id = node_id
        self.acceptors = acceptors
        self.proposal_number = 0

    def propose(self, value):
        while True:
            # Generate a unique, monotonically increasing proposal number.
            # Higher numbers always win — this is how Paxos breaks ties.
            self.proposal_number += 1
            n = self.proposal_number

            # Phase 1 — Prepare: ask acceptors to promise not to accept
            # any proposal numbered lower than n
            promises = []
            for acceptor in self.acceptors:
                resp = acceptor.prepare(n)
                if resp.promised:
                    promises.append(resp)

            # Need a majority to proceed — any two quorums overlap by
            # at least one node, which prevents conflicting decisions
            if len(promises) < len(self.acceptors) // 2 + 1:
                continue  # Retry with higher n

            # If any acceptor already accepted a value, we MUST propose
            # that value instead of ours. This rule guarantees safety:
            # once a value is chosen, all future proposals converge on it.
            prev = [p for p in promises if p.accepted_value is not None]
            if prev:
                value = max(prev, key=lambda p: p.accepted_n).accepted_value

            # Phase 2 — Accept: ask acceptors to accept (n, value)
            accepted = sum(
                1 for a in self.acceptors if a.accept(n, value)
            )

            if accepted >= len(self.acceptors) // 2 + 1:
                return value  # Consensus reached!
            # Else: a higher-n proposal preempted us; loop and retry
```

### Paxos Conflict Resolution

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Paxos: Concurrent Proposal Handling                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Scenario: Two Proposers propose simultaneously                        │
│                                                                         │
│  Proposer1     Proposer2          Acceptors                            │
│      │             │            A1      A2      A3                     │
│      │             │             │       │       │                     │
│      │─Prepare(1)─►│             │       │       │                     │
│      │             │─Prepare(2)─►│       │       │                     │
│      │             │             │       │       │                     │
│      │◄─Promise(1)─┼─────────────│       │       │   A1: n=1          │
│      │             │◄─Promise(2)─────────│       │   A2: n=2          │
│      │             │◄─Promise(2)─────────────────│   A3: n=2          │
│      │             │             │       │       │                     │
│      │             │             │       │       │                     │
│      │─Accept(1,X)►│             │       │       │                     │
│      │             │             X NACK! (n=1 < promised n=2)          │
│      │             │             │       │       │                     │
│      │             │─Accept(2,Y)►│       │       │                     │
│      │             │◄─Accepted───────────│       │                     │
│      │             │◄─Accepted───────────────────│                     │
│      │             │             │       │       │                     │
│      │             ▼             │       │       │                     │
│      │         Value "Y" decided │       │       │                     │
│      │             │             │       │       │                     │
│      │             │             │       │       │                     │
│  Proposer1 needs to retry with higher n                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Multi-Paxos

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Multi-Paxos                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Basic Paxos Problem: Prepare needed every time (2 RTT)                │
│                                                                         │
│  Multi-Paxos Optimization:                                             │
│  - Elect a stable leader                                               │
│  - Leader decides consecutive values                                   │
│  - Skip Prepare phase (1 RTT)                                          │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Leader           Acceptors                                      │   │
│  │    │           A1      A2      A3                               │   │
│  │    │            │       │       │                               │   │
│  │    │──Prepare(n=1)────►│       │  (only once initially)        │   │
│  │    │◄─Promise(n=1)─────│       │                                │   │
│  │    │            │       │       │                               │   │
│  │    │            │       │       │                               │   │
│  │    │──Accept(1,v1)────►│       │  Log Entry 1                   │   │
│  │    │◄─Accepted─────────│       │                                │   │
│  │    │            │       │       │                               │   │
│  │    │──Accept(1,v2)────►│       │  Log Entry 2                   │   │
│  │    │◄─Accepted─────────│       │                                │   │
│  │    │            │       │       │                               │   │
│  │    │──Accept(1,v3)────►│       │  Log Entry 3                   │   │
│  │    │◄─Accepted─────────│       │                                │   │
│  │    │            │       │       │                               │   │
│  │                                                                  │   │
│  │  While leader is stable, only Accept repeats (1 RTT)            │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Raft Algorithm

### Raft Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Raft Algorithm                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  "In Search of an Understandable Consensus Algorithm" (2014)           │
│  Diego Ongaro, John Ousterhout                                         │
│                                                                         │
│  Design Goal: An understandable consensus algorithm                    │
│  - Same performance/safety as Paxos                                    │
│  - Much easier to understand                                           │
│                                                                         │
│  Key Concept Separation:                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  1. Leader Election                                              │   │
│  │  2. Log Replication                                              │   │
│  │  3. Safety                                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Node States:                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │           Timeout           Majority vote                        │   │
│  │  ┌────────┐   ───────►   ┌────────────┐   ──────►   ┌────────┐  │   │
│  │  │Follower│              │ Candidate  │             │ Leader │  │   │
│  │  └────────┘   ◄───────   └────────────┘   ◄──────   └────────┘  │   │
│  │              Higher Term        Timeout     Higher Term         │   │
│  │                                (re-election)                     │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Term

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Raft: Term                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Term: Logical time unit                                               │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │   Term 1    │   Term 2    │  Term 3  │    Term 4    │           │   │
│  │  ┌───────┐  │  ┌───────┐  │  ┌───┐   │  ┌─────────┐ │           │   │
│  │  │Election│  │  │Election│  │  │Elec│  │  │Election │ │           │   │
│  │  │       │  │  │       │  │  │fail│  │  │         │ │           │   │
│  │  │Node 1 │  │  │Node 3 │  │  │    │  │  │ Node 2  │ │           │   │
│  │  │Leader │  │  │Leader │  │  │    │  │  │ Leader  │ │           │   │
│  │  └───────┘  │  └───────┘  │  └───┘   │  └─────────┘ │           │   │
│  │             │             │          │              │           │   │
│  │  ───────────┼─────────────┼──────────┼──────────────┼───►      │   │
│  │                                                       Time      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Term Rules:                                                           │
│  - At most one leader per Term                                         │
│  - Term starts: new election begins                                    │
│  - On discovering higher Term, immediately become Follower             │
│  - Ignore messages from older Terms                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Leader Election

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Raft: Leader Election                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Step 1: Election Timeout                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Node A (Follower)                                               │   │
│  │  │                                                               │   │
│  │  │─────────────────────────────────────────────►                 │   │
│  │  │           No heartbeat (150-300ms)         │                 │   │
│  │  │                                             ▼                 │   │
│  │  │                                      Timeout!                │   │
│  │  │                                      → Become Candidate      │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Step 2: Request Vote                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Node A (Candidate)              Node B        Node C            │   │
│  │  term: 2                         term: 1       term: 1           │   │
│  │       │                            │             │               │   │
│  │       │── RequestVote(term=2) ────►│             │               │   │
│  │       │── RequestVote(term=2) ──────────────────►│               │   │
│  │       │                            │             │               │   │
│  │       │◄── VoteGranted ────────────│             │               │   │
│  │       │◄── VoteGranted ──────────────────────────│               │   │
│  │       │                            │             │               │   │
│  │       ▼                                                          │   │
│  │  Majority (2/3) achieved → Leader elected!                       │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Voting Rules:                                                         │
│  - Vote at most once per Term                                          │
│  - Only vote if requester's log is at least as up-to-date             │
│  - Reset timer after voting                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Log Replication

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Raft: Log Replication                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Leader's Log:                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Index:  1       2       3       4       5       6              │   │
│  │        ┌─────┬─────┬─────┬─────┬─────┬─────┐                   │   │
│  │  Term: │  1  │  1  │  2  │  2  │  2  │  3  │                   │   │
│  │        ├─────┼─────┼─────┼─────┼─────┼─────┤                   │   │
│  │  Cmd:  │ x=1 │ y=2 │ x=3 │ z=1 │ y=4 │ x=5 │                   │   │
│  │        └─────┴─────┴─────┴─────┴─────┴─────┘                   │   │
│  │                                  ▲                              │   │
│  │                              commitIndex                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Replication Process:                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Leader              Follower 1        Follower 2               │   │
│  │    │                     │                 │                    │   │
│  │    │  AppendEntries      │                 │                    │   │
│  │    │  (term, prevLogIdx, │                 │                    │   │
│  │    │   prevLogTerm,      │                 │                    │   │
│  │    │   entries[])        │                 │                    │   │
│  │    │────────────────────►│                 │                    │   │
│  │    │────────────────────────────────────►  │                    │   │
│  │    │                     │                 │                    │   │
│  │    │◄── Success ─────────│                 │                    │   │
│  │    │◄── Success ─────────────────────────  │                    │   │
│  │    │                     │                 │                    │   │
│  │    │  Majority replicated                  │                    │   │
│  │    │  → Increment commitIndex              │                    │   │
│  │    │  → Apply to state machine             │                    │   │
│  │    │                     │                 │                    │   │
│  │    │  Next AppendEntries includes commitIndex                   │   │
│  │    │  → Followers also commit              │                    │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Log Consistency

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Raft: Log Consistency                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Log Matching Property:                                                │
│  - Same index, same term → same command                                │
│  - Same index, same term → all previous entries identical              │
│                                                                         │
│  On Inconsistency:                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Leader:   [1,1] [1,2] [2,3] [3,4] [3,5] [3,6]                  │   │
│  │  Follower: [1,1] [1,2] [2,3] [2,4] [2,5]                        │   │
│  │                              ▲                                  │   │
│  │                          Divergence point                       │   │
│  │                                                                  │   │
│  │  Resolution: Leader overwrites Follower log to match            │   │
│  │                                                                  │   │
│  │  1. AppendEntries(prevLogIdx=3, prevLogTerm=2) fails            │   │
│  │  2. Decrement nextIndex and retry                               │   │
│  │  3. Find match point (index=3)                                  │   │
│  │  4. Overwrite from index 4 with Leader's log                    │   │
│  │                                                                  │   │
│  │  Result:                                                         │   │
│  │  Follower: [1,1] [1,2] [2,3] [3,4] [3,5] [3,6]                  │   │
│  │            (identical to Leader)                                 │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Raft vs Paxos

| Property | Raft | Paxos |
|----------|------|-------|
| Understandability | High | Low |
| Leader Role | Clear (strong leader) | Flexible |
| Log Order | Sequential | Gaps possible |
| Implementation Complexity | Low | High |
| Paper Clarity | Detailed | Abstract |
| Use Cases | etcd, CockroachDB | Chubby, Spanner |

---

## 4. Byzantine Fault Tolerance

### Byzantine Generals Problem

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Byzantine Generals Problem                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Situation: N generals must decide to attack/retreat                   │
│             Some generals may be traitors                               │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │                    ┌──────────┐                                  │   │
│  │                    │ General 1│                                  │   │
│  │                    │ (traitor)│                                  │   │
│  │                    └────┬─────┘                                  │   │
│  │                         │                                        │   │
│  │           "Attack"      │       "Retreat"                        │   │
│  │              ▼                    ▼                              │   │
│  │       ┌──────────┐         ┌──────────┐                         │   │
│  │       │ General 2│         │ General 3│                         │   │
│  │       │ (loyal)  │         │ (loyal)  │                         │   │
│  │       └──────────┘         └──────────┘                         │   │
│  │                                                                  │   │
│  │  Gen 2: "#1 said attack" / Gen 3: "#1 said retreat"             │   │
│  │  → Loyal generals can make different decisions!                  │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Solution: With 3f + 1 generals, can tolerate f traitors               │
│  Example: 4 generals → tolerate 1 traitor                              │
│           7 generals → tolerate 2 traitors                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### PBFT (Practical Byzantine Fault Tolerance)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     PBFT Algorithm                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  3-Phase Protocol: Pre-Prepare → Prepare → Commit                      │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Client     Primary     Replica 1    Replica 2    Replica 3     │   │
│  │    │           │            │            │            │         │   │
│  │    │──Request──►│            │            │            │         │   │
│  │    │           │            │            │            │         │   │
│  │    │           │─Pre-Prepare─►│           │            │         │   │
│  │    │           │─Pre-Prepare──────────────►│           │         │   │
│  │    │           │─Pre-Prepare──────────────────────────►│         │   │
│  │    │           │            │            │            │         │   │
│  │    │           │◄──Prepare──│            │            │         │   │
│  │    │           │────Prepare─►│◄─Prepare──│            │         │   │
│  │    │           │            │────Prepare──►◄─Prepare──│         │   │
│  │    │           │            │            │────Prepare─►│         │   │
│  │    │           │            │            │            │         │   │
│  │    │           │◄───Commit──│            │            │         │   │
│  │    │           │────Commit──►│◄──Commit──│            │         │   │
│  │    │           │            │────Commit───►◄──Commit──│         │   │
│  │    │           │            │            │────Commit──►│         │   │
│  │    │           │            │            │            │         │   │
│  │    │◄──Reply───│            │            │            │         │   │
│  │    │◄──Reply───────────────│            │            │         │   │
│  │    │◄──Reply──────────────────────────── │            │         │   │
│  │    │◄──Reply────────────────────────────────────────  │         │   │
│  │    │           │            │            │            │         │   │
│  │                                                                  │   │
│  │  Client accepts result when receiving f+1 identical Replies     │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Complexity: O(n²) messages                                            │
│  Limitation: Low scalability (typically ~20 nodes)                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### BFT Applications

| Domain | Examples |
|--------|----------|
| Blockchain | Tendermint, Hyperledger Fabric |
| Aviation/Space | Flight control systems |
| Finance | High-availability trading systems |

---

## 5. Practical Applications

### ZooKeeper (ZAB)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ZooKeeper / ZAB                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ZAB (ZooKeeper Atomic Broadcast):                                     │
│  - Paxos variant                                                       │
│  - Leader-based                                                        │
│  - Order guaranteed                                                    │
│                                                                         │
│  Data Model:                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  /                        (root)                                │   │
│  │  ├── /config             Configuration data                     │   │
│  │  │   └── /config/db      {"host": "localhost"}                  │   │
│  │  ├── /locks              Distributed locks                      │   │
│  │  │   └── /locks/job1     (ephemeral)                            │   │
│  │  └── /leader             Leader election                        │   │
│  │      └── /leader/node-1  (ephemeral + sequential)               │   │
│  │                                                                  │   │
│  │  Node Types:                                                     │   │
│  │  - Persistent: Remains until explicitly deleted                 │   │
│  │  - Ephemeral: Auto-deleted on session end                       │   │
│  │  - Sequential: Auto-incrementing number assigned                │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Usage:                                                                │
│  - Kafka: Broker coordination, topic metadata                          │
│  - HBase: Master election, region assignment                           │
│  - Hadoop: HA NameNode                                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### etcd (Raft)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     etcd                                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Features:                                                             │
│  - Raft consensus algorithm                                            │
│  - Key-value store                                                     │
│  - gRPC API                                                            │
│  - Watch capability                                                    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Kubernetes Control Plane                                        │   │
│  │                                                                  │   │
│  │  ┌─────────────┐         ┌─────────────────────────┐            │   │
│  │  │ API Server  │────────►│        etcd Cluster     │            │   │
│  │  │             │         │                         │            │   │
│  │  │             │         │  ┌─────┐ ┌─────┐ ┌─────┐│            │   │
│  │  │ - Pods      │         │  │etcd1│ │etcd2│ │etcd3││            │   │
│  │  │ - Services  │         │  │Raft │ │Raft │ │Raft ││            │   │
│  │  │ - ConfigMaps│         │  └─────┘ └─────┘ └─────┘│            │   │
│  │  └─────────────┘         └─────────────────────────┘            │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Usage Examples:                                                       │
│  ```                                                                   │
│  # Store key                                                           │
│  etcdctl put /config/db/host "localhost"                              │
│                                                                         │
│  # Get key                                                             │
│  etcdctl get /config/db/host                                          │
│                                                                         │
│  # Watch (detect changes)                                              │
│  etcdctl watch /config/db/                                            │
│                                                                         │
│  # Check leader                                                        │
│  etcdctl endpoint status                                               │
│  ```                                                                   │
│                                                                         │
│  Applications:                                                         │
│  - Kubernetes: Cluster state storage                                   │
│  - Service Discovery: Consul alternative                               │
│  - Feature Flags: Dynamic configuration                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Consul

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Consul                                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Features:                                                             │
│  - Raft consensus algorithm                                            │
│  - Service discovery                                                   │
│  - Health checking                                                     │
│  - KV store                                                            │
│  - Multi-datacenter                                                    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │   DC1 (Primary)                   DC2 (Secondary)                │   │
│  │   ┌─────────────────┐            ┌─────────────────┐            │   │
│  │   │  Consul Servers │◄──WAN────►│  Consul Servers │            │   │
│  │   │  (3 nodes, Raft)│  Gossip    │  (3 nodes, Raft)│            │   │
│  │   └────────┬────────┘            └────────┬────────┘            │   │
│  │            │                              │                      │   │
│  │     ┌──────┴──────┐                ┌──────┴──────┐              │   │
│  │     │             │                │             │              │   │
│  │  ┌──┴──┐       ┌──┴──┐          ┌──┴──┐       ┌──┴──┐          │   │
│  │  │Agent│       │Agent│          │Agent│       │Agent│          │   │
│  │  │Svc A│       │Svc B│          │Svc A│       │Svc C│          │   │
│  │  └─────┘       └─────┘          └─────┘       └─────┘          │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Key Features:                                                         │
│  - DNS-based service discovery: service.consul                         │
│  - Distributed locks: Consul Sessions                                  │
│  - Leader election: consul lock                                        │
│  - Configuration management: Consul KV + Template                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tool Comparison

| Property | ZooKeeper | etcd | Consul |
|----------|-----------|------|--------|
| Consensus Algorithm | ZAB (Paxos variant) | Raft | Raft |
| Data Model | Hierarchical (filesystem) | Flat KV | Flat KV |
| Watch | O | O | O |
| Service Discovery | Manual implementation | Manual implementation | Built-in |
| Health Check | X | X | Built-in |
| Multi-DC | X | X | O |
| Language | Java | Go | Go |
| Primary Use Case | Hadoop, Kafka | Kubernetes | HashiCorp ecosystem |

---

## 6. Practice Problems

### Exercise 1: Paxos Scenario

Analyze the following scenario in a Paxos system with 5 Acceptors:
- Proposer 1 proposes "A" with n=1
- Proposer 2 proposes "B" with n=2
- Messages get reordered due to network delay
- What value gets selected in the end?

### Exercise 2: Raft Log Recovery

Explain the log consistency recovery process in Raft for the following situation:
```
Leader:   [1,1] [2,2] [2,3] [3,4] [3,5]
Follower: [1,1] [2,2] [2,3] [2,4]
```

### Exercise 3: Consensus System Design

Design a distributed configuration management system with the following requirements:
- Deployed across 3 datacenters
- Continue service even with one DC failure
- Configuration changes propagate within milliseconds
- Optimize read performance

---

**Previous**: [Distributed Systems Concepts](./15_Distributed_Systems_Concepts.md) | **Next**: [Practical Design Examples 1](./17_Design_Example_1.md)

---

## References

- "Paxos Made Simple" - Leslie Lamport
- "In Search of an Understandable Consensus Algorithm (Raft)" - Diego Ongaro
- "Practical Byzantine Fault Tolerance" - Castro, Liskov
- etcd Documentation: etcd.io
- ZooKeeper Documentation: zookeeper.apache.org
- Consul Documentation: consul.io
- Raft Visualization: raft.github.io

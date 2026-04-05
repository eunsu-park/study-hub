# Lesson 15: Formal Verification with TLA+

[Overview](./00_Overview.md) | [Previous: Distributed Coordination Primitives](./14_Distributed_Coordination_Primitives.md) | [Next: Capstone — Building a Distributed KV Store](./16_Capstone_Building_Distributed_KV_Store.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why formal verification is necessary beyond testing for distributed systems correctness
2. Write TLA+ specifications using variables, operators, temporal logic, and the prime notation
3. Specify safety (invariants) and liveness (temporal) properties for concurrent protocols
4. Use PlusCal to translate pseudocode-level algorithms into verifiable TLA+ specifications
5. Apply the TLC model checker to exhaustively verify protocol properties and interpret counterexamples

---

## Table of Contents

1. [Why Formal Verification](#1-why-formal-verification)
2. [TLA+ Overview](#2-tla-overview)
3. [TLA+ Language Basics](#3-tla-language-basics)
4. [Writing a Simple Spec: Mutual Exclusion](#4-writing-a-simple-spec-mutual-exclusion)
5. [PlusCal: Algorithmic Language for TLA+](#5-pluscal-algorithmic-language-for-tla)
6. [TLC Model Checker](#6-tlc-model-checker)
7. [Case Study: Simplified Raft Leader Election in TLA+](#7-case-study-simplified-raft-leader-election-in-tla)
8. [Modeling Consensus Protocols in TLA+](#8-modeling-consensus-protocols-in-tla)
9. [How AWS Uses TLA+](#9-how-aws-uses-tla)
10. [Practical Verification Workflow](#10-practical-verification-workflow)
11. [Other Formal Methods](#11-other-formal-methods)
12. [Implementation: TLA+ Spec Validator and PlusCal Translator](#12-implementation-tla-spec-validator-and-pluscal-translator)
13. [Summary and Further Reading](#13-summary-and-further-reading)

---

## 1. Why Formal Verification

### 1.1 The Limits of Testing

Testing distributed systems is fundamentally limited:

```
Testing:     Explores a tiny fraction of the state space
             ┌─────────────────────────────────┐
             │         All possible states      │
             │    ┌─────────────┐               │
             │    │             │               │
             │    │  Tested     │  Untested     │
             │    │  states     │  states       │
             │    │  (tiny)     │  (vast)       │
             │    └─────────────┘               │
             └─────────────────────────────────┘

Verification: Exhaustively checks ALL reachable states
             ┌─────────────────────────────────┐
             │         All possible states      │
             │  ┌───────────────────────────┐   │
             │  │                           │   │
             │  │  Verified                 │   │ Unreachable
             │  │  (all reachable states)   │   │ states
             │  │                           │   │
             │  └───────────────────────────┘   │
             └─────────────────────────────────┘
```

**Why testing misses bugs in distributed systems**:

| Bug Type | Why Testing Misses It | Example |
|----------|----------------------|---------|
| Race conditions | Require specific interleaving | Two leaders elected in same term |
| Failover edge cases | Require specific failure timing | Data loss during leader change |
| Liveness violations | May only manifest over long runs | A process starves forever |
| Multi-node interactions | Combinatorial explosion of orderings | 3 nodes, 5 messages = 120+ orderings |

### 1.2 What Formal Verification Provides

Formal verification proves properties about **all possible behaviors** of a system:

- **Safety**: "Nothing bad ever happens" (e.g., at most one leader per term)
- **Liveness**: "Something good eventually happens" (e.g., a leader is eventually elected)

```
                Testing              Formal Verification
Finds bugs?       Yes                   Yes
Proves absence?   No                    Yes (within the model)
Scales to         Large systems         Small models (~10-20 nodes)
production code?  Yes                   No (verifies the design, not code)
Human effort      Write test cases      Write specification
Automation        High                  High (model checking)
```

### 1.3 Heisenbugs in Distributed Systems

The most dangerous bugs in distributed systems are **non-deterministic** failures that depend on precise timing and message ordering:

```
Timeline showing a subtle split-brain bug:

Node A (leader, term 1):     ──[append]──────[commit]──────►
                                    │                    ▲
                                    ▼                    │ (stale ack)
Network partition:           ═══════════════════════════════
                                    │                    │
Node B (becomes leader, term 2): ──[elect]──[append']──[commit']──►
                                              │
                                              ▼
Node C:                      ──────────────[accept B]──────────►

Bug: Node A commits entry at index 3 (term 1) while
     Node B commits a different entry at index 3 (term 2).
     Requires partition + exact timing to reproduce.
```

Such bugs share common traits:

| Property | Testing | Formal Verification |
|---|---|---|
| Coverage | Samples paths randomly | Explores **all** reachable states |
| Reproducibility | Hard — timing-dependent | Deterministic counterexamples |
| Confidence | Probabilistic | Mathematical proof (within the model) |
| Scalability | Linear in test count | Exponential in state variables |
| Effort | Low initial, high maintenance | High initial, low maintenance |

### 1.4 Real-World Failures That Escaped Testing

**Amazon DynamoDB (2015):** A metadata corruption bug survived 6 months of testing
and was only found after a formal TLA+ specification revealed an edge case where
a coordinator could finalize a reconfiguration while a storage node still held a
stale membership view.

**Apache ZooKeeper (CVE-2017-5637):** A bug in the leader election protocol
allowed two leaders to coexist under a specific sequence of four network events.
Random testing had never produced that exact sequence.

**CockroachDB (2019):** A transaction serialization anomaly required three
concurrent transactions and a specific conflict pattern. Found via Jepsen testing
after escaping all unit and integration tests.

### 1.5 Where Formal Verification Fits

Formal verification does not replace testing — it operates at a different level:

```
                   ┌─────────────────────────────┐
                   │     Specification (TLA+)     │  ← Formal verification
                   │   "The algorithm is correct" │     verifies THIS level
                   └──────────────┬──────────────┘
                                  │ Refinement gap
                   ┌──────────────▼──────────────┐
                   │   Implementation (Code)      │  ← Testing verifies
                   │   "The code matches the spec"│     THIS level
                   └──────────────┬──────────────┘
                                  │ Environment gap
                   ┌──────────────▼──────────────┐
                   │   Deployment (Production)    │  ← Chaos engineering
                   │   "It works under real faults│     verifies THIS level
                   └─────────────────────────────┘
```

The verification stack:
- **Model checking (TLA+/TLC):** Proves the algorithm correct
- **Unit/integration tests:** Validates that code implements the algorithm
- **Jepsen/chaos testing:** Validates behavior under real failure modes

```
Design Phase         ──▶  Write TLA+ specification
                          ┌──────────────────────┐
                          │ TLC model checker     │
                          │ finds design bugs     │
                          │ BEFORE any code       │
                          └──────┬───────────────┘
                                 │
Implementation Phase ──▶  Write code based on verified spec
                          ┌──────────────────────┐
                          │ Unit tests, property  │
                          │ tests verify code     │
                          │ matches spec          │
                          └──────┬───────────────┘
                                 │
Production Phase     ──▶  Monitor invariants at runtime
                          ┌──────────────────────┐
                          │ Jepsen, chaos testing │
                          │ verify deployment     │
                          └──────────────────────┘
```

---

## 2. TLA+ Overview

### 2.1 What Is TLA+?

TLA+ (Temporal Logic of Actions) is a formal specification language created by Leslie Lamport. It describes systems as mathematical objects — states, transitions, and properties — rather than as executable programs.

**Key insight**: TLA+ does not describe *how* a system works (implementation), but *what* the system does (behavior). This level of abstraction is precisely what makes it powerful for finding design bugs.

### 2.2 Core Concepts

**State**: An assignment of values to all variables. A state is a snapshot of the system at one point in time.

```
Example state of a mutual exclusion system:
  pc = [p1 ↦ "waiting", p2 ↦ "critical"]
  lock = "p2"
```

**Action**: A relation between a state and its successor. An action describes one step of the system.

```
Example action: Process p1 enters the critical section
  Precondition: pc[p1] = "waiting" ∧ lock = "free"
  Effect: pc'[p1] = "critical" ∧ lock' = "p1"
  (primed variables refer to the next state)
```

**Behavior**: An infinite sequence of states. A behavior represents one possible execution of the system.

```
s0 → s1 → s2 → s3 → s4 → ...

Example behavior:
  [pc = [p1↦"idle", p2↦"idle"], lock = "free"]
  → [pc = [p1↦"waiting", p2↦"idle"], lock = "free"]
  → [pc = [p1↦"critical", p2↦"idle"], lock = "p1"]
  → [pc = [p1↦"idle", p2↦"idle"], lock = "free"]
  → ...
```

**Specification**: A formula that describes the set of all valid behaviors:

```
Spec ≡ Init ∧ □[Next]_vars

Where:
  Init    = initial state predicate
  Next    = disjunction of all possible actions
  □       = "always" (in every state of the behavior)
  [A]_v   = A ∨ (v' = v)  (action A or stuttering)
  vars    = tuple of all variables
```

### 2.3 Why Stuttering?

The `[Next]_vars` notation allows **stuttering steps** — steps where nothing changes. This is essential for composability: when you combine two specifications, the actions of one system look like stuttering to the other.

```
System A: s0 ─a1─▶ s1 ─a2─▶ s2 ─a3─▶ s3
System B: t0 ─────────────── b1 ──▶ t1 ────── b2 ──▶ t2

Combined: (s0,t0) ─a1─▶ (s1,t0) ─a2─▶ (s2,t0) ─b1─▶ (s2,t1) ─a3─▶ (s3,t1) ─b2─▶ (s3,t2)

B sees stuttering during a1, a2, a3
A sees stuttering during b1, b2
```

---

## 3. TLA+ Language Basics

### 3.1 Variables and Constants

```tla
---- MODULE SimpleKV ----
EXTENDS Integers, Sequences, FiniteSets

CONSTANTS Keys, Values, Nodes   \* Model parameters (set at check time)
VARIABLES store, pending        \* State variables (change during execution)

vars == <<store, pending>>      \* Tuple of all variables (for stuttering)
====
```

- `CONSTANTS`: Parameters fixed for a given model check. Example: `Keys = {"k1", "k2"}`.
- `VARIABLES`: The mutable state. Each state assigns values to all variables.
- `vars`: A convenience tuple grouping all variables.

### 3.2 Data Structures

TLA+ has built-in support for sets, functions, records, sequences, and tuples:

```tla
\* Sets
S == {1, 2, 3}
T == {x \in S : x > 1}           \* Filter: {2, 3}
U == {x * 2 : x \in S}           \* Map: {2, 4, 6}

\* Functions (total mappings from domain to range)
f == [x \in {"a", "b"} |-> 0]    \* f["a"] = 0, f["b"] = 0
g == [f EXCEPT !["a"] = 1]       \* g["a"] = 1, g["b"] = 0 (functional update)

\* Records (functions with string domains)
r == [name |-> "Alice", age |-> 30]
r.name                              \* "Alice"

\* Sequences (functions with domain 1..n)
seq == <<1, 2, 3>>
Len(seq)                            \* 3
Append(seq, 4)                      \* <<1, 2, 3, 4>>
Head(seq)                           \* 1
Tail(seq)                           \* <<2, 3>>

\* Tuples (just sequences used as pairs/triples)
pair == <<x, y>>
```

### 3.3 Actions and Priming

An **action** is a boolean formula involving primed and unprimed variables:
- Unprimed `x` refers to the current state
- Primed `x'` refers to the next state

```tla
\* Action: increment counter
Increment ==
    /\ counter < MAX          \* Precondition (current state)
    /\ counter' = counter + 1 \* Effect (next state)
    /\ UNCHANGED other_var    \* other_var does not change

\* UNCHANGED x is shorthand for x' = x
```

### 3.4 ENABLED and Fairness

```tla
\* ENABLED A is TRUE if action A can be taken in the current state
\* (i.e., there exists a next state satisfying A)

\* Weak fairness: if A is continuously enabled, it is eventually taken
WF_vars(A) == □(□ENABLED <A>_vars => ◇<A>_vars)

\* Strong fairness: if A is repeatedly enabled, it is eventually taken
SF_vars(A) == □(□◇ENABLED <A>_vars => ◇<A>_vars)
```

**When to use which**:
- Weak fairness (WF): the action must be *continuously* enabled. Use for actions that once enabled, stay enabled (e.g., "process can always make progress").
- Strong fairness (SF): the action is *repeatedly* enabled but might be disabled in between. Use for actions that compete with others (e.g., lock acquisition).

### 3.5 Temporal Operators

| Operator | TLA+ Syntax | Meaning |
|----------|-------------|---------|
| Always | `[]P` or `□P` | P is true in every state of every behavior |
| Eventually | `<>P` or `◇P` | P is true in at least one state of every behavior |
| Leads-to | `P ~> Q` | Whenever P becomes true, Q eventually becomes true |
| Always-eventually | `[]<>P` | P is true infinitely often |
| Eventually-always | `<>[]P` | P eventually becomes and remains true |

**Important combinations**:

```
Safety:    □(¬bad_state)          "Bad things never happen"
Liveness:  ◇(good_state)          "Good things eventually happen"
Progress:  request ~> response     "Every request gets a response"
Fairness:  □◇(enabled => taken)   "Enabled actions eventually execute"
```

---

## 4. Writing a Simple Spec: Mutual Exclusion

### 4.1 The Problem

Two processes want to enter a critical section. At most one should be in the critical section at any time.

### 4.2 TLA+ Specification

```tla
---- MODULE MutualExclusion ----
EXTENDS Integers

\* Two processes
CONSTANTS Procs
ASSUME Procs = {0, 1}

VARIABLES pc, turn, flag

vars == <<pc, turn, flag>>

\* Process states: "idle", "want", "wait", "critical"

Init ==
    /\ pc = [p \in Procs |-> "idle"]
    /\ turn = 0
    /\ flag = [p \in Procs |-> FALSE]

\* ----- Actions -----

\* Process p wants to enter critical section
Want(p) ==
    /\ pc[p] = "idle"
    /\ pc' = [pc EXCEPT ![p] = "want"]
    /\ flag' = [flag EXCEPT ![p] = TRUE]
    /\ UNCHANGED turn

\* Process p sets turn and waits (Peterson's algorithm)
Wait(p) ==
    /\ pc[p] = "want"
    /\ turn' = 1 - p                    \* Give priority to other process
    /\ pc' = [pc EXCEPT ![p] = "wait"]
    /\ UNCHANGED flag

\* Process p enters critical section
Enter(p) ==
    /\ pc[p] = "wait"
    /\ \/ flag[1-p] = FALSE             \* Other process not interested
       \/ turn = p                       \* It's our turn
    /\ pc' = [pc EXCEPT ![p] = "critical"]
    /\ UNCHANGED <<turn, flag>>

\* Process p exits critical section
Exit(p) ==
    /\ pc[p] = "critical"
    /\ flag' = [flag EXCEPT ![p] = FALSE]
    /\ pc' = [pc EXCEPT ![p] = "idle"]
    /\ UNCHANGED turn

\* ----- Specification -----

Next ==
    \E p \in Procs :
        \/ Want(p)
        \/ Wait(p)
        \/ Enter(p)
        \/ Exit(p)

Spec == Init /\ [][Next]_vars
        /\ WF_vars(Want(0)) /\ WF_vars(Wait(0))
        /\ WF_vars(Enter(0)) /\ WF_vars(Exit(0))
        /\ WF_vars(Want(1)) /\ WF_vars(Wait(1))
        /\ WF_vars(Enter(1)) /\ WF_vars(Exit(1))

\* ----- Properties to check -----

\* Safety: mutual exclusion
MutualExclusion ==
    ~(pc[0] = "critical" /\ pc[1] = "critical")

\* Liveness: no starvation (if a process wants the CS, it eventually gets it)
NoStarvation ==
    /\ (pc[0] = "want" ~> pc[0] = "critical")
    /\ (pc[1] = "want" ~> pc[1] = "critical")

\* Type invariant (helps TLC check faster)
TypeOK ==
    /\ pc \in [Procs -> {"idle", "want", "wait", "critical"}]
    /\ turn \in Procs
    /\ flag \in [Procs -> BOOLEAN]

====
```

### 4.3 State Space Visualization

```
The TLC model checker explores all reachable states:

Initial state:
  pc=[0↦idle, 1↦idle], turn=0, flag=[0↦F, 1↦F]

From here, two actions are possible:
  Want(0) or Want(1)

Want(0) leads to:
  pc=[0↦want, 1↦idle], turn=0, flag=[0↦T, 1↦F]

  From here: Wait(0) or Want(1)
  ... and so on

State graph (partial):

  (idle,idle) ──Want(0)──▶ (want,idle)
       │                       │
   Want(1)                 Wait(0)
       │                       │
       ▼                       ▼
  (idle,want) ──────▶    (wait,idle) ──Enter(0)──▶ (critical,idle)
                              │                         │
                          Want(1)                    Exit(0)
                              │                         │
                              ▼                         ▼
                        (wait,want) ──────▶      (idle,idle)

TLC explores ALL paths through this graph, checking
MutualExclusion at every state.
```

### 4.4 What TLC Checks

For the mutual exclusion spec above:

| Property | Type | What TLC Does |
|----------|------|---------------|
| TypeOK | Invariant | Check in every reachable state |
| MutualExclusion | Invariant | Check in every reachable state |
| NoStarvation | Temporal (liveness) | Check on every possible infinite behavior |

If any property is violated, TLC outputs a **counterexample**: a specific sequence of states that leads to the violation.

---

## 5. PlusCal: Algorithmic Language for TLA+

### 5.1 Why PlusCal?

TLA+ is a mathematical logic — powerful but unfamiliar to programmers. PlusCal provides a pseudocode-like syntax that is automatically translated to TLA+ by the `pcal` translator.

```
PlusCal code                    Translated TLA+
(readable pseudocode)    ────▶  (mathematical spec)
                                     │
                                     ▼
                                TLC model checker
```

### 5.2 PlusCal Syntax

```tla
---- MODULE PetersonPlusCal ----
EXTENDS Integers

(*--algorithm Peterson
variables
    flag = [i \in {0, 1} |-> FALSE],
    turn = 0;

process proc \in {0, 1}
begin
    P1: \* Label: want to enter
        flag[self] := TRUE;
    P2: \* Give priority to other
        turn := 1 - self;
    P3: \* Wait until safe to enter
        await flag[1 - self] = FALSE \/ turn = self;
    CS: \* Critical section
        skip; \* (do critical work here)
    P4: \* Exit critical section
        flag[self] := FALSE;
    P5: \* Return to beginning
        goto P1;
end process;

end algorithm; *)
====
```

### 5.3 PlusCal Key Features

**Labels**: Define the granularity of atomicity. Everything between two labels executes atomically. This is crucial for modeling concurrency correctly.

```
\* WRONG: too much atomicity
begin
    FullyCritical:   \* Everything is one atomic step — no concurrency!
        flag[self] := TRUE;
        turn := 1 - self;
        await ...;
        skip;
        flag[self] := FALSE;

\* RIGHT: each step is a separate atomic action
begin
    SetFlag:    flag[self] := TRUE;
    SetTurn:    turn := 1 - self;
    Wait:       await flag[1-self] = FALSE \/ turn = self;
    CritSec:    skip;
    ClearFlag:  flag[self] := FALSE;
```

**await**: Blocks until the condition becomes true. Unlike busy-waiting, it tells TLC to only explore states where the condition holds.

**with**: Non-deterministic choice.

```
\* Choose a random element from a set
with node \in Nodes do
    Send(node, message);
end with;
```

**either/or**: Non-deterministic branching.

```
either
    \* Path 1: message is delivered
    Deliver(msg);
or
    \* Path 2: message is lost
    skip;
or
    \* Path 3: message is delayed
    Enqueue(msg, delayed_queue);
end either;
```

### 5.4 Modeling Non-Determinism

Non-determinism is essential for modeling distributed systems because:
1. Message delivery order is non-deterministic
2. Process scheduling is non-deterministic
3. Failures can occur at any time

PlusCal models all possible non-deterministic choices, and TLC explores all of them:

```tla
(*--algorithm NonDetExample
variables msgs = {};

process sender = "S"
begin
    Send:
        msgs := msgs \union {<<"hello">>, <<"world">>};
end process;

process receiver = "R"
begin
    Recv:
        either
            \* Receive in order
            with m \in msgs do
                msgs := msgs \ {m};
            end with;
        or
            \* Network failure: lose a message
            with m \in msgs do
                msgs := msgs \ {m};
            end with;
        end either;
end process;

end algorithm; *)
```

---

## 6. TLC Model Checker

### 6.1 How TLC Works

TLC performs **explicit state model checking**: it constructs the complete state graph by exploring every reachable state and every possible transition.

```
Algorithm:
  1. Compute all initial states satisfying Init
  2. For each unexplored state:
     a. Check all invariants
     b. Compute all successor states via Next
     c. Add new states to the exploration queue
  3. For liveness: check temporal properties on the state graph
     (find cycles that violate liveness using Tarjan's algorithm)

BFS exploration ensures shortest counterexample.
```

### 6.2 State Space Size

The number of states grows exponentially with the number of variables and their domains:

```
Example: Mutual exclusion with 2 processes
  pc: 4 states each = 4² = 16
  turn: 2 states = 2
  flag: 2 states each = 2² = 4
  Total: 16 × 2 × 4 = 128 states (manageable)

Example: Raft with 3 nodes, 2 terms, log size 2
  state: 3 values × 3 nodes = 27
  currentTerm: 2 values × 3 nodes = 8
  votedFor: 4 values × 3 nodes = 64
  log: (entries)³ = hundreds
  Total: millions of states (still manageable for TLC)

Example: Raft with 5 nodes, 3 terms, log size 5
  Total: billions of states (may take hours/days)
```

### 6.3 Configuring TLC

TLC is configured via a `.cfg` file or the TLA+ Toolbox:

```
\* ModelChecking.cfg

SPECIFICATION Spec

\* Set concrete values for CONSTANTS
CONSTANTS
    Procs = {0, 1}

\* Check these invariants at every state
INVARIANT TypeOK
INVARIANT MutualExclusion

\* Check these temporal properties
PROPERTY NoStarvation

\* Constraint: limit state space (optional)
CONSTRAINT
    turn \in 0..1

\* Symmetry: {0, 1} are interchangeable (reduces state space by 2x)
SYMMETRY Permutations(Procs)
```

### 6.4 Symmetry Sets

Symmetry reduction can dramatically reduce the state space. If a set of model values is symmetric (interchangeable), TLC only needs to explore one representative ordering.

```
Without symmetry: nodes = {n1, n2, n3}
  State (n1=leader, n2=follower, n3=follower)
  State (n2=leader, n1=follower, n3=follower)
  State (n3=leader, n1=follower, n2=follower)
  → 3 distinct states

With symmetry: nodes is a symmetry set
  All three states are equivalent (one representative explored)
  → 1 state checked instead of 3

For n nodes: up to n! reduction factor
```

### 6.5 Interpreting TLC Output

```
TLC output for a successful check:

  Model checking completed. No error has been found.
    Estimates of the probability that TLC did not check all
    reachable states because two distinct states had the
    same fingerprint:
      calculated: 1.2E-15

    2847 states generated, 1423 distinct states found, 0 states left on queue.
    The depth of the complete state graph search is 23.

TLC output for an invariant violation:

  Error: Invariant MutualExclusion is violated.

  Error: The following behavior constitutes a counterexample:

  State 1: <Initial predicate>
    /\ pc = [0 |-> "idle", 1 |-> "idle"]
    /\ flag = [0 |-> FALSE, 1 |-> FALSE]
    /\ turn = 0

  State 2: <Want(0)>
    /\ pc = [0 |-> "want", 1 |-> "idle"]
    /\ flag = [0 |-> TRUE, 1 |-> FALSE]
    /\ turn = 0

  State 3: ...
  (sequence continues to the violating state)
```

### 6.6 Practical Tips for TLC

| Tip | Rationale |
|-----|-----------|
| Start with small constants | Debug spec logic before scaling up |
| Add a TypeOK invariant | Catches many bugs early; also speeds up TLC |
| Use symmetry sets | Can reduce state space by n! |
| Add state constraints | Bound unbounded variables (e.g., log length) |
| Check invariants before liveness | Invariant checking is much faster |
| Use multiple workers | TLC supports multi-threaded exploration |
| Profile with `-coverage` | Shows which actions are taken and how often |

---

## 7. Case Study: Simplified Raft Leader Election in TLA+

### 7.1 What We Specify

A simplified version of Raft leader election with:
- N servers, each in state {Follower, Candidate, Leader}
- Terms (logical clocks for elections)
- Vote requests and grants
- Safety property: at most one leader per term

We omit log replication to focus on the election mechanism.

### 7.2 Full TLA+ Specification

```tla
---- MODULE RaftLeaderElection ----
\*
\* Simplified Raft Leader Election
\*
\* Specifies the election protocol from the Raft paper.
\* Safety property: at most one leader per term.
\* Liveness property: a leader is eventually elected (with fairness).
\*

EXTENDS Integers, FiniteSets

CONSTANTS
    Server,       \* Set of server IDs, e.g., {s1, s2, s3}
    MaxTerm       \* Maximum term number (bounds the state space)

VARIABLES
    currentTerm,  \* currentTerm[s]: latest term server s has seen
    votedFor,     \* votedFor[s]: candidate that server s voted for in current term (or Nil)
    state,        \* state[s]: Follower, Candidate, or Leader
    votesGranted  \* votesGranted[s]: set of servers that granted vote to s

vars == <<currentTerm, votedFor, state, votesGranted>>

Nil == "nil"

\* ----- Helper Definitions -----

\* A quorum is a majority of servers
Quorum == {Q \in SUBSET Server : Cardinality(Q) * 2 > Cardinality(Server)}

\* ----- Initial State -----

Init ==
    /\ currentTerm = [s \in Server |-> 0]
    /\ votedFor    = [s \in Server |-> Nil]
    /\ state       = [s \in Server |-> "Follower"]
    /\ votesGranted = [s \in Server |-> {}]

\* ----- Actions -----

\* Server s times out and starts an election
Timeout(s) ==
    /\ state[s] \in {"Follower", "Candidate"}
    /\ currentTerm[s] < MaxTerm        \* Bound for model checking
    /\ currentTerm' = [currentTerm EXCEPT ![s] = currentTerm[s] + 1]
    /\ votedFor'    = [votedFor EXCEPT ![s] = s]   \* Vote for self
    /\ state'       = [state EXCEPT ![s] = "Candidate"]
    /\ votesGranted' = [votesGranted EXCEPT ![s] = {s}]  \* Self-vote

\* Server s (candidate) requests vote from server t
\* Server t grants vote if:
\*   1. t's term <= s's term
\*   2. t hasn't voted in s's term (or already voted for s)
RequestVote(s, t) ==
    /\ state[s] = "Candidate"
    /\ s # t
    /\ currentTerm[s] >= currentTerm[t]
    /\ \/ votedFor[t] = Nil
       \/ votedFor[t] = s
       \/ currentTerm[s] > currentTerm[t]
    \* Update t's state
    /\ currentTerm' = [currentTerm EXCEPT ![t] = currentTerm[s]]
    /\ votedFor'    = [votedFor EXCEPT ![t] = s]
    /\ votesGranted' = [votesGranted EXCEPT ![s] = votesGranted[s] \union {t}]
    /\ state'       = [state EXCEPT ![t] = IF currentTerm[s] > currentTerm[t]
                                            THEN "Follower"
                                            ELSE state[t]]

\* Server s becomes leader (has received votes from a quorum)
BecomeLeader(s) ==
    /\ state[s] = "Candidate"
    /\ votesGranted[s] \in Quorum
    /\ state' = [state EXCEPT ![s] = "Leader"]
    /\ UNCHANGED <<currentTerm, votedFor, votesGranted>>

\* Server s discovers a higher term and steps down
StepDown(s, t) ==
    /\ currentTerm[t] > currentTerm[s]
    /\ state[s] \in {"Candidate", "Leader"}
    /\ currentTerm' = [currentTerm EXCEPT ![s] = currentTerm[t]]
    /\ state'       = [state EXCEPT ![s] = "Follower"]
    /\ votedFor'    = [votedFor EXCEPT ![s] = Nil]
    /\ UNCHANGED votesGranted

\* ----- Next State Relation -----

Next ==
    \/ \E s \in Server : Timeout(s)
    \/ \E s, t \in Server : RequestVote(s, t)
    \/ \E s \in Server : BecomeLeader(s)
    \/ \E s, t \in Server : StepDown(s, t)

\* ----- Specification -----

Spec == Init /\ [][Next]_vars
        /\ \A s \in Server : WF_vars(BecomeLeader(s))

\* ----- Safety Properties -----

\* At most one leader per term
AtMostOneLeaderPerTerm ==
    \A s, t \in Server :
        (s # t /\ state[s] = "Leader" /\ state[t] = "Leader")
        => currentTerm[s] # currentTerm[t]

\* Type invariant
TypeOK ==
    /\ currentTerm \in [Server -> 0..MaxTerm]
    /\ votedFor \in [Server -> Server \union {Nil}]
    /\ state \in [Server -> {"Follower", "Candidate", "Leader"}]
    /\ votesGranted \in [Server -> SUBSET Server]

\* A server's votesGranted only contains servers that voted for it
VotesCorrect ==
    \A s \in Server :
        \A t \in votesGranted[s] :
            votedFor[t] = s \/ currentTerm[t] # currentTerm[s]

\* ----- Liveness Properties -----

\* A leader is eventually elected (requires fairness)
EventuallyLeader ==
    <>(\E s \in Server : state[s] = "Leader")

====
```

### 7.3 Specification Walkthrough

**State variables and their roles**:

```
currentTerm[s]:
  ┌─────┐ Timeout  ┌─────┐ Timeout  ┌─────┐
  │  0  │─────────▶│  1  │─────────▶│  2  │──▶ ...
  └─────┘          └─────┘          └─────┘
  Monotonically increasing. Never decreases.

votedFor[s]:
  In each term, a server votes for at most one candidate.
  Reset to Nil when entering a new term via StepDown.

state[s]:
  Follower ──Timeout──▶ Candidate ──BecomeLeader──▶ Leader
     ▲                      │                          │
     │                      │ StepDown                 │ StepDown
     └──────────────────────┘──────────────────────────┘

votesGranted[s]:
  Set of servers that voted for s in the current election.
  Reset when s starts a new election (Timeout).
```

**Key invariant: AtMostOneLeaderPerTerm**:

This is the most critical safety property of Raft. The proof sketch is:

```
1. A server becomes leader only after receiving votes from a quorum
2. Each server votes for at most one candidate per term (votedFor)
3. Any two quorums must overlap (majority property)
4. Therefore, two candidates in the same term cannot both get quorums
5. Therefore, at most one leader per term □
```

### 7.4 Running TLC on This Spec

```
Configuration:
  Server = {s1, s2, s3}   (3-node cluster)
  MaxTerm = 3              (bound term space)

TLC results:
  States generated: ~45,000
  Distinct states: ~12,000
  Time: ~2 seconds
  Properties checked: TypeOK ✓, AtMostOneLeaderPerTerm ✓, EventuallyLeader ✓

With Server = {s1, s2, s3, s4, s5} and MaxTerm = 2:
  States generated: ~2,000,000
  Distinct states: ~400,000
  Time: ~30 seconds
  Properties checked: All pass ✓
```

---

## 8. Modeling Consensus Protocols in TLA+

### 8.1 Two-Phase Commit in TLA+

Two-Phase Commit (2PC) is the simplest consensus-like protocol to model:

```tla
---- MODULE TwoPhaseCommit ----
EXTENDS Integers, FiniteSets

CONSTANTS Participants

VARIABLES
    tmState,        \* Transaction manager state: "init", "committed", "aborted"
    tmPrepared,     \* Set of participants that have responded "prepared"
    pmState         \* Function: participant -> "working", "prepared", "committed", "aborted"

vars == <<tmState, tmPrepared, pmState>>

Init ==
    /\ tmState = "init"
    /\ tmPrepared = {}
    /\ pmState = [p \in Participants |-> "working"]

\* --- Participant actions ---

\* Participant votes to prepare
Prepare(p) ==
    /\ pmState[p] = "working"
    /\ pmState' = [pmState EXCEPT ![p] = "prepared"]
    /\ UNCHANGED <<tmState, tmPrepared>>

\* Participant spontaneously aborts (simulates crash/timeout)
ParticipantAbort(p) ==
    /\ pmState[p] = "working"
    /\ pmState' = [pmState EXCEPT ![p] = "aborted"]
    /\ UNCHANGED <<tmState, tmPrepared>>

\* --- Transaction manager actions ---

\* TM receives a prepare vote
ReceivePrepare(p) ==
    /\ tmState = "init"
    /\ pmState[p] = "prepared"
    /\ tmPrepared' = tmPrepared \cup {p}
    /\ UNCHANGED <<tmState, pmState>>

\* TM decides to commit (all participants prepared)
TMCommit ==
    /\ tmState = "init"
    /\ tmPrepared = Participants
    /\ tmState' = "committed"
    /\ pmState' = [p \in Participants |-> "committed"]
    /\ UNCHANGED tmPrepared

\* TM decides to abort
TMAbort ==
    /\ tmState = "init"
    /\ tmState' = "aborted"
    /\ pmState' = [p \in Participants |->
                    IF pmState[p] = "working" THEN "aborted"
                    ELSE pmState[p]]
    /\ UNCHANGED tmPrepared

Next ==
    \/ \E p \in Participants : Prepare(p)
    \/ \E p \in Participants : ParticipantAbort(p)
    \/ \E p \in Participants : ReceivePrepare(p)
    \/ TMCommit
    \/ TMAbort

\* --- Properties ---

\* Safety: If any participant committed, no participant aborted (and vice versa)
Consistency ==
    \A p1, p2 \in Participants :
        ~ (pmState[p1] = "committed" /\ pmState[p2] = "aborted")

Spec == Init /\ [][Next]_vars

====
```

### 8.2 Single-Decree Paxos in TLA+

Modeling Paxos requires capturing the message-passing nature of the protocol:

```tla
---- MODULE Paxos ----
EXTENDS Integers, FiniteSets

CONSTANTS Acceptors, Values, Quorums, MaxBallot

ASSUME QuorumAssumption ==
    /\ \A Q \in Quorums : Q \subseteq Acceptors
    /\ \A Q1, Q2 \in Quorums : Q1 \cap Q2 /= {}

VARIABLES
    maxBal,      \* maxBal[a]: highest ballot acceptor a has seen
    maxVBal,     \* maxVBal[a]: ballot of highest accepted proposal
    maxVal,      \* maxVal[a]: value of highest accepted proposal
    msgs         \* Set of all messages sent

vars == <<maxBal, maxVBal, maxVal, msgs>>

Init ==
    /\ maxBal = [a \in Acceptors |-> -1]
    /\ maxVBal = [a \in Acceptors |-> -1]
    /\ maxVal = [a \in Acceptors |-> "none"]
    /\ msgs = {}

\* Phase 1a: Proposer sends Prepare(ballot)
Phase1a(b) ==
    /\ b <= MaxBallot
    /\ msgs' = msgs \cup {[type |-> "1a", bal |-> b]}
    /\ UNCHANGED <<maxBal, maxVBal, maxVal>>

\* Phase 1b: Acceptor responds with Promise
Phase1b(a) ==
    /\ \E m \in msgs :
        /\ m.type = "1a"
        /\ m.bal > maxBal[a]
        /\ maxBal' = [maxBal EXCEPT ![a] = m.bal]
        /\ msgs' = msgs \cup {[type |-> "1b",
                                bal |-> m.bal,
                                acc |-> a,
                                mbal |-> maxVBal[a],
                                mval |-> maxVal[a]]}
    /\ UNCHANGED <<maxVBal, maxVal>>

\* Phase 2a: Proposer sends Accept with chosen value
Phase2a(b, v) ==
    /\ \E Q \in Quorums :
        LET promises == {m \in msgs : m.type = "1b" /\ m.bal = b}
            promisers == {m.acc : m \in promises}
        IN
        /\ Q \subseteq promisers
        /\ \/ \A m \in promises : m.mbal = -1    \* No prior accepted value
              /\ v \in Values                      \* Free to choose any value
           \/ LET maxPromise ==                    \* Must use highest accepted
                    CHOOSE m \in promises :
                        \A m2 \in promises : m.mbal >= m2.mbal
              IN v = maxPromise.mval
    /\ msgs' = msgs \cup {[type |-> "2a", bal |-> b, val |-> v]}
    /\ UNCHANGED <<maxBal, maxVBal, maxVal>>

\* Phase 2b: Acceptor accepts proposal
Phase2b(a) ==
    /\ \E m \in msgs :
        /\ m.type = "2a"
        /\ m.bal >= maxBal[a]
        /\ maxBal' = [maxBal EXCEPT ![a] = m.bal]
        /\ maxVBal' = [maxVBal EXCEPT ![a] = m.bal]
        /\ maxVal' = [maxVal EXCEPT ![a] = m.val]
        /\ msgs' = msgs \cup {[type |-> "2b",
                                bal |-> m.bal,
                                acc |-> a,
                                val |-> m.val]}

Next ==
    \/ \E b \in 0..MaxBallot : Phase1a(b)
    \/ \E a \in Acceptors : Phase1b(a)
    \/ \E b \in 0..MaxBallot, v \in Values : Phase2a(b, v)
    \/ \E a \in Acceptors : Phase2b(a)

\* --- Safety: Agreement ---
\* If a value is chosen (accepted by a quorum), no other value is chosen
ChosenValues ==
    {v \in Values :
        \E Q \in Quorums :
            \A a \in Q : \E m \in msgs :
                /\ m.type = "2b"
                /\ m.acc = a
                /\ m.val = v}

Agreement == Cardinality(ChosenValues) <= 1

Spec == Init /\ [][Next]_vars

====
```

### 8.3 Raft Consensus Core in TLA+

The key challenge in modeling Raft is capturing **log replication** with **network non-determinism**:

```tla
---- MODULE RaftConsensus ----
\* Simplified Raft focusing on core safety properties
EXTENDS Integers, Sequences, FiniteSets

CONSTANTS Nodes, Values, MaxTerm, MaxLogLen

VARIABLES
    currentTerm,    \* [Nodes -> Nat]
    state,          \* [Nodes -> {"follower", "candidate", "leader"}]
    votedFor,       \* [Nodes -> Nodes \cup {"none"}]
    log,            \* [Nodes -> Seq([term: Nat, val: Values])]
    commitIndex,    \* [Nodes -> Nat]
    messages        \* Set of in-flight messages

vars == <<currentTerm, state, votedFor, log, commitIndex, messages>>

\* --- Helper operators ---

\* A quorum is a strict majority
IsQuorum(S) == Cardinality(S) * 2 > Cardinality(Nodes)

\* The last log term for a node
LastLogTerm(n) ==
    IF Len(log[n]) > 0 THEN log[n][Len(log[n])].term ELSE 0

\* Log n1 is at least as up-to-date as log n2
LogUpToDate(n1, n2) ==
    \/ LastLogTerm(n1) > LastLogTerm(n2)
    \/ /\ LastLogTerm(n1) = LastLogTerm(n2)
       /\ Len(log[n1]) >= Len(log[n2])

\* --- Key Safety Properties ---
\* Election Safety: at most one leader per term
ElectionSafety ==
    \A n1, n2 \in Nodes :
        /\ state[n1] = "leader"
        /\ state[n2] = "leader"
        /\ currentTerm[n1] = currentTerm[n2]
        => n1 = n2

\* State Machine Safety: if a node has applied entry at index i,
\* no other node applies a different entry at index i
StateMachineSafety ==
    \A n1, n2 \in Nodes :
        \A i \in 1..Min(commitIndex[n1], commitIndex[n2]) :
            log[n1][i] = log[n2][i]

====
```

### 8.4 Modeling Network Non-Determinism

A critical aspect of modeling distributed systems is capturing all the ways the network can misbehave:

```tla
\* --- Network Model ---

\* Messages can be delivered, duplicated, or dropped (but not corrupted)
\* This models an asynchronous network with reliable delivery guarantees

\* Deliver a message (non-deterministic choice)
DeliverMessage ==
    /\ \E m \in messages :
        /\ HandleMessage(m)           \* Process the message
        /\ messages' = messages \ {m}  \* Remove from network

\* Drop a message (models packet loss)
DropMessage ==
    /\ \E m \in messages :
        /\ messages' = messages \ {m}
        /\ UNCHANGED <<currentTerm, state, votedFor, log, commitIndex>>

\* Duplicate a message (already modeled: messages is a set,
\* but we could use a multiset/bag for duplicate tracking)

\* Network partition: temporarily prevent delivery between node groups
\* Modeled implicitly by allowing DropMessage on any message

\* Message reordering: automatic, since we pick from a SET (unordered)
```

State space sizes for typical specifications:

```
Specification                 | Nodes | States     | Time
──────────────────────────────┼───────┼────────────┼──────────
Two-Phase Commit              | 3     | 12,408     | <1 sec
Two-Phase Commit              | 5     | 524,288    | 3 sec
Paxos (single decree)         | 3     | 1.2M       | 8 sec
Paxos (single decree)         | 5     | 847M       | 4 hours
Raft Leader Election          | 3     | 58,904     | 2 sec
Raft (full, 2 log entries)    | 3     | 38M        | 15 min
Raft (full, 3 log entries)    | 5     | >10B       | days
```

---

## 9. How AWS Uses TLA+

### 9.1 Overview

Amazon Web Services has used TLA+ since 2011 to verify the correctness of critical infrastructure components. The practice was introduced by Chris Newcombe and documented in a 2014 paper.

### 9.2 Systems Verified with TLA+

| AWS Service | Component | Bugs Found | Impact |
|-------------|-----------|------------|--------|
| **S3** | Object storage fault tolerance | Yes | Prevented data loss bug |
| **DynamoDB** | Replication and failover | Yes | Found subtle race in failover |
| **EBS** | Block storage replication | Yes | Found data corruption path |
| **Internal lock manager** | Distributed lock protocol | Yes | Multiple edge cases |

### 9.3 Case Study: DynamoDB Replication Bug

From the Newcombe et al. paper:

```
The Bug:
  - DynamoDB replication protocol had a subtle race condition
  - During failover, if two events occurred in a specific order:
    1. Primary fails
    2. Secondary receives a late replication message
  - A small window existed where data could be lost

  This bug was NEVER found by:
    - Code review (multiple engineers reviewed)
    - Unit tests
    - Integration tests
    - Load tests running for months
    - Production traffic

  TLA+ found it in minutes.

Why testing missed it:
  - Required exact interleaving of 3 events across 3 nodes
  - Probability of occurring in any single test run: ~10^-6
  - Would take centuries of random testing to find reliably
```

### 9.4 How TLA+ Fits AWS Development

```
AWS Engineering Workflow:

1. Design Phase
   └── Engineer writes informal design document
   └── Engineer writes TLA+ specification
   └── Run TLC to find design bugs ◀── CATCHES MOST BUGS
   └── Fix spec, re-run TLC
   └── Iterate until all properties pass

2. Implementation Phase
   └── Implement based on verified design
   └── Write unit and integration tests
   └── Reference spec for edge cases

3. Review Phase
   └── Code review includes spec review
   └── Reviewers check spec matches implementation
   └── Any design change → update spec first, re-verify

4. Maintenance Phase
   └── Spec serves as precise documentation
   └── New engineers read spec to understand design
   └── Design changes start with spec modification
```

### 9.5 Lessons from AWS's Experience

From Chris Newcombe's report:

| Lesson | Detail |
|--------|--------|
| Specs are small | TLA+ specs are typically 200-800 lines (vs 100K+ lines of code) |
| Engineers learn TLA+ in 2-3 weeks | Most AWS engineers found it approachable |
| ROI is very high | One spec found 2 bugs in 2 days that months of testing missed |
| Specs improve design | Writing the spec often simplifies the design |
| Specs are documentation | Best documentation for complex protocols |
| Not for all code | Only worth it for complex, concurrent, fault-tolerant components |

### 9.6 What TLA+ Does NOT Do

Important limitations to understand:

```
TLA+ verifies:                    TLA+ does NOT verify:
  ✓ Protocol design                 ✗ Implementation correctness
  ✓ Safety properties               ✗ Performance characteristics
  ✓ Liveness properties             ✗ Memory safety
  ✓ Edge cases in state machines    ✗ Security properties
  ✓ Concurrent interactions         ✗ Actual code (only the model)

The gap between spec and code:
  TLA+ spec says: "in action A, variable x gets value v"
  Code might have: off-by-one error implementing A
  TLA+ won't catch that.

Solution: use property-based testing to bridge the gap
  - Generate random test inputs
  - Check that code behavior matches spec behavior
  - E.g., Jepsen checks real databases against consistency specs
```

---

## 10. Practical Verification Workflow

### 10.1 Amazon's Experience with TLA+

Amazon Web Services published a landmark paper in 2014 ("Use of Formal Methods at Amazon Web Services" by Newcombe et al.) documenting their experience:

```
Teams that adopted TLA+ at AWS:
┌────────────────────┬──────────────────────────────────────────┐
│ Service            │ What was verified                        │
├────────────────────┼──────────────────────────────────────────┤
│ S3                 │ Fault-tolerant replication protocol      │
│ DynamoDB           │ Replication and group membership         │
│ EBS                │ Volume management state machine          │
│ Internal Lock Svc  │ Distributed lock correctness             │
│ CloudFormation     │ Stack update orchestration               │
└────────────────────┴──────────────────────────────────────────┘

Key findings:
  - TLA+ found 7 serious bugs across 5 systems
  - 2 bugs were in systems already in production
  - All bugs involved subtle combinations of events that
    conventional testing had missed
  - Engineers with no prior formal methods experience learned
    TLA+ in 2-3 weeks
  - Typical specification: 500-2000 lines of TLA+
  - Average verification time: 1-5 minutes for model checking
```

### 10.2 When to Use Formal Verification

Not every system needs formal verification. Use it when:

```
High Value                          Low Value
─────────────────────────────────────────────────────
✓ Consensus protocols               ✗ CRUD applications
✓ Distributed lock managers          ✗ Single-node programs
✓ Replication state machines         ✗ Stateless services
✓ Membership/reconfiguration         ✗ Simple request-response
✓ Transaction protocols              ✗ UI logic
✓ Cache coherence protocols          ✗ Batch processing

Decision criteria:
  1. Is the algorithm concurrent or distributed? → Yes: consider TLA+
  2. Are there safety-critical invariants? → Yes: strongly consider
  3. Is the state space too large for testing? → Yes: formal verification
  4. Will the protocol be used in production? → Yes: strongly consider
  5. Is the team willing to invest 2-4 weeks? → Yes: do it
```

### 10.3 The Specification-Driven Development Workflow

```
Step 1: Write English Prose Description
  └── "Nodes elect a leader using majority voting.
       A node votes for at most one candidate per term.
       The candidate with a majority becomes leader."

Step 2: Identify State Variables
  └── currentTerm, state, votedFor, votesGranted

Step 3: Write Safety Properties First
  └── OneLeaderPerTerm, TypeOK

Step 4: Write the Specification
  └── Init, Actions (StartElection, GrantVote, BecomeLeader), Next

Step 5: Run TLC — Expect Failures
  └── Fix bugs in the spec (most common: missing UNCHANGED,
       wrong action guards, underspecified types)

Step 6: Iterate Until Clean
  └── All invariants pass, no deadlocks

Step 7: Add Liveness Properties
  └── EventualLeader, ElectionProgress + fairness conditions

Step 8: Translate Spec to Code
  └── Use the spec as a design document for implementation
       Each action → one function/method
       Each variable → one data structure

Step 9: Test Code Against Spec Behaviors
  └── Trace validation: run the code, record state transitions,
       verify they match the specification's allowed behaviors
```

### 10.4 Common Specification Patterns

Patterns that appear across many distributed system specifications:

```tla
\* --- Pattern 1: Message-passing with a network set ---
VARIABLE messages
Send(m) == messages' = messages \cup {m}
Receive(m) == /\ m \in messages
              /\ messages' = messages \ {m}

\* --- Pattern 2: Quorum intersection ---
CONSTANTS Quorums
ASSUME \A Q1, Q2 \in Quorums : Q1 \cap Q2 /= {}

\* --- Pattern 3: Epoch/term-based reasoning ---
StepDown(n, term) ==
    /\ term > currentTerm[n]
    /\ currentTerm' = [currentTerm EXCEPT ![n] = term]
    /\ state' = [state EXCEPT ![n] = "follower"]
    /\ votedFor' = [votedFor EXCEPT ![n] = "none"]

\* --- Pattern 4: Non-deterministic failures ---
NodeCrash(n) ==
    /\ state' = [state EXCEPT ![n] = "crashed"]
    /\ UNCHANGED <<persistent_vars>>  \* Persistent state survives

NodeRecover(n) ==
    /\ state[n] = "crashed"
    /\ state' = [state EXCEPT ![n] = "follower"]
    /\ UNCHANGED <<persistent_vars>>

\* --- Pattern 5: State constraint for bounded model checking ---
StateConstraint ==
    /\ \A n \in Nodes : currentTerm[n] <= MaxTerm
    /\ Cardinality(messages) <= MaxMessages
    /\ \A n \in Nodes : Len(log[n]) <= MaxLogLen
```

### 10.5 Verification Pitfalls and Best Practices

Common mistakes when writing specifications:

```
Pitfall 1: Forgetting UNCHANGED
  Wrong:  Action(n) == /\ x' = x + 1
  Right:  Action(n) == /\ x' = x + 1
                       /\ UNCHANGED <<y, z>>
  TLC error: "Variable y has no next-state value"

Pitfall 2: Over-constraining the model
  Wrong:  Messages are delivered in FIFO order
  Right:  Messages can be delivered in ANY order
  Why:    Real networks reorder packets. The spec should model the
          weakest assumptions under which the protocol must work.

Pitfall 3: Under-constraining types
  Wrong:  VARIABLE x  (no type constraint)
  Right:  TypeOK == x \in 0..MaxVal
  Why:    TLC may explore states with unbounded values, never terminating.

Pitfall 4: Modeling implementation details
  Wrong:  Specifying TCP handshakes, serialization formats, timeouts
  Right:  Abstract to message delivery, non-deterministic failure
  Why:    The spec should model the PROTOCOL, not the implementation.

Pitfall 5: Ignoring fairness for liveness
  Wrong:  Spec == Init /\ [][Next]_vars  (checking liveness)
  Right:  Spec == Init /\ [][Next]_vars /\ Fairness
  Why:    Without fairness, TLC finds trivial counterexamples where
          enabled actions never fire.
```

### 10.6 From Specification to Production

A practical workflow for integrating TLA+ into an engineering team:

```
Phase 1: Learn (1-2 weeks)
  ├── Work through Lamport's "Specifying Systems" chapters 1-6
  ├── Write specs for well-known algorithms (mutex, producer-consumer)
  └── Use the TLA+ Toolbox IDE for immediate feedback

Phase 2: Specify (1-2 weeks per protocol)
  ├── Start with the English-language design document
  ├── Write TypeOK invariant first (catches most variable errors)
  ├── Add actions one at a time, checking after each addition
  └── Use small constants (3 nodes, 2 values) for fast iteration

Phase 3: Verify (days to weeks)
  ├── Run TLC with small constants, fix all violations
  ├── Gradually increase constants to build confidence
  ├── Add liveness properties after safety properties pass
  └── Document every TLC counterexample and the fix applied

Phase 4: Implement (ongoing)
  ├── Use the spec as the authoritative design document
  ├── Map each TLA+ action to a code function
  ├── Write tests that validate code traces against spec behaviors
  └── Re-verify the spec when the protocol changes

Phase 5: Maintain (ongoing)
  ├── Keep the spec in the same repository as the code
  ├── Update the spec before changing the protocol
  ├── Run TLC in CI for regression checking
  └── Onboard new team members by reading the spec first
```

### 10.7 Tools Ecosystem

```
┌──────────────────┬───────────────────────────────────────────┐
│ Tool             │ Purpose                                   │
├──────────────────┼───────────────────────────────────────────┤
│ TLA+ Toolbox     │ IDE with syntax highlighting, model       │
│                  │ configuration, and TLC integration        │
├──────────────────┼───────────────────────────────────────────┤
│ TLC              │ Explicit-state model checker              │
│                  │ (exhaustive or simulation mode)           │
├──────────────────┼───────────────────────────────────────────┤
│ TLAPS            │ TLA+ Proof System — interactive theorem   │
│                  │ prover for mathematical proofs            │
├──────────────────┼───────────────────────────────────────────┤
│ PlusCal          │ Algorithmic language that compiles to TLA+│
├──────────────────┼───────────────────────────────────────────┤
│ Apalache         │ Symbolic model checker using SMT solvers  │
│                  │ — handles larger state spaces than TLC    │
├──────────────────┼───────────────────────────────────────────┤
│ TLA+ VSCode      │ VSCode extension for TLA+ editing         │
│                  │ and basic TLC integration                 │
├──────────────────┼───────────────────────────────────────────┤
│ Alloy            │ Alternative: relational modeling language  │
│                  │ with SAT-based analysis                   │
├──────────────────┼───────────────────────────────────────────┤
│ Ivy              │ Alternative: verification language for    │
│                  │ distributed protocols with decidable logic│
├──────────────────┼───────────────────────────────────────────┤
│ P language       │ Alternative: Microsoft's state machine    │
│                  │ language with systematic testing           │
└──────────────────┴───────────────────────────────────────────┘
```

---

## 11. Other Formal Methods

### 11.1 Comparison of Formal Methods

| Tool | Paradigm | Automation | Learning Curve | Best For |
|------|----------|-----------|----------------|----------|
| **TLA+** | Temporal logic | Model checking (TLC) | Medium | Distributed protocols, concurrent systems |
| **Alloy** | Relational logic | SAT-based model finding | Low-Medium | Data models, APIs, structural properties |
| **Coq** | Dependent types | Interactive theorem proving | Very High | Mathematical proofs, verified compilers |
| **Isabelle/HOL** | Higher-order logic | Interactive theorem proving | High | Mathematical proofs, OS verification |
| **Spin/Promela** | Process algebra | Model checking | Medium | Communication protocols |
| **CBMC** | C semantics | Bounded model checking | Low | C code verification |
| **Dafny** | Hoare logic | Automated verification | Medium | Verified programs |

### 11.2 Alloy

Alloy (Daniel Jackson, MIT) uses relational logic and SAT solving to find counterexamples:

```alloy
// Example: simple file system
sig File {}
sig Directory {
    contents: set (File + Directory)
}
sig Root extends Directory {}

// No cycles in directory structure
fact NoCycles {
    no d: Directory | d in d.^contents
}

// Root contains everything (directly or indirectly)
fact RootContainsAll {
    all f: File | f in Root.^contents
}

// Check: can we have an empty root?
check NoEmptyRoot {
    some Root.contents
} for 5   // Check with up to 5 objects of each type
```

**Alloy vs TLA+**:
- Alloy: better for structural properties (data models, invariants)
- TLA+: better for behavioral properties (protocols, concurrent systems)

### 11.3 Coq

Coq is an interactive theorem prover. You write proofs that the Coq kernel mechanically verifies. Much more effort than model checking, but proofs hold for all parameter values (not just small instances).

```coq
(* Example: proving commutativity of addition *)
Theorem plus_comm : forall n m : nat, n + m = m + n.
Proof.
  intros n m.
  induction n.
  - simpl. rewrite <- plus_n_O. reflexivity.
  - simpl. rewrite IHn. rewrite plus_n_Sm. reflexivity.
Qed.
```

**Notable Coq projects**:
- **CompCert**: A fully verified C compiler
- **sel4**: Verified microkernel (uses Isabelle/HOL, similar approach)
- **Verdi**: Framework for writing verified distributed systems in Coq
- **IronFleet**: Verified distributed system implementations (Dafny)

### 11.4 When to Use Which

```
Decision tree:

Is the property about structure (data model, API)?
  YES → Alloy
  NO → Continue

Is the property about concurrent/distributed behavior?
  YES → TLA+ or Spin
  NO → Continue

Do you need a proof for ALL input sizes (not just small ones)?
  YES → Coq/Isabelle/Dafny
  NO → TLA+ with model checking

Do you want to verify actual running code (not just a model)?
  YES → Dafny, CBMC, or runtime verification
  NO → TLA+
```

---

## 12. Implementation: TLA+ Spec Validator and PlusCal Translator

### 12.1 TLA+ Spec Structure Validator

```python
"""
TLA+ Specification Structure Validator

Validates the structure of TLA+ specifications by checking for
required sections, matching delimiters, and common mistakes.

This is NOT a full TLA+ parser — it validates structural conventions
to catch common authoring errors before running TLC.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class Severity(Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationIssue:
    """A single validation finding."""
    severity: Severity
    line: int
    message: str
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Complete validation result for a TLA+ spec."""
    filename: str
    issues: List[ValidationIssue] = field(default_factory=list)
    module_name: Optional[str] = None
    variables: List[str] = field(default_factory=list)
    constants: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(i.severity == Severity.ERROR for i in self.issues)

    def summary(self) -> str:
        errors = sum(1 for i in self.issues if i.severity == Severity.ERROR)
        warnings = sum(1 for i in self.issues if i.severity == Severity.WARNING)
        infos = sum(1 for i in self.issues if i.severity == Severity.INFO)
        return (
            f"Module: {self.module_name or '(unknown)'}\n"
            f"Variables: {', '.join(self.variables) or '(none)'}\n"
            f"Constants: {', '.join(self.constants) or '(none)'}\n"
            f"Actions: {', '.join(self.actions) or '(none)'}\n"
            f"Properties: {', '.join(self.properties) or '(none)'}\n"
            f"Issues: {errors} errors, {warnings} warnings, {infos} info"
        )


class TLAPlusValidator:
    """
    Validates TLA+ specification structure.

    Checks for:
      - Module header and footer
      - EXTENDS declarations
      - VARIABLES and CONSTANTS
      - Init and Next definitions
      - Matching delimiters
      - Common mistakes
    """

    def __init__(self):
        self._patterns = {
            "module_start": re.compile(
                r"^-{4,}\s*MODULE\s+(\w+)\s*-{4,}$"
            ),
            "module_end": re.compile(r"^={4,}$"),
            "extends": re.compile(r"^EXTENDS\s+(.+)$"),
            "variables": re.compile(r"^VARIABLES?\s+(.+)$"),
            "constants": re.compile(r"^CONSTANTS?\s+(.+)$"),
            "definition": re.compile(r"^(\w+)\s*=="),
            "action_def": re.compile(r"^(\w+)\s*\(.*\)\s*=="),
            "invariant_check": re.compile(
                r"(TypeOK|Invariant|Safety|Mutex|Mutual)"
            ),
            "temporal_prop": re.compile(
                r"(Liveness|Eventually|NoStarvation|Progress|Fairness)"
            ),
            "pluscal_start": re.compile(r"\(\*--algorithm"),
            "pluscal_end": re.compile(r"end\s+algorithm;\s*\*\)"),
        }

    def validate(self, spec_text: str, filename: str = "spec.tla") -> ValidationResult:
        """
        Validate a TLA+ specification.

        Args:
            spec_text: The complete TLA+ specification text
            filename: Name of the file (for reporting)

        Returns:
            ValidationResult with findings
        """
        result = ValidationResult(filename=filename)
        lines = spec_text.split("\n")

        self._check_module_structure(lines, result)
        self._check_variables_and_constants(lines, result)
        self._check_init_next(lines, result)
        self._check_delimiters(lines, result)
        self._extract_definitions(lines, result)
        self._check_common_mistakes(lines, result)
        self._check_pluscal(lines, result)

        return result

    def _check_module_structure(
        self, lines: List[str], result: ValidationResult
    ) -> None:
        """Check for proper module header and footer."""
        found_start = False
        found_end = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            match = self._patterns["module_start"].match(stripped)
            if match:
                found_start = True
                result.module_name = match.group(1)

            if self._patterns["module_end"].match(stripped):
                found_end = True

        if not found_start:
            result.issues.append(ValidationIssue(
                Severity.ERROR, 0,
                "Missing module header. Expected: ---- MODULE Name ----",
                "Add '---- MODULE YourModuleName ----' at the top"
            ))

        if not found_end:
            result.issues.append(ValidationIssue(
                Severity.ERROR, len(lines),
                "Missing module footer. Expected: ====",
                "Add '====' at the end of the file"
            ))

    def _check_variables_and_constants(
        self, lines: List[str], result: ValidationResult
    ) -> None:
        """Check for VARIABLES and CONSTANTS declarations."""
        found_vars = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            match = self._patterns["variables"].match(stripped)
            if match:
                found_vars = True
                # Parse variable names (comma-separated)
                var_str = match.group(1).rstrip(",")
                # Handle multi-line: collect continuation lines
                var_names = [v.strip().rstrip(",") for v in var_str.split(",")]
                result.variables.extend(
                    v for v in var_names if v and not v.startswith("\\*")
                )

            match = self._patterns["constants"].match(stripped)
            if match:
                const_str = match.group(1).rstrip(",")
                const_names = [c.strip().rstrip(",") for c in const_str.split(",")]
                result.constants.extend(
                    c for c in const_names if c and not c.startswith("\\*")
                )

        if not found_vars:
            result.issues.append(ValidationIssue(
                Severity.WARNING, 0,
                "No VARIABLES declaration found",
                "Add 'VARIABLES var1, var2, ...' to declare state variables"
            ))

    def _check_init_next(
        self, lines: List[str], result: ValidationResult
    ) -> None:
        """Check for Init and Next definitions."""
        has_init = False
        has_next = False
        has_spec = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r"^Init\s*==", stripped):
                has_init = True
            if re.match(r"^Next\s*==", stripped):
                has_next = True
            if re.match(r"^Spec\s*==", stripped):
                has_spec = True

        if not has_init:
            result.issues.append(ValidationIssue(
                Severity.ERROR, 0,
                "No Init definition found",
                "Define 'Init == ...' specifying the initial state predicate"
            ))

        if not has_next:
            result.issues.append(ValidationIssue(
                Severity.ERROR, 0,
                "No Next definition found",
                "Define 'Next == ...' specifying the next-state relation"
            ))

        if not has_spec:
            result.issues.append(ValidationIssue(
                Severity.WARNING, 0,
                "No Spec definition found",
                "Define 'Spec == Init /\\ [][Next]_vars' as the full specification"
            ))

    def _check_delimiters(
        self, lines: List[str], result: ValidationResult
    ) -> None:
        """Check for matching delimiters."""
        stack = []
        openers = {"(": ")", "[": "]", "{": "}", "<<": ">>"}
        closers = {")", "]", "}", ">>"}

        for i, line in enumerate(lines):
            # Skip comments
            stripped = line.split("\\*")[0]

            j = 0
            while j < len(stripped):
                # Check for <<
                if stripped[j:j+2] == "<<":
                    stack.append(("<<", i + 1))
                    j += 2
                    continue
                # Check for >>
                if stripped[j:j+2] == ">>":
                    if stack and stack[-1][0] == "<<":
                        stack.pop()
                    else:
                        result.issues.append(ValidationIssue(
                            Severity.ERROR, i + 1,
                            f"Unmatched '>>' at line {i + 1}"
                        ))
                    j += 2
                    continue

                ch = stripped[j]
                if ch in ("(", "[", "{"):
                    stack.append((ch, i + 1))
                elif ch in closers:
                    expected_opener = {")": "(", "]": "[", "}": "{"}[ch]
                    if stack and stack[-1][0] == expected_opener:
                        stack.pop()
                    elif stack:
                        result.issues.append(ValidationIssue(
                            Severity.ERROR, i + 1,
                            f"Mismatched delimiter: expected "
                            f"'{openers[stack[-1][0]]}' but found '{ch}'"
                        ))
                    else:
                        result.issues.append(ValidationIssue(
                            Severity.ERROR, i + 1,
                            f"Unmatched closing delimiter '{ch}'"
                        ))
                j += 1

        for opener, line_num in stack:
            result.issues.append(ValidationIssue(
                Severity.ERROR, line_num,
                f"Unclosed delimiter '{opener}' opened at line {line_num}"
            ))

    def _extract_definitions(
        self, lines: List[str], result: ValidationResult
    ) -> None:
        """Extract action and property definitions."""
        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip comments
            if stripped.startswith("\\*"):
                continue

            # Check for action definitions (with parameters)
            match = self._patterns["action_def"].match(stripped)
            if match:
                name = match.group(1)
                if name not in ("Init", "Next", "Spec"):
                    result.actions.append(name)
                continue

            # Check for simple definitions
            match = self._patterns["definition"].match(stripped)
            if match:
                name = match.group(1)
                if name in ("Init", "Next", "Spec", "vars"):
                    continue
                if self._patterns["invariant_check"].search(name):
                    result.properties.append(name)
                elif self._patterns["temporal_prop"].search(name):
                    result.properties.append(name)

    def _check_common_mistakes(
        self, lines: List[str], result: ValidationResult
    ) -> None:
        """Check for common TLA+ authoring mistakes."""
        for i, line in enumerate(lines):
            stripped = line.strip()

            # Check for = instead of ==
            if re.match(r"^\w+\s*=[^=]", stripped):
                if not stripped.startswith("CONSTANT"):
                    result.issues.append(ValidationIssue(
                        Severity.WARNING, i + 1,
                        "Possible '=' instead of '==' in definition",
                        "TLA+ uses '==' for definitions and '=' for equality"
                    ))

            # Check for missing UNCHANGED
            if "'" in stripped and "UNCHANGED" not in stripped:
                primed_vars = re.findall(r"(\w+)'", stripped)
                if primed_vars and len(primed_vars) < len(result.variables):
                    # Only flag if clearly in an action body (contains /\)
                    if "/\\" in stripped or "\\/" in stripped:
                        pass  # Could check more carefully but avoid false positives

            # Check for common temporal operator mistakes
            if "[]" in stripped and "<>" in stripped:
                if "[]<>" not in stripped and "<>[]" not in stripped:
                    result.issues.append(ValidationIssue(
                        Severity.INFO, i + 1,
                        "Line contains both [] and <> — verify the nesting is correct"
                    ))

    def _check_pluscal(
        self, lines: List[str], result: ValidationResult
    ) -> None:
        """Check for PlusCal algorithm blocks."""
        pcal_start = None
        pcal_end = None

        for i, line in enumerate(lines):
            if self._patterns["pluscal_start"].search(line):
                pcal_start = i + 1
            if self._patterns["pluscal_end"].search(line):
                pcal_end = i + 1

        if pcal_start and not pcal_end:
            result.issues.append(ValidationIssue(
                Severity.ERROR, pcal_start,
                "PlusCal algorithm started but never ended",
                "Add 'end algorithm; *)' to close the PlusCal block"
            ))
        elif not pcal_start and pcal_end:
            result.issues.append(ValidationIssue(
                Severity.ERROR, pcal_end,
                "PlusCal end found without matching start",
                "Add '(*--algorithm Name' before the algorithm body"
            ))
        elif pcal_start and pcal_end:
            result.issues.append(ValidationIssue(
                Severity.INFO, pcal_start,
                f"PlusCal algorithm found (lines {pcal_start}-{pcal_end}). "
                f"Run pcal translator before TLC."
            ))


### 12.2 PlusCal-to-Pseudocode Translator

```python
"""
PlusCal to Pseudocode Translator

Translates PlusCal algorithm blocks into readable pseudocode.
Useful for documentation and design review by engineers
who are not familiar with TLA+/PlusCal syntax.
"""


class PlusCalToPseudocode:
    """
    Translates PlusCal algorithm syntax to readable pseudocode.

    Handles:
      - Variable declarations
      - Process declarations
      - Labels and statements
      - Control flow (if/then/else, while, either/or)
      - await → wait until
      - with → for each / choose
    """

    def __init__(self):
        self._indent_level = 0
        self._output_lines: List[str] = []

    def translate(self, pluscal_text: str) -> str:
        """
        Translate PlusCal text to pseudocode.

        Args:
            pluscal_text: PlusCal algorithm text (without TLA+ wrapper)

        Returns:
            Human-readable pseudocode string
        """
        self._indent_level = 0
        self._output_lines = []

        lines = pluscal_text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            i = self._process_line(line, lines, i)
            i += 1

        return "\n".join(self._output_lines)

    def _emit(self, text: str) -> None:
        """Emit a line of pseudocode with current indentation."""
        indent = "    " * self._indent_level
        self._output_lines.append(f"{indent}{text}")

    def _process_line(self, line: str, all_lines: List[str], idx: int) -> int:
        """Process a single PlusCal line. Returns the (possibly advanced) index."""

        # Skip empty lines and comments
        if not line or line.startswith("\\*"):
            return idx

        # Algorithm declaration
        match = re.match(r"\(\*--algorithm\s+(\w+)", line)
        if match:
            self._emit(f"ALGORITHM {match.group(1)}")
            self._emit("=" * 40)
            return idx

        if line.startswith("end algorithm"):
            self._emit("=" * 40)
            self._emit("END ALGORITHM")
            return idx

        # Variables
        if line.startswith("variables"):
            self._emit("SHARED VARIABLES:")
            self._indent_level += 1
            return idx

        if line.startswith("variable "):
            var_decl = line[len("variable "):].rstrip(";").rstrip(",")
            self._emit(f"- {self._translate_expr(var_decl)}")
            return idx

        # Process
        match = re.match(r"process\s+(\w+)\s*(?:\\in|=)\s*(.+)", line)
        if match:
            proc_name = match.group(1)
            proc_set = self._translate_expr(match.group(2))
            self._indent_level = 0
            self._emit("")
            self._emit(f"PROCESS {proc_name} in {proc_set}:")
            self._indent_level = 1
            return idx

        if line == "end process;":
            self._indent_level = 0
            self._emit("END PROCESS")
            return idx

        # Begin
        if line == "begin":
            return idx

        # Labels
        match = re.match(r"^(\w+):\s*(.*)", line)
        if match and not line.startswith("if") and not line.startswith("while"):
            label = match.group(1)
            rest = match.group(2).strip()
            self._emit(f"[{label}]")
            if rest:
                self._indent_level += 1
                self._emit(self._translate_statement(rest))
                self._indent_level -= 1
            return idx

        # Control flow
        if line.startswith("if ") or line.startswith("if("):
            cond = line[3:].rstrip(" then").strip()
            self._emit(f"IF {self._translate_expr(cond)} THEN")
            self._indent_level += 1
            return idx

        if line.startswith("elsif ") or line.startswith("elif "):
            cond = line.split(" ", 1)[1].rstrip(" then").strip()
            self._indent_level -= 1
            self._emit(f"ELSE IF {self._translate_expr(cond)} THEN")
            self._indent_level += 1
            return idx

        if line == "else":
            self._indent_level -= 1
            self._emit("ELSE")
            self._indent_level += 1
            return idx

        if line == "end if;" or line == "end if":
            self._indent_level -= 1
            self._emit("END IF")
            return idx

        if line.startswith("while "):
            cond = line[6:].rstrip(" do").strip()
            self._emit(f"WHILE {self._translate_expr(cond)} DO")
            self._indent_level += 1
            return idx

        if line == "end while;" or line == "end while":
            self._indent_level -= 1
            self._emit("END WHILE")
            return idx

        if line == "either":
            self._emit("NON-DETERMINISTICALLY CHOOSE:")
            self._indent_level += 1
            self._emit("OPTION 1:")
            self._indent_level += 1
            return idx

        if line == "or":
            self._indent_level -= 1
            option_num = sum(
                1 for l in self._output_lines if "OPTION" in l
            ) + 1
            self._emit(f"OPTION {option_num}:")
            self._indent_level += 1
            return idx

        if line == "end either;" or line == "end either":
            self._indent_level -= 2
            self._emit("END CHOOSE")
            return idx

        # Statements
        self._emit(self._translate_statement(line))
        return idx

    def _translate_statement(self, stmt: str) -> str:
        """Translate a single PlusCal statement to pseudocode."""
        stmt = stmt.rstrip(";").strip()

        # await → wait until
        if stmt.startswith("await "):
            cond = stmt[6:]
            return f"WAIT UNTIL {self._translate_expr(cond)}"

        # Assignment
        if ":=" in stmt:
            parts = stmt.split(":=", 1)
            lhs = parts[0].strip()
            rhs = self._translate_expr(parts[1].strip())
            return f"{lhs} <- {rhs}"

        # skip
        if stmt == "skip":
            return "// do nothing"

        # goto
        if stmt.startswith("goto "):
            return f"GOTO [{stmt[5:]}]"

        # assert
        if stmt.startswith("assert "):
            return f"ASSERT {self._translate_expr(stmt[7:])}"

        # print
        if stmt.startswith("print "):
            return f"PRINT {stmt[6:]}"

        return stmt

    def _translate_expr(self, expr: str) -> str:
        """Translate TLA+ expression syntax to more readable form."""
        # Replace TLA+ operators with readable equivalents
        replacements = [
            (r"\\in", "in"),
            (r"\\/", "OR"),
            (r"/\\", "AND"),
            (r"\\union", "UNION"),
            (r"\\subseteq", "SUBSET OF"),
            (r"\\cup", "UNION"),
            (r"\\cap", "INTERSECT"),
            (r"~", "NOT"),
            (r"\\lnot", "NOT"),
            (r"\\E", "EXISTS"),
            (r"\\A", "FOR ALL"),
            (r"\\X", "CROSS"),
            (r"EXCEPT !\[", "with ["),
            (r"\|->", "=>"),
        ]

        result = expr
        for pattern, replacement in replacements:
            result = result.replace(pattern, replacement)

        return result.strip()


# --- Demo and Testing ---

def demo_validator():
    """Demonstrate the TLA+ spec validator."""
    spec = """
---- MODULE SimpleCounter ----
EXTENDS Integers

CONSTANTS MaxValue

VARIABLES counter

vars == <<counter>>

Init ==
    /\\ counter = 0

Increment ==
    /\\ counter < MaxValue
    /\\ counter' = counter + 1

Decrement ==
    /\\ counter > 0
    /\\ counter' = counter - 1

Next ==
    \\/ Increment
    \\/ Decrement

Spec == Init /\\ [][Next]_vars /\\ WF_vars(Next)

TypeOK == counter \\in 0..MaxValue

EventuallyMax == <>(counter = MaxValue)

====
"""
    validator = TLAPlusValidator()
    result = validator.validate(spec, "SimpleCounter.tla")

    print("=== TLA+ Spec Validation ===\n")
    print(result.summary())
    print()

    for issue in result.issues:
        print(f"  [{issue.severity.value}] Line {issue.line}: {issue.message}")
        if issue.suggestion:
            print(f"    Suggestion: {issue.suggestion}")

    print(f"\n  Valid: {result.is_valid}")


def demo_translator():
    """Demonstrate the PlusCal to pseudocode translator."""
    pluscal = """(*--algorithm Peterson
variables
    flag = [i \\in {0, 1} |-> FALSE],
    turn = 0;

process proc \\in {0, 1}
begin
    P1: flag[self] := TRUE;
    P2: turn := 1 - self;
    P3: await flag[1 - self] = FALSE \\/ turn = self;
    CS: skip;
    P4: flag[self] := FALSE;
    P5: goto P1;
end process;

end algorithm; *)"""

    translator = PlusCalToPseudocode()
    pseudocode = translator.translate(pluscal)

    print("=== PlusCal to Pseudocode Translation ===\n")
    print("--- PlusCal Input ---")
    print(pluscal)
    print("\n--- Pseudocode Output ---")
    print(pseudocode)


if __name__ == "__main__":
    demo_validator()
    print("\n" + "=" * 60 + "\n")
    demo_translator()
```

---

## 13. Summary and Further Reading

### Key Takeaways

| Concept | Key Insight |
|---------|-------------|
| Testing vs verification | Testing finds bugs; verification proves absence (within the model) |
| TLA+ | Describes *what* a system does, not *how* — powerful for finding design bugs |
| States and actions | System = initial state + next-state relation; behaviors = infinite state sequences |
| Safety properties | Invariants checked in every reachable state (□P) |
| Liveness properties | Temporal properties checked on all behaviors (◇P, P ~> Q) |
| PlusCal | Pseudocode-like syntax that compiles to TLA+; easier entry point |
| TLC model checker | Exhaustively explores all reachable states; finds shortest counterexamples |
| AWS experience | TLA+ found critical bugs that months of testing missed; ROI is very high |
| Practical scope | Verify the design (not the code); bridge gap with property-based testing |

### Essential Reading

1. **Lamport (2002)** — "Specifying Systems: The TLA+ Language and Tools for Hardware and Software Engineers" (the definitive textbook)
2. **Newcombe et al. (2015)** — "How Amazon Web Services Uses Formal Methods" (CACM)
3. **Lamport (2009)** — "The PlusCal Algorithm Language" (PlusCal tutorial)
4. **Lamport** — "Learn TLA+" website (https://learntla.com)
5. **Wayne (2018)** — "Practical TLA+: Planning Driven Development" (book)

### Connection to Other Lessons

- **Lesson 3 (FLP)**: FLP impossibility can be expressed and explored in TLA+
- **Lesson 5 (Paxos)**: Lamport's original Paxos specification is in TLA+
- **Lesson 6 (Raft)**: The Raft paper includes a TLA+ specification by Diego Ongaro
- **Lesson 16 (Capstone)**: Our KV store design could be verified with TLA+ before implementation

---

[Next: Capstone — Building a Distributed KV Store](./16_Capstone_Building_Distributed_KV_Store.md)

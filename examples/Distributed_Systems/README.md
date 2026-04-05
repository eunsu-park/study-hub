# Distributed Systems — Examples

Hands-on Python examples covering core distributed systems concepts: logical clocks, consensus, CRDTs, consistent hashing, failure detection, and replicated state machines.

## Files

| # | File | Concepts | Description |
|---|------|----------|-------------|
| 01 | `01_lamport_clock.py` | Logical clocks, happens-before | Lamport timestamp implementation with simulated message passing between processes |
| 02 | `02_vector_clock.py` | Vector clocks, causality detection | Vector clock with comparison operators; detects concurrent vs causally-related events |
| 03 | `03_raft_leader_election.py` | Raft, leader election, terms | Async Raft leader election with 5-node cluster, randomized timeouts, and re-election |
| 04 | `04_raft_log_replication.py` | Raft, log replication, partitions | Log replication with consistency checks, network partitions, and log reconciliation |
| 05 | `05_crdt_counters.py` | CRDTs, eventual consistency | G-Counter and PN-Counter with partition simulation and CRDT property verification |
| 06 | `06_consistent_hashing.py` | Consistent hashing, virtual nodes | Hash ring with configurable vnodes; measures key redistribution and distribution uniformity |
| 07 | `07_phi_accrual_detector.py` | Failure detection, statistics | Phi accrual failure detector tracking heartbeat distributions (used in Cassandra/Akka) |
| 08 | `08_distributed_kv_store.py` | Raft consensus, state machines | Complete distributed KV store with Raft consensus, node failures, and partition recovery |

## Learning Path

```
Start here
    |
    v
01_lamport_clock ──> 02_vector_clock
    (logical time)      (causality)
                            |
                            v
                   03_raft_leader_election ──> 04_raft_log_replication
                        (consensus)               (replication)
                                                       |
                                                       v
05_crdt_counters        06_consistent_hashing   08_distributed_kv_store
  (convergence)           (partitioning)          (puts it all together)
       |                       |
       v                       v
07_phi_accrual_detector ───────┘
   (failure detection)
```

**Recommended order:**
1. **01 → 02**: Understand logical time and causality (foundation for everything else)
2. **03 → 04**: Learn Raft consensus (leader election, then log replication)
3. **05**: Explore an alternative to consensus — CRDTs for eventual consistency
4. **06**: Learn how data is distributed across nodes
5. **07**: Understand how nodes detect failures
6. **08**: See everything come together in a distributed KV store

## How to Run

Each file is self-contained and runnable:

```bash
python 01_lamport_clock.py
python 02_vector_clock.py
python 03_raft_leader_election.py
# ... and so on
```

## Prerequisites

- **Python 3.10+** (uses `match` statements, `X | Y` union types)
- **No external packages required** — all examples use the standard library only
- `03_raft_leader_election.py` uses `asyncio` (included in stdlib)

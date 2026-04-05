# Distributed_Systems Exercises

Practice problem solutions for the Distributed Systems topic (16 lessons). Each file corresponds to a lesson and contains 3 working exercises with implementations.

## Exercise Files

| # | File | Lesson | Description |
|---|------|--------|-------------|
| 01 | `01_system_models_and_failure_modes.py` | System Models and Failure Modes | System model classification, crash-recovery simulation, Byzantine process |
| 02 | `02_time_clocks_and_ordering.py` | Time, Clocks, and Ordering | Hybrid logical clock, vector clock causal violations, version vectors |
| 03 | `03_flp_impossibility_and_bounds.py` | FLP Impossibility and Bounds | Adversarial consensus, Ben-Or randomized consensus, failure detectors |
| 04 | `04_consistency_models.py` | Consistency Models | Linearizability checker, sequential consistency, session guarantees |
| 05 | `05_paxos_family.py` | Paxos Family | Single-decree Paxos, dueling proposers livelock, Multi-Paxos |
| 06 | `06_raft_in_depth.py` | Raft in Depth | Log compaction/snapshots, Pre-Vote protocol, ReadIndex |
| 07 | `07_byzantine_fault_tolerance.py` | Byzantine Fault Tolerance | OM(1) algorithm, PBFT phases, view change |
| 08 | `08_distributed_transactions.py` | Distributed Transactions | 2PC with WAL, Saga pattern, 2PC vs 3PC comparison |
| 09 | `09_replication_strategies.py` | Replication Strategies | Quorum read/write, read repair, Merkle tree anti-entropy |
| 10 | `10_crdts_and_eventual_consistency.py` | CRDTs and Eventual Consistency | OR-Set, LWW-Element-Set, CRDT text editor |
| 11 | `11_partitioning_and_sharding.py` | Partitioning and Sharding | Consistent hashing with bounded load, vnode uniformity, range partitioning |
| 12 | `12_distributed_storage_case_studies.py` | Distributed Storage Case Studies | Dynamo sloppy quorum, Kafka ISR manager, TrueTime API |
| 13 | `13_failure_detection_and_membership.py` | Failure Detection and Membership | SWIM protocol, push-pull gossip, failure detector comparison |
| 14 | `14_distributed_coordination_primitives.py` | Distributed Coordination Primitives | Fencing tokens, Redlock with GC pause, Snowflake ID generator |
| 15 | `15_formal_verification_tlaplus.py` | Formal Verification with TLA+ | TLA+ spec validation, PlusCal to pseudocode, Raft election invariants |
| 16 | `16_capstone_building_distributed_kv_store.py` | Capstone: Building a Distributed KV Store | Snapshot/restore, request deduplication, fault injection harness |

## How to Run

Each file is self-contained and executable:

```bash
# Run a single exercise file
python exercises/Distributed_Systems/01_system_models_and_failure_modes.py

# Run all exercises
for f in exercises/Distributed_Systems/*.py; do python "$f"; done
```

## Prerequisites

- Python 3.8+
- No external dependencies (all exercises use the standard library only)

## File Structure

Each `.py` file follows this pattern:

```python
"""
Exercises for Lesson XX: Title
Topic: Distributed_Systems
"""

# === Exercise 1: Title ===
# Problem: Description

def exercise_1():
    """Solution with explanation."""
    ...

# === Exercise 2: Title ===
...

# === Exercise 3: Title ===
...

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
```

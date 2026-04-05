# Distributed Systems Study Guide

## Introduction

Distributed systems coordinate multiple networked computers to achieve tasks that no single machine can handle alone. This topic goes beyond conceptual introductions (covered in System_Design) to provide implementation-level mastery of consensus protocols, distributed transactions, conflict-free replicated data types (CRDTs), and formal verification — the building blocks of every modern large-scale system from cloud databases to blockchain networks.

## Target Audience

- Backend engineers designing fault-tolerant, scalable services
- Infrastructure engineers building or operating distributed databases and message queues
- Systems researchers studying consensus, consistency, and replication
- Anyone preparing for distributed systems interviews or graduate-level coursework

## Prerequisites

- **[Networking](../Networking/00_Overview.md)**: TCP/IP fundamentals, message passing, network failure modes
- **[System_Design](../System_Design/00_Overview.md)**: Foundational distributed concepts (CAP theorem, leader election, Lamport clocks introduced in L15-L16)
- **[Algorithm](../Algorithm/00_Overview.md)**: Graph algorithms, complexity analysis, proof techniques

## Learning Roadmap

```
Block A: Foundations (L01-04)         Block B: Core Protocols (L05-08)
┌─────────────────────────────┐      ┌─────────────────────────────┐
│ L01 System Models & Failures│      │ L05 Paxos Family            │
│ L02 Time, Clocks & Ordering │─────▶│ L06 Raft In Depth           │
│ L03 FLP Impossibility       │      │ L07 Byzantine Fault Toler.  │
│ L04 Consistency Models      │      │ L08 Distributed Transactions│
└─────────────────────────────┘      └──────────────┬──────────────┘
                                                     │
                                                     ▼
Block D: Production (L13-16)         Block C: Data & Consistency (L09-12)
┌─────────────────────────────┐      ┌─────────────────────────────┐
│ L13 Failure Detection       │◀─────│ L09 Replication Strategies  │
│ L14 Coordination Primitives │      │ L10 CRDTs & Eventual Cons.  │
│ L15 TLA+ Verification       │      │ L11 Partitioning & Sharding │
│ L16 Capstone: Distributed KV│      │ L12 Storage Case Studies    │
└──────────────┬──────────────┘      └─────────────────────────────┘
               │
               ▼
Block E: Advanced Impl (L18-22)      Block F: Patterns & Testing (L23-28)
┌─────────────────────────────┐      ┌─────────────────────────────┐
│ L18 Raft Implementation Pt.1│      │ L23 Distributed Rate Limit  │
│ L19 Raft Implementation Pt.2│─────▶│ L24 Event Sourcing & CQRS   │
│ L20 Distributed Hash Tables │      │ L25 Vector Clocks           │
│ L21 Gossip Protocols        │      │ L26 Distributed Testing     │
│ L22 Service Discovery       │      │ L27 Distributed Observab.   │
└─────────────────────────────┘      │ L28 Capstone: Prod. KV Store│
                                     └─────────────────────────────┘
```

## File List

| Lesson | Filename | Difficulty | Description |
|--------|----------|------------|-------------|
| L01 | `01_System_Models_and_Failure_Modes.md` | ⭐⭐ | Synchronous/asynchronous/partially synchronous models, crash and Byzantine failures, safety vs liveness |
| L02 | `02_Time_Clocks_and_Ordering.md` | ⭐⭐⭐ | Happens-before, Lamport/vector/hybrid logical clocks, causality tracking |
| L03 | `03_FLP_Impossibility_and_Bounds.md` | ⭐⭐⭐⭐ | FLP impossibility theorem, consensus round lower bounds, circumvention strategies |
| L04 | `04_Consistency_Models.md` | ⭐⭐⭐ | Linearizability, sequential/causal/eventual consistency, PACELC theorem |
| L05 | `05_Paxos_Family.md` | ⭐⭐⭐⭐ | Single-decree Paxos, Multi-Paxos, FPaxos, EPaxos with edge case analysis |
| L06 | `06_Raft_In_Depth.md` | ⭐⭐⭐ | Pre-vote, log compaction, joint consensus membership, ReadIndex/LeaseRead |
| L07 | `07_Byzantine_Fault_Tolerance.md` | ⭐⭐⭐⭐ | PBFT message flow, HotStuff linear BFT, Tendermint, 3f+1 vs 2f+1 bounds |
| L08 | `08_Distributed_Transactions.md` | ⭐⭐⭐ | 2PC blocking analysis, 3PC limits, Percolator, Spanner TrueTime, Calvin, Sagas |
| L09 | `09_Replication_Strategies.md` | ⭐⭐⭐ | Single/multi/leaderless replication, chain replication, quorums, read-repair |
| L10 | `10_CRDTs_and_Eventual_Consistency.md` | ⭐⭐⭐ | State/op-based CRDTs (G-Counter, OR-Set, LWW-Register), Automerge, Yjs |
| L11 | `11_Partitioning_and_Sharding.md` | ⭐⭐⭐ | Hash/range partitioning, consistent hashing, secondary index partitioning |
| L12 | `12_Distributed_Storage_Case_Studies.md` | ⭐⭐⭐⭐ | Spanner, Dynamo, Kafka, CockroachDB deep analysis |
| L13 | `13_Failure_Detection_and_Membership.md` | ⭐⭐⭐ | Phi accrual detector, SWIM protocol, gossip-based membership |
| L14 | `14_Distributed_Coordination_Primitives.md` | ⭐⭐⭐ | Distributed locks (Chubby, Redlock), barriers, fencing tokens, service discovery |
| L15 | `15_Formal_Verification_TLAplus.md` | ⭐⭐⭐⭐ | TLA+ basics, protocol specification, TLC model checking, AWS use cases |
| L16 | `16_Capstone_Building_Distributed_KV_Store.md` | ⭐⭐⭐⭐ | End-to-end Raft-based distributed KV store with fault injection testing |
| L18 | `18_Raft_Implementation_Part1.md` | ⭐⭐⭐⭐ | Raft leader election, log replication, safety proofs, state machine |
| L19 | `19_Raft_Implementation_Part2.md` | ⭐⭐⭐⭐ | Membership changes, log compaction, snapshots, linearizable reads |
| L20 | `20_Distributed_Hash_Tables.md` | ⭐⭐⭐ | Consistent hashing, Chord, Kademlia, virtual nodes, bounded load |
| L21 | `21_Gossip_Protocols.md` | ⭐⭐⭐ | Epidemic protocols, SWIM, push/pull gossip, phi-accrual failure detection |
| L22 | `22_Service_Discovery.md` | ⭐⭐⭐ | Consul, etcd, DNS-based discovery, health checking, load balancing |
| L23 | `23_Distributed_Rate_Limiting.md` | ⭐⭐⭐ | Token bucket, sliding window, distributed counters, Redis-based limiting |
| L24 | `24_Event_Sourcing_CQRS.md` | ⭐⭐⭐ | Event sourcing patterns, CQRS, event store, projections, snapshots |
| L25 | `25_Vector_Clocks.md` | ⭐⭐⭐⭐ | Logical clocks, vector clocks, version vectors, conflict resolution, HLC |
| L26 | `26_Distributed_Testing.md` | ⭐⭐⭐⭐ | Jepsen, fault injection, chaos engineering, deterministic simulation |
| L27 | `27_Distributed_Observability.md` | ⭐⭐⭐ | Distributed tracing, correlation IDs, structured logging, metrics |
| L28 | `28_Capstone_Distributed_KV.md` | ⭐⭐⭐⭐ | Production distributed KV store: Raft + sharding + replication + testing |

## Difficulty Guide

- ⭐⭐: Builds on prerequisites; conceptual understanding with straightforward implementations
- ⭐⭐⭐: Requires hands-on implementation skills; involves algorithmic complexity
- ⭐⭐⭐⭐: Research-level protocols; formal proofs and complex multi-component integration

## Environment Setup

```bash
pip install numpy matplotlib
pip install grpcio grpcio-tools    # for RPC examples
pip install asyncio aiohttp        # async networking
pip install sortedcontainers       # for ordered data structures
```

## Related Topics

- **[System_Design](../System_Design/00_Overview.md)**: Conceptual introduction to distributed systems (L15-L16 are prerequisites)
- **[Database_Theory](../Database_Theory/00_Overview.md)**: Distributed databases from a storage perspective (L14)
- **[Data_Engineering](../Data_Engineering/00_Overview.md)**: Distributed data pipelines, Kafka, stream processing
- **[DevOps](../DevOps/00_Overview.md)**: Distributed tracing and observability (L12)
- **[Security](../Security/00_Overview.md)**: Byzantine fault tolerance connects to secure distributed protocols

## Study Tips

1. **Implement the algorithms** — reading about Paxos is not enough; write a simulator to see edge cases firsthand
2. **Draw sequence diagrams** — consensus protocols become clear when you trace message flows on paper
3. **Inject failures deliberately** — test your implementations with dropped messages, delayed responses, and node crashes
4. **Read the original papers** — Lamport's Paxos, Ongaro's Raft, and Castro's PBFT are surprisingly readable
5. **Use Jepsen reports** — Kyle Kingsbury's analyses reveal how real databases violate their consistency claims
6. **Build incrementally** — start with a single-node KV store, add replication, then consensus, then fault tolerance

## Learning Outcomes

After completing this topic, you will be able to:

- Classify distributed system models and reason about safety and liveness properties
- Implement Lamport clocks, vector clocks, and hybrid logical clocks
- Explain and implement core consensus protocols (Paxos, Raft) with edge case handling
- Design distributed transactions using 2PC, Percolator, or Saga patterns
- Build CRDT data structures for conflict-free eventual consistency
- Analyze real-world systems (Spanner, Dynamo, Kafka) through a distributed systems lens
- Write TLA+ specifications to formally verify protocol correctness
- Build a complete Raft-based distributed key-value store

## Next Steps

- **For database engineers**: Explore [Data_Engineering](../Data_Engineering/00_Overview.md) and [Database_Theory](../Database_Theory/00_Overview.md) for storage-specific patterns
- **For infrastructure engineers**: Continue to [DevOps](../DevOps/00_Overview.md) and [Cloud_Computing](../Cloud_Computing/00_Overview.md)
- **For researchers**: Dive into [Algorithm](../Algorithm/00_Overview.md) for formal proof techniques

---

**License**: CC BY-NC 4.0

[Start: Lesson 01](./01_System_Models_and_Failure_Modes.md)

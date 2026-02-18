# Database Theory

## Overview

This topic covers the theoretical foundations of database systems, from the relational model and normalization theory to transaction processing, query optimization, and modern distributed paradigms. These lessons provide the academic underpinning that every database practitioner, backend engineer, and data architect needs to design correct, efficient, and scalable data systems.

## Prerequisites

- Basic programming experience (Python or any language)
- SQL familiarity helpful but not required (covered from first principles)
- Elementary set theory and logic (sets, relations, predicates)
- Basic understanding of file systems and data storage concepts

## Lesson Plan

### Phase 1: Foundations (L01-L04)

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [01_Introduction_to_Database_Systems.md](./01_Introduction_to_Database_Systems.md) | ⭐ | DBMS, Three-Schema Architecture, Data Independence, ANSI/SPARC | Conceptual foundation |
| [02_Relational_Model.md](./02_Relational_Model.md) | ⭐⭐ | Codd's Rules, Relations, Keys, Integrity Constraints, NULL Semantics | Mathematical foundation |
| [03_Relational_Algebra.md](./03_Relational_Algebra.md) | ⭐⭐ | σ, π, ⋈, ÷, Query Trees, Relational Calculus, SQL Equivalence | Formal query languages |
| [04_ER_Modeling.md](./04_ER_Modeling.md) | ⭐⭐ | Entities, Relationships, Cardinality, EER, ER-to-Relational Mapping | Conceptual design |

### Phase 2: Design Theory (L05-L08)

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [05_Functional_Dependencies.md](./05_Functional_Dependencies.md) | ⭐⭐⭐ | FDs, Armstrong's Axioms, Closure, Canonical Cover, Attribute Closure | Formal dependency theory |
| [06_Normalization.md](./06_Normalization.md) | ⭐⭐⭐ | 1NF-5NF, BCNF, Decomposition, Lossless Join, Dependency Preservation | Schema refinement |
| [07_SQL_Theory.md](./07_SQL_Theory.md) | ⭐⭐ | DDL, DML, Subqueries, Aggregation, Views, NULL Handling, Relational Completeness | SQL as formal language |
| [08_Constraints_and_Triggers.md](./08_Constraints_and_Triggers.md) | ⭐⭐⭐ | CHECK, UNIQUE, FK Actions, Assertions, Triggers, Active Databases | Declarative integrity |

### Phase 3: Internals (L09-L12)

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [09_Storage_and_Indexing.md](./09_Storage_and_Indexing.md) | ⭐⭐⭐ | Disk I/O, B+ Trees, Hash Indexes, Clustered vs Non-Clustered, Buffer Management | Physical storage layer |
| [10_Query_Processing.md](./10_Query_Processing.md) | ⭐⭐⭐⭐ | Parsing, Optimization, Cost Estimation, Join Algorithms, Execution Plans | Query engine internals |
| [11_Transaction_Theory.md](./11_Transaction_Theory.md) | ⭐⭐⭐⭐ | ACID, Serializability, Conflict/View Equivalence, Schedules, Recoverability | Concurrency theory |
| [12_Concurrency_Control.md](./12_Concurrency_Control.md) | ⭐⭐⭐⭐ | 2PL, Deadlock, Timestamp Ordering, MVCC, Isolation Levels, Snapshot Isolation | Lock and version protocols |

### Phase 4: Advanced Topics (L13-L16)

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [13_Recovery_Systems.md](./13_Recovery_Systems.md) | ⭐⭐⭐⭐ | WAL, ARIES, Checkpointing, Undo/Redo, Shadow Paging, Media Recovery | Crash recovery |
| [14_NoSQL_and_NewSQL.md](./14_NoSQL_and_NewSQL.md) | ⭐⭐⭐ | Key-Value, Document, Column-Family, Graph, CAP, BASE, NewSQL | Beyond relational |
| [15_Distributed_Databases.md](./15_Distributed_Databases.md) | ⭐⭐⭐⭐ | Fragmentation, Replication, 2PC, Paxos/Raft, Distributed Joins, CAP Theorem | Distributed systems |
| [16_Database_Design_Project.md](./16_Database_Design_Project.md) | ⭐⭐⭐⭐ | Full Design Lifecycle: Requirements → ER → Relational → Normalization → SQL | Capstone project |

## Recommended Learning Path

```
Phase 1: Foundations (L01-L04)         Phase 2: Design Theory (L05-L08)
       │                                       │
       ▼                                       ▼
  DBMS Concepts                         Functional Dependencies
  Relational Model                      Normalization (1NF-5NF)
  Relational Algebra                    SQL as Formal Language
  ER Modeling                           Constraints & Triggers
       │                                       │
       └───────────────┬───────────────────────┘
                       │
                       ▼
         Phase 3: Internals (L09-L12)
         Storage, Indexing, B+ Trees
         Query Processing & Optimization
         Transactions & Concurrency
                       │
                       ▼
         Phase 4: Advanced (L13-L16)
         Recovery (WAL, ARIES)
         NoSQL & NewSQL Paradigms
         Distributed Databases
                       │
                       ▼
         Capstone Design Project (L16)
```

## Related Topics

- **[PostgreSQL](../PostgreSQL/00_Overview.md)**: Practical SQL and advanced PostgreSQL features (JSON, FTS, replication, RLS)
- **[System_Design](../System_Design/00_Overview.md)**: Scalability, distributed systems, and architecture patterns that build on database theory
- **[Data_Engineering](../Data_Engineering/00_Overview.md)**: ETL pipelines, data warehousing, and data infrastructure
- **[Data_Science](../Data_Science/00_Overview.md)**: Statistical analysis and data manipulation that relies on sound database design

## Total

- **16 lessons** (4 foundations + 4 design theory + 4 internals + 4 advanced)
- **Difficulty range**: ⭐ to ⭐⭐⭐⭐
- **Languages**: SQL (primary), Python (supplementary)
- **Key concepts**: Relational model, normalization, ACID, serializability, B+ trees, ARIES, CAP theorem

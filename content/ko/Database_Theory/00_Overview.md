# 데이터베이스 이론(Database Theory)

## 개요(Overview)

이 주제는 관계형 모델(Relational Model)과 정규화 이론(Normalization Theory)부터 트랜잭션 처리(Transaction Processing), 쿼리 최적화(Query Optimization), 그리고 현대의 분산 패러다임(Distributed Paradigms)에 이르기까지 데이터베이스 시스템의 이론적 기반을 다룹니다. 이 레슨들은 모든 데이터베이스 실무자, 백엔드 엔지니어, 그리고 데이터 아키텍트가 올바르고 효율적이며 확장 가능한 데이터 시스템을 설계하기 위해 필요한 학문적 기반을 제공합니다.

## 사전 준비(Prerequisites)

- 기본 프로그래밍 경험 (Python 또는 다른 언어)
- SQL 기초 지식이 도움되지만 필수는 아님 (기본 원리부터 다룸)
- 기초 집합론 및 논리학 (집합, 관계, 서술논리)
- 파일 시스템과 데이터 저장 개념에 대한 기본 이해

## 학습 계획(Lesson Plan)

### Phase 1: 기초(Foundations) (L01-L04)

| 파일명 | 난이도 | 핵심 주제 | 비고 |
|----------|------------|------------|-------|
| [01_Introduction_to_Database_Systems.md](./01_Introduction_to_Database_Systems.md) | ⭐ | DBMS, Three-Schema Architecture, Data Independence, ANSI/SPARC | 개념적 기반 |
| [02_Relational_Model.md](./02_Relational_Model.md) | ⭐⭐ | Codd's Rules, Relations, Keys, Integrity Constraints, NULL Semantics | 수학적 기반 |
| [03_Relational_Algebra.md](./03_Relational_Algebra.md) | ⭐⭐ | σ, π, ⋈, ÷, Query Trees, Relational Calculus, SQL Equivalence | 형식 질의 언어 |
| [04_ER_Modeling.md](./04_ER_Modeling.md) | ⭐⭐ | Entities, Relationships, Cardinality, EER, ER-to-Relational Mapping | 개념적 설계 |

### Phase 2: 설계 이론(Design Theory) (L05-L08)

| 파일명 | 난이도 | 핵심 주제 | 비고 |
|----------|------------|------------|-------|
| [05_Functional_Dependencies.md](./05_Functional_Dependencies.md) | ⭐⭐⭐ | FDs, Armstrong's Axioms, Closure, Canonical Cover, Attribute Closure | 형식 종속성 이론 |
| [06_Normalization.md](./06_Normalization.md) | ⭐⭐⭐ | 1NF-5NF, BCNF, Decomposition, Lossless Join, Dependency Preservation | 스키마 정제 |
| [07_Advanced_Normalization.md](./07_Advanced_Normalization.md) | ⭐⭐⭐ | 4NF, 5NF, DKNF, MVDs, Join Dependencies | 고급 정규화 |
| [08_Query_Processing.md](./08_Query_Processing.md) | ⭐⭐⭐⭐ | Parsing, Optimization, Cost Estimation, Join Algorithms | 쿼리 엔진 내부 |

### Phase 3: 내부 구조(Internals) (L09-L12)

| 파일명 | 난이도 | 핵심 주제 | 비고 |
|----------|------------|------------|-------|
| [09_Indexing.md](./09_Indexing.md) | ⭐⭐⭐ | B+ Trees, Hash Indexes, Bitmap, Multi-level Indexing | 인덱스 구조 |
| [10_Transaction_Theory.md](./10_Transaction_Theory.md) | ⭐⭐⭐⭐ | ACID, Serializability, Conflict/View Equivalence, Schedules, Recoverability | 트랜잭션 이론 |
| [11_Concurrency_Control.md](./11_Concurrency_Control.md) | ⭐⭐⭐⭐ | 2PL, Deadlock, Timestamp Ordering, MVCC, Isolation Levels, Snapshot Isolation | 동시성 제어 프로토콜 |
| [12_Recovery_Systems.md](./12_Recovery_Systems.md) | ⭐⭐⭐⭐ | WAL, ARIES, Checkpointing, Undo/Redo, Shadow Paging, Media Recovery | 장애 복구 |

### Phase 4: 고급 주제(Advanced Topics) (L13-L16)

| 파일명 | 난이도 | 핵심 주제 | 비고 |
|----------|------------|------------|-------|
| [13_NoSQL_Data_Models.md](./13_NoSQL_Data_Models.md) | ⭐⭐⭐ | Key-Value, Document, Column-Family, Graph, CAP Theorem | NoSQL 패러다임 |
| [14_Distributed_Databases.md](./14_Distributed_Databases.md) | ⭐⭐⭐⭐ | Fragmentation, Replication, 2PC, Paxos/Raft, Distributed Joins | 분산 시스템 |
| [15_NewSQL_and_Modern_Trends.md](./15_NewSQL_and_Modern_Trends.md) | ⭐⭐⭐ | Spanner, CockroachDB, HTAP, NewSQL Architecture | 현대 트렌드 |
| [16_Database_Design_Case_Study.md](./16_Database_Design_Case_Study.md) | ⭐⭐⭐⭐ | Full Design Lifecycle: Requirements → ER → Relational → Normalization → SQL | 종합 프로젝트 |

## 추천 학습 경로(Recommended Learning Path)

```
Phase 1: 기초 (L01-L04)         Phase 2: 설계 이론 (L05-L08)
       │                                       │
       ▼                                       ▼
  DBMS 개념                           함수 종속성
  관계형 모델                         정규화 (1NF-5NF)
  관계 대수                           고급 정규화 (4NF-DKNF)
  ER 모델링                           쿼리 처리 및 최적화
       │                                       │
       └───────────────┬───────────────────────┘
                       │
                       ▼
         Phase 3: 내부 구조 (L09-L12)
         인덱싱 (B+ 트리, 해시)
         트랜잭션 및 직렬화가능성
         동시성 제어 및 복구
                       │
                       ▼
         Phase 4: 고급 (L13-L16)
         NoSQL 데이터 모델
         분산 데이터베이스
         NewSQL 및 현대 트렌드
                       │
                       ▼
         설계 사례 연구 (L16)
```

## 관련 주제(Related Topics)

- **[PostgreSQL](../PostgreSQL/00_Overview.md)**: 실용적인 SQL 및 고급 PostgreSQL 기능 (JSON, FTS, replication, RLS)
- **[System_Design](../System_Design/00_Overview.md)**: 데이터베이스 이론을 기반으로 한 확장성, 분산 시스템, 아키텍처 패턴
- **[Data_Engineering](../Data_Engineering/00_Overview.md)**: ETL 파이프라인, 데이터 웨어하우징, 데이터 인프라
- **[Data_Science](../Data_Science/00_Overview.md)**: 견고한 데이터베이스 설계에 기반한 통계 분석 및 데이터 조작

## 총계(Total)

- **16개 레슨** (4개 기초 + 4개 설계 이론 + 4개 내부 구조 + 4개 고급)
- **난이도 범위**: ⭐ ~ ⭐⭐⭐⭐
- **언어**: SQL (주), Python (보조)
- **핵심 개념**: 관계형 모델, 정규화, ACID, 직렬화가능성, B+ 트리, ARIES, CAP 정리

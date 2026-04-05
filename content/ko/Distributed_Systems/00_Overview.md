# 분산 시스템 학습 가이드

## 소개

분산 시스템은 여러 네트워크 연결된 컴퓨터들이 단일 머신으로는 처리할 수 없는 작업을 달성하기 위해 조율하는 시스템입니다. 이 토픽은 개념적 소개(System_Design에서 다룸)를 넘어, 합의 프로토콜, 분산 트랜잭션, 충돌 없는 복제 데이터 타입(CRDT), 형식 검증 등의 구현 수준 숙달을 제공합니다 — 클라우드 데이터베이스부터 블록체인 네트워크까지 모든 현대 대규모 시스템의 구성 요소입니다.

## 대상 독자

- 장애 허용적이고 확장 가능한 서비스를 설계하는 백엔드 엔지니어
- 분산 데이터베이스와 메시지 큐를 구축하거나 운영하는 인프라 엔지니어
- 합의, 일관성, 복제를 연구하는 시스템 연구자
- 분산 시스템 면접 또는 대학원 수준 과정을 준비하는 모든 분

## 선수 과목

- **[Networking](../Networking/00_Overview.md)**: TCP/IP 기초, 메시지 전달, 네트워크 장애 모드
- **[System_Design](../System_Design/00_Overview.md)**: 기초 분산 개념 (CAP 정리, leader 선출, Lamport 클럭 — L15-L16에서 소개)
- **[Algorithm](../Algorithm/00_Overview.md)**: 그래프 알고리즘, 복잡도 분석, 증명 기법

## 학습 로드맵

```
Block A: 기초 (L01-04)                Block B: 핵심 프로토콜 (L05-08)
┌─────────────────────────────┐      ┌─────────────────────────────┐
│ L01 시스템 모델과 장애 모드  │      │ L05 Paxos 계열              │
│ L02 시간, 클럭, 순서 결정    │─────▶│ L06 Raft 심층 분석          │
│ L03 FLP 불가능성             │      │ L07 Byzantine 장애 허용     │
│ L04 일관성 모델              │      │ L08 분산 트랜잭션           │
└─────────────────────────────┘      └──────────────┬──────────────┘
                                                    │
                                                    ▼
Block D: 프로덕션 (L13-16)            Block C: 데이터와 일관성 (L09-12)
┌─────────────────────────────┐      ┌─────────────────────────────┐
│ L13 장애 감지                │◀─────│ L09 복제 전략               │
│ L14 조율 프리미티브          │      │ L10 CRDT와 최종 일관성      │
│ L15 TLA+ 검증               │      │ L11 파티셔닝과 샤딩         │
│ L16 캡스톤: 분산 KV 스토어   │      │ L12 스토리지 사례 연구      │
└──────────────┬──────────────┘      └─────────────────────────────┘
               │
               ▼
Block E: 고급 구현 (L18-22)          Block F: 패턴과 테스트 (L23-28)
┌─────────────────────────────┐      ┌─────────────────────────────┐
│ L18 Raft 구현 Part 1        │      │ L23 분산 속도 제한          │
│ L19 Raft 구현 Part 2        │─────▶│ L24 이벤트 소싱과 CQRS     │
│ L20 분산 해시 테이블         │      │ L25 벡터 클럭               │
│ L21 Gossip 프로토콜          │      │ L26 분산 테스트             │
│ L22 서비스 디스커버리        │      │ L27 분산 관측 가능성        │
└─────────────────────────────┘      │ L28 캡스톤: 프로덕션 KV     │
                                     └─────────────────────────────┘
```

## 파일 목록

| 레슨 | 파일명 | 난이도 | 설명 |
|--------|----------|------------|-------------|
| L01 | `01_System_Models_and_Failure_Modes.md` | ⭐⭐ | 동기/비동기/부분 동기 모델, crash와 Byzantine 장애, safety vs liveness |
| L02 | `02_Time_Clocks_and_Ordering.md` | ⭐⭐⭐ | Happens-before, Lamport/vector/hybrid logical 클럭, 인과성 추적 |
| L03 | `03_FLP_Impossibility_and_Bounds.md` | ⭐⭐⭐⭐ | FLP 불가능성 정리, 합의 라운드 하한, 우회 전략 |
| L04 | `04_Consistency_Models.md` | ⭐⭐⭐ | Linearizability, sequential/causal/eventual consistency, PACELC 정리 |
| L05 | `05_Paxos_Family.md` | ⭐⭐⭐⭐ | Single-decree Paxos, Multi-Paxos, FPaxos, EPaxos와 엣지 케이스 분석 |
| L06 | `06_Raft_In_Depth.md` | ⭐⭐⭐ | Pre-vote, 로그 압축, joint consensus 멤버십, ReadIndex/LeaseRead |
| L07 | `07_Byzantine_Fault_Tolerance.md` | ⭐⭐⭐⭐ | PBFT 메시지 흐름, HotStuff linear BFT, Tendermint, 3f+1 vs 2f+1 한계 |
| L08 | `08_Distributed_Transactions.md` | ⭐⭐⭐ | 2PC blocking 분석, 3PC 한계, Percolator, Spanner TrueTime, Calvin, Sagas |
| L09 | `09_Replication_Strategies.md` | ⭐⭐⭐ | 단일/다중/리더리스 복제, chain replication, quorum, read-repair |
| L10 | `10_CRDTs_and_Eventual_Consistency.md` | ⭐⭐⭐ | State/op 기반 CRDT (G-Counter, OR-Set, LWW-Register), Automerge, Yjs |
| L11 | `11_Partitioning_and_Sharding.md` | ⭐⭐⭐ | 해시/범위 파티셔닝, consistent hashing, 보조 인덱스 파티셔닝 |
| L12 | `12_Distributed_Storage_Case_Studies.md` | ⭐⭐⭐⭐ | Spanner, Dynamo, Kafka, CockroachDB 심층 분석 |
| L13 | `13_Failure_Detection_and_Membership.md` | ⭐⭐⭐ | Phi accrual 감지기, SWIM 프로토콜, gossip 기반 멤버십 |
| L14 | `14_Distributed_Coordination_Primitives.md` | ⭐⭐⭐ | 분산 락 (Chubby, Redlock), barrier, fencing token, 서비스 디스커버리 |
| L15 | `15_Formal_Verification_TLAplus.md` | ⭐⭐⭐⭐ | TLA+ 기초, 프로토콜 명세, TLC 모델 검사, AWS 사례 |
| L16 | `16_Capstone_Building_Distributed_KV_Store.md` | ⭐⭐⭐⭐ | Raft 기반 분산 KV 스토어 전체 구축과 장애 주입 테스트 |
| L18 | `18_Raft_Implementation_Part1.md` | ⭐⭐⭐⭐ | Raft 리더 선출, 로그 복제, 안전성 증명, 상태 머신 |
| L19 | `19_Raft_Implementation_Part2.md` | ⭐⭐⭐⭐ | 멤버십 변경, 로그 압축, 스냅샷, 선형화 가능 읽기 |
| L20 | `20_Distributed_Hash_Tables.md` | ⭐⭐⭐ | 일관된 해싱, Chord, Kademlia, 가상 노드, 제한된 부하 |
| L21 | `21_Gossip_Protocols.md` | ⭐⭐⭐ | 전염병 프로토콜, SWIM, push/pull gossip, phi-accrual 장애 감지 |
| L22 | `22_Service_Discovery.md` | ⭐⭐⭐ | Consul, etcd, DNS 기반 디스커버리, 헬스 체크, 로드 밸런싱 |
| L23 | `23_Distributed_Rate_Limiting.md` | ⭐⭐⭐ | 토큰 버킷, 슬라이딩 윈도우, 분산 카운터, Redis 기반 제한 |
| L24 | `24_Event_Sourcing_CQRS.md` | ⭐⭐⭐ | 이벤트 소싱 패턴, CQRS, 이벤트 스토어, 프로젝션, 스냅샷 |
| L25 | `25_Vector_Clocks.md` | ⭐⭐⭐⭐ | 논리 클럭, 벡터 클럭, 버전 벡터, 충돌 해결, HLC |
| L26 | `26_Distributed_Testing.md` | ⭐⭐⭐⭐ | Jepsen, 장애 주입, 카오스 엔지니어링, 결정론적 시뮬레이션 |
| L27 | `27_Distributed_Observability.md` | ⭐⭐⭐ | 분산 트레이싱, 상관관계 ID, 구조화된 로깅, 메트릭 |
| L28 | `28_Capstone_Distributed_KV.md` | ⭐⭐⭐⭐ | 프로덕션 분산 KV 스토어: Raft + 샤딩 + 복제 + 테스트 |

## 난이도 가이드

- ⭐⭐: 선수 지식 기반; 직관적 구현을 동반한 개념적 이해
- ⭐⭐⭐: 실습 구현 능력 필요; 알고리즘적 복잡도 수반
- ⭐⭐⭐⭐: 연구 수준 프로토콜; 형식 증명과 복잡한 다중 컴포넌트 통합

## 환경 설정

```bash
pip install numpy matplotlib
pip install grpcio grpcio-tools    # for RPC examples
pip install asyncio aiohttp        # async networking
pip install sortedcontainers       # for ordered data structures
```

## 관련 토픽

- **[System_Design](../System_Design/00_Overview.md)**: 분산 시스템 개념적 소개 (L15-L16이 선수 과목)
- **[Database_Theory](../Database_Theory/00_Overview.md)**: 스토리지 관점의 분산 데이터베이스 (L14)
- **[Data_Engineering](../Data_Engineering/00_Overview.md)**: 분산 데이터 파이프라인, Kafka, 스트림 처리
- **[DevOps](../DevOps/00_Overview.md)**: 분산 트레이싱과 관측 가능성 (L12)
- **[Security](../Security/00_Overview.md)**: Byzantine 장애 허용은 안전한 분산 프로토콜과 연결

## 학습 팁

1. **알고리즘을 직접 구현하라** — Paxos에 대해 읽는 것만으로는 부족하다; 시뮬레이터를 작성하여 엣지 케이스를 직접 확인하라
2. **시퀀스 다이어그램을 그려라** — 합의 프로토콜은 메시지 흐름을 종이에 추적하면 명확해진다
3. **의도적으로 장애를 주입하라** — 메시지 유실, 응답 지연, 노드 crash를 통해 구현을 테스트하라
4. **원본 논문을 읽어라** — Lamport의 Paxos, Ongaro의 Raft, Castro의 PBFT는 놀라울 정도로 읽기 쉽다
5. **Jepsen 보고서를 활용하라** — Kyle Kingsbury의 분석은 실제 데이터베이스가 일관성 주장을 어떻게 위반하는지 보여준다
6. **점진적으로 구축하라** — 단일 노드 KV 스토어로 시작하여, 복제를 추가하고, 합의를 추가하고, 장애 허용을 추가하라

## 학습 성과

이 토픽을 완료하면 다음을 할 수 있게 됩니다:

- 분산 시스템 모델을 분류하고 safety와 liveness 속성에 대해 추론
- Lamport 클럭, vector 클럭, hybrid logical 클럭 구현
- 핵심 합의 프로토콜(Paxos, Raft)을 엣지 케이스 처리와 함께 설명하고 구현
- 2PC, Percolator, 또는 Saga 패턴을 사용한 분산 트랜잭션 설계
- 충돌 없는 최종 일관성을 위한 CRDT 데이터 구조 구축
- 실제 시스템(Spanner, Dynamo, Kafka)을 분산 시스템 관점에서 분석
- 프로토콜 정확성을 형식적으로 검증하기 위한 TLA+ 명세 작성
- 완전한 Raft 기반 분산 키-값 스토어 구축

## 다음 단계

- **데이터베이스 엔지니어**: [Data_Engineering](../Data_Engineering/00_Overview.md)과 [Database_Theory](../Database_Theory/00_Overview.md)에서 스토리지 특화 패턴 탐구
- **인프라 엔지니어**: [DevOps](../DevOps/00_Overview.md)와 [Cloud_Computing](../Cloud_Computing/00_Overview.md)으로 계속 진행
- **연구자**: [Algorithm](../Algorithm/00_Overview.md)에서 형식 증명 기법 심화 학습

---

**License**: CC BY-NC 4.0

[시작: 레슨 01](./01_System_Models_and_Failure_Modes.md)

# NoSQL 데이터 모델

**이전**: [12. 동시성 제어](./12_Concurrency_Control.md) | **다음**: [14. 분산 데이터베이스](./14_Distributed_Databases.md)

---

관계형 모델은 Codd의 1970년 획기적인 논문 이후 데이터 관리의 지배적인 패러다임으로 자리잡았습니다. 그러나 인터넷이 수천 명의 사용자에서 수십억 명으로 진화하고, 데이터가 메가바이트에서 페타바이트로 증가하면서, 실무자들은 관계형 모델의 엄격한 보장이 자산이 아닌 부채가 되는 시나리오를 발견했습니다. 이 레슨에서는 NoSQL 혁명을 탐구합니다: 왜 발생했는지, 어떤 모델이 등장했는지, 그리고 주어진 문제에 적합한 데이터 모델을 선택하는 방법을 다룹니다.

**난이도**: ⭐⭐⭐

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 관계형 모델이 웹 스케일에서 어려움을 겪는 이유 설명
2. CAP 정리와 형식적 증명 스케치를 진술하고 해석
3. BASE와 ACID 일관성 모델 대조
4. 키-값, 문서, 와이드 컬럼, 그래프 패러다임을 사용하여 데이터 모델 설계
5. 각 NoSQL 패러다임에서 기본 쿼리 작성
6. 적절한 데이터 모델을 선택하기 위한 의사결정 프레임워크 적용
7. 폴리글랏 지속성(Polyglot Persistence)과 그 아키텍처적 함의 이해

---

## 목차

1. [동기: 관계형 모델의 한계](#1-동기-관계형-모델의-한계)
2. [CAP 정리](#2-cap-정리)
3. [BASE vs ACID](#3-base-vs-acid)
4. [키-값 저장소](#4-키-값-저장소)
5. [문서 저장소](#5-문서-저장소)
6. [와이드 컬럼 저장소](#6-와이드-컬럼-저장소)
7. [그래프 데이터베이스](#7-그래프-데이터베이스)
8. [비교 매트릭스: 어떤 모델을 사용할지](#8-비교-매트릭스-어떤-모델을-사용할지)
9. [폴리글랏 지속성](#9-폴리글랏-지속성)
10. [연습문제](#10-연습문제)
11. [참고문헌](#11-참고문헌)

---

## 1. 동기: 관계형 모델의 한계

### 1.1 임피던스 불일치(Impedance Mismatch)

관계형 데이터베이스는 데이터를 평평한 2차원 테이블에 저장합니다. 그러나 현대 애플리케이션은 풍부하고 중첩된 객체를 다룹니다: 전자상거래 애플리케이션의 단일 "주문"에는 품목, 배송 주소, 결제 세부사항, 프로모션 코드가 포함됩니다. 이러한 계층적 객체를 정규화된 테이블에 매핑하려면 읽기 시점에는 복잡한 JOIN 작업이 필요하고 쓰기 시점에는 여러 테이블 INSERT 작업이 필요합니다.

```
애플리케이션 객체              관계형 테이블
┌──────────────────┐            ┌──────────┐   ┌────────────┐
│ Order             │            │ orders   │   │ line_items │
│  ├─ customer      │    ──▶    ├──────────┤   ├────────────┤
│  ├─ items[]       │            │ order_id │──▶│ order_id   │
│  ├─ shipping_addr │            │ cust_id  │   │ product_id │
│  └─ payment       │            │ total    │   │ quantity   │
└──────────────────┘            └──────────┘   └────────────┘
                                      │
                                      ▼
                                ┌──────────────┐   ┌──────────┐
                                │ addresses    │   │ payments │
                                ├──────────────┤   ├──────────┤
                                │ order_id     │   │ order_id │
                                │ street       │   │ method   │
                                │ city         │   │ amount   │
                                └──────────────┘   └──────────┘
```

애플리케이션의 객체 모델과 데이터베이스의 관계형 모델 사이의 이러한 "임피던스 불일치"는 코드 복잡성, 개발 시간, 런타임 성능에서 오버헤드를 발생시킵니다.

### 1.2 확장성 과제

관계형 데이터베이스는 수직 확장(더 큰 기계)을 위해 설계되었습니다. 더 많은 용량이 필요할 때 더 빠른 CPU, 더 많은 RAM, 또는 더 빠른 디스크를 구매합니다. 이 접근법에는 엄격한 한계가 있습니다:

| 확장 차원 | 수직 (Scale Up) | 수평 (Scale Out) |
|-----------|-----------------|------------------|
| **접근법** | 더 큰 기계 | 더 많은 기계 |
| **비용 곡선** | 지수적 | 선형 |
| **이론적 한계** | 하드웨어 상한 | 사실상 무제한 |
| **업그레이드 다운타임** | 일반적으로 필요 | 롤링 업그레이드 |
| **관계형 DB 지원** | 자연스러운 적합 | 매우 어려움 |
| **NoSQL 지원** | 가능 | 자연스러운 적합 |

수평 확장은 여러 노드에 데이터를 분산해야 하며, 이는 근본적으로 여러 관계형 보장과 충돌합니다:

- **파티션 간 JOIN**은 예측 불가능한 지연 시간을 갖는 네트워크 작업이 됩니다.
- **분산 트랜잭션**은 처리량을 감소시키는 복잡한 조정 프로토콜(2PC)이 필요합니다.
- 수백 개의 노드에 걸쳐 수십억 행에 대한 **스키마 변경**(ALTER TABLE)은 운영상 위험합니다.

### 1.3 스키마 경직성

관계형 데이터베이스는 엄격한 스키마를 강제합니다: 테이블의 모든 행은 동일한 컬럼을 갖습니다. 애자일 개발에서 요구사항은 자주 변경됩니다. 각 스키마 변경에는 다음이 필요합니다:

1. 마이그레이션 스크립트 작성
2. 프로덕션 크기 데이터에 대한 마이그레이션 테스트
3. 애플리케이션 코드 변경과의 배포 조정
4. ALTER TABLE 작업 중 잠재적으로 테이블 잠금

빠르게 진화하는 데이터 모델을 가진 애플리케이션(소셜 미디어 피드, IoT 센서 데이터, 콘텐츠 관리)의 경우, 이 경직성은 상당한 운영 오버헤드를 부과합니다.

### 1.4 NoSQL 운동

"NoSQL"이라는 용어는 2009년경에 등장했으며, 처음에는 "Not Only SQL"의 약자로 이러한 시스템이 관계형 데이터베이스를 대체하기보다는 보완함을 강조했습니다. 주요 동기는 다음과 같았습니다:

- **수평 확장성**: 상용 하드웨어에 데이터 분산
- **유연한 스키마**: 마이그레이션 없이 변화하는 데이터 구조에 적응
- **높은 가용성**: 노드 장애 시에도 운영 유지
- **성능**: 범용 쿼리가 아닌 특정 액세스 패턴에 최적화
- **개발자 생산성**: 애플리케이션 객체와 일치하는 형식으로 데이터 저장

> **역사적 참고**: Google의 Bigtable 논문(2006)과 Amazon의 Dynamo 논문(2007)은 NoSQL 운동의 기초 논문으로 간주됩니다. 이들은 관계형 보장을 완화하면 이전에는 불가능했던 성능과 확장성을 달성할 수 있음을 입증했습니다.

---

## 2. CAP 정리

### 2.1 진술

Eric Brewer가 2000년에 공식화하고 Gilbert와 Lynch가 2002년에 형식적으로 증명한 CAP 정리는 다음을 명시합니다:

> **CAP 정리**: 분산 데이터 저장소에서 다음 세 가지 속성을 모두 동시에 보장하는 것은 불가능합니다:
> - **일관성(Consistency, C)**: 모든 읽기는 가장 최근의 쓰기 또는 오류를 받습니다.
> - **가용성(Availability, A)**: 모든 요청은 가장 최근의 쓰기를 포함한다는 보장 없이 (오류가 아닌) 응답을 받습니다.
> - **파티션 허용성(Partition Tolerance, P)**: 노드 간 네트워크에서 임의 수의 메시지가 손실되거나 지연되더라도 시스템이 계속 작동합니다.

```
                    ┌─────────────────┐
                    │   Consistency   │
                    │       (C)       │
                    └────────┬────────┘
                             │
                   CA        │        CP
               (분산         │    (가용성
               시스템에서    │    희생)
               불가능)       │
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────┴─────┐                            ┌─────┴─────┐
   │Availability│                           │ Partition │
   │    (A)    │                            │ Tolerance │
   │           │                            │    (P)    │
   └────┬──────┘                            └─────┬─────┘
        │                                         │
        └──────────────── AP ─────────────────────┘
                    (일관성
                     희생)
```

### 2.2 각 속성 이해

**일관성(Consistency)** (선형화 가능성): 클라이언트 A가 키 `k`에 값 `v`를 쓰면, 이후 모든 클라이언트에 의한 `k`의 읽기는 `v`(또는 더 최근의 값)를 반환해야 합니다. 이것이 선형화 가능 일관성 -- 가장 강력한 형태입니다. 시스템이 마치 데이터의 단일 복사본이 있는 것처럼 동작함을 의미합니다.

**가용성(Availability)**: 장애가 없는 노드가 받은 모든 요청은 응답을 발생시켜야 합니다. 시스템은 상태에 대해 확실하지 않을 때 요청을 무시하거나 오류를 반환할 수 없습니다.

**파티션 허용성(Partition Tolerance)**: 네트워크 파티션은 노드 간 메시지가 손실되거나 임의로 지연될 때 발생합니다. 모든 실제 분산 시스템에서 네트워크 파티션은 불가피합니다(케이블이 끊어지고, 스위치가 고장나고, 데이터센터가 연결성을 잃습니다). 따라서 P는 실제로 선택 사항이 아닙니다.

### 2.3 증명 스케치

Gilbert와 Lynch(2002)의 증명은 모순을 통해 진행됩니다:

**설정**: 네트워크로 연결된 가장 간단한 분산 시스템, 두 개의 노드 N1과 N2를 고려합니다. 두 노드 모두 변수 `v`의 복사본을 저장하며, 초기값은 `v0`입니다.

**가정**: 시스템이 세 가지 속성 모두를 보장합니다: C, A, P.

**단계 1**: N1과 N2 사이에 네트워크 파티션이 발생합니다. N1에서 N2로(그 반대도) 보낸 메시지가 손실됩니다. P에 의해 시스템은 계속 작동해야 합니다.

**단계 2**: 클라이언트가 N1에 쓰기 요청을 보내 `v = v1`로 설정합니다. A에 의해 N1은 이 쓰기를 수락하고 클라이언트에 응답해야 합니다. 그러나 파티션 때문에 N1은 이 업데이트를 N2에 전파할 수 없습니다.

**단계 3**: 다른 클라이언트가 `v`의 값을 위해 N2에 읽기 요청을 보냅니다. A에 의해 N2는 응답해야 합니다. N2가 가진 유일한 값은 `v0`(이전 값)입니다. N1의 업데이트가 파티션에서 손실되었기 때문입니다.

**단계 4**: N2는 `v0`를 반환하지만, 가장 최근의 쓰기는 `v1`입니다. 이는 C(선형화 가능성)를 위반합니다.

**모순**: C, A, P를 가정했지만 C의 위반을 도출했습니다. 따라서 세 가지를 동시에 보장하는 것은 불가능합니다.

```
    N1                          N2
    ┌──┐                       ┌──┐
    │v0│                       │v0│    초기 상태
    └──┘                       └──┘
     │                          │
     │    ╳╳╳ PARTITION ╳╳╳    │     네트워크 파티션
     │                          │
   write(v1)                    │
     │                          │
    ┌──┐                       ┌──┐
    │v1│    복제 불가           │v0│    N1 업데이트, N2 오래됨
    └──┘    ──────╳──────▶     └──┘
                                │
                             read(v) → v0 반환  ← C 위반
```

### 2.4 실제 CAP

네트워크 파티션이 분산 시스템에서 불가피하기 때문에, 실제 선택은 다음 사이입니다:

- **CP 시스템**: 파티션 중 가용성을 희생합니다. 파티션이 발생하면 시스템은 일관성을 유지하기 위해 일부 요청을 거부할 수 있습니다. 예: HBase, MongoDB(다수 읽기 관심 수준), etcd, ZooKeeper.

- **AP 시스템**: 파티션 중 일관성을 희생합니다. 파티션이 발생하면 시스템은 계속 요청을 처리하지만 오래된 데이터를 반환할 수 있습니다. 예: Cassandra, DynamoDB, CouchDB, Riak.

**중요한 뉘앙스**:

1. **CAP는 파티션 중의 동작에만 관한 것입니다.** 파티션이 없을 때 시스템은 C와 A를 모두 제공할 수 있습니다.

2. **일관성은 스펙트럼입니다.** 엄격한 선형화 가능성과 최종 일관성 사이에는 많은 중간 수준이 있습니다(인과 일관성, read-your-writes, 단조 읽기).

3. **선택은 작업별이지 시스템별이 아닙니다.** 단일 데이터베이스가 다른 작업에 대해 다른 일관성 보장을 제공할 수 있습니다. 예를 들어, MongoDB는 쿼리별로 `readConcern: "majority"`(CP) 또는 `readConcern: "local"`(AP)를 선택할 수 있습니다.

### 2.5 PACELC 확장

Daniel Abadi는 PACELC 정리를 개선안으로 제안했습니다:

> **PACELC**: **P**artition이 있으면 **A**vailability와 **C**onsistency 중 선택; **E**lse(시스템이 정상 작동 중일 때) **L**atency와 **C**onsistency 중 선택.

| 시스템 | P → A or C? | E → L or C? | 분류 |
|--------|-------------|-------------|------|
| DynamoDB | A | L | PA/EL |
| Cassandra | A | L | PA/EL |
| MongoDB | C | C | PC/EC |
| HBase | C | C | PC/EC |
| PNUTS (Yahoo) | A | C | PA/EC |
| Spanner | C | C | PC/EC |

이는 파티션이 없을 때도 시스템이 지연 시간-일관성 트레이드오프를 만든다는 관찰을 포착합니다.

---

## 3. BASE vs ACID

[레슨 11](./11_Transaction_Theory.md)에서 ACID를 자세히 다뤘습니다. 이제 대부분의 NoSQL 시스템이 채택하는 BASE 모델과 대조해 봅시다.

### 3.1 ACID 요약

| 속성 | 의미 |
|------|------|
| **원자성(Atomicity)** | 트랜잭션의 모든 작업이 성공하거나 모두 실패 |
| **일관성(Consistency)** | 트랜잭션이 데이터베이스를 하나의 유효한 상태에서 다른 상태로 이동 |
| **격리성(Isolation)** | 동시 트랜잭션은 서로 간섭하지 않음 |
| **지속성(Durability)** | 커밋되면 데이터는 시스템 장애에도 살아남음 |

### 3.2 BASE 속성

BASE는 Eric Brewer가 ACID의 대립으로 제안한 역약어(backronym)입니다:

| 속성 | 의미 |
|------|------|
| **기본적으로 가용(Basically Available)** | 시스템은 가용성(CAP 의미에서)을 보장 |
| **소프트 상태(Soft state)** | 최종 일관성 전파로 인해 입력 없이도 시스템의 상태가 시간이 지나면서 변할 수 있음 |
| **최종적으로 일관됨(Eventually consistent)** | 새로운 업데이트 없이 충분한 시간이 주어지면, 모든 복제본은 동일한 상태로 수렴 |

> **비유 -- 마을에 뉴스가 퍼지는 것**:
> 인터넷이나 TV가 없는 작은 마을을 상상해 보세요. 뉴스는 오직 입소문으로만 퍼집니다. 어떤 일이 발생하면(새 가게가 열리면), 근처 사람들이 먼저 알게 됩니다. 다음 몇 시간 동안 그들이 이웃에게 말하고, 이웃이 또 이웃에게 전합니다. 어느 시점에서는 일부 주민만 뉴스를 알고 다른 주민은 모르는 상태 -- 마을은 "소프트 상태(soft state)"에 있습니다. 그러나 새로운 사건 없이 충분한 시간이 주어지면, *모든 사람이* 결국 같은 이야기를 듣게 됩니다. 이것이 최종 일관성(eventual consistency)입니다: 모든 사람을 동시에 업데이트하는 단일 방송은 없지만(ACID의 "전부 아니면 전무" 커밋과 달리), 시스템은 시간이 지남에 따라 일관된 상태로 수렴합니다. 트레이드오프는 같은 순간에 두 주민에게 물으면 서로 다른 답을 얻을 수 있다는 것입니다 -- 소셜 미디어 "좋아요" 카운터에는 허용 가능하지만 은행 잔고에는 허용할 수 없습니다.

### 3.3 상세 비교

```
          ACID                                    BASE
┌──────────────────────┐            ┌──────────────────────┐
│ 강한 일관성           │            │ 최종 일관성           │
│ 비관적 잠금           │            │ 낙관적 복제           │
│ 중앙집중식            │            │ 분산                  │
│ 스키마 우선           │            │ 스키마 유연           │
│ Scale up              │            │ Scale out             │
│ 복잡한 쿼리 지원      │            │ 단순 쿼리 패턴        │
│ 낮은 가용성           │            │ 높은 가용성           │
│ 높은 지연(2PC)        │            │ 낮은 지연             │
└──────────────────────┘            └──────────────────────┘
```

| 차원 | ACID | BASE |
|------|------|------|
| **목표** | 무엇보다 정확성 | 무엇보다 가용성 |
| **읽기** | 항상 최신 쓰기 확인 | 오래된 데이터를 볼 수 있음 |
| **충돌 해결** | 충돌 방지(잠금) | 충돌 감지 및 해결(CRDT, LWW) |
| **애플리케이션 복잡성** | 단순(DB가 일관성 처리) | 복잡(앱이 충돌 처리) |
| **사용 사례** | 은행, 재고, 예약 | 소셜 피드, 분석, 캐싱 |

### 3.4 최종 일관성 모델

"최종적으로 일관됨"은 단일 모델이 아니라 일관성 보장의 집합입니다:

| 모델 | 보장 |
|------|------|
| **강한 최종 일관성(Strong eventual consistency)** | 동일한 업데이트 집합을 받은 복제본은 동일한 상태(충돌 없음) |
| **인과 일관성(Causal consistency)** | 작업 A가 인과적으로 B에 선행하면, 모든 노드는 A를 B 전에 봄 |
| **Read-your-writes** | 프로세스는 항상 자신의 쓰기를 봄 |
| **단조 읽기(Monotonic reads)** | 프로세스가 값 v를 읽으면, 후속 읽기는 v보다 오래된 값을 반환하지 않음 |
| **단조 쓰기(Monotonic writes)** | 프로세스의 쓰기는 모든 복제본에서 순서대로 적용 |
| **세션 일관성(Session consistency)** | 세션 내에서 read-your-writes + 단조 읽기 |

---

## 4. 키-값 저장소

### 4.1 데이터 모델

키-값 저장소는 가장 간단한 NoSQL 데이터 모델입니다. 본질적으로 분산 해시 테이블입니다:

```
┌─────────────────────────────────────────────────┐
│                  키-값 저장소                    │
│                                                 │
│   키 (문자열)          값 (불투명 블롭)          │
│   ─────────────         ───────────────────     │
│   "user:1001"     →     {"name": "Alice", ...}  │
│   "session:abc"   →     {token_data...}         │
│   "cache:page:42" →     "<html>...</html>"      │
│   "counter:views" →     "1547382"               │
│                                                 │
└─────────────────────────────────────────────────┘
```

**특성**:
- **키**는 고유한 문자열(또는 바이트 배열)
- **값**은 불투명 블롭 -- 저장소는 내용을 해석하지 않음
- **연산**은 단순: GET, PUT, DELETE (그리고 때때로 원자적 증가, TTL)
- **보조 인덱스 없음** (별도 기능으로 추가되지 않는 한)
- **JOIN, 집계, 복잡한 쿼리 없음**

### 4.2 연산

API는 의도적으로 최소화되었습니다:

```
# 기본 연산
PUT(key, value)         # 키-값 쌍 저장
GET(key) → value        # 키로 값 검색
DELETE(key)             # 키-값 쌍 제거

# 확장 연산 (벤더별)
EXISTS(key) → bool      # 키 존재 확인
EXPIRE(key, ttl)        # 생존 시간 설정
INCR(key)               # 원자적 증가
MGET(key1, key2, ...)   # 일괄 검색
```

### 4.3 Redis

Redis(Remote Dictionary Server)는 풍부한 데이터 구조를 지원하는 인메모리 키-값 저장소입니다.

**Redis의 데이터 구조**:

```
┌───────────────────────────────────────────────────────┐
│  Redis 데이터 타입                                     │
│                                                       │
│  STRING:  "hello"                                     │
│  LIST:    [a, b, c, d]                                │
│  SET:     {a, b, c}                                   │
│  SORTED SET: {(a,1.0), (b,2.5), (c,3.7)}            │
│  HASH:    {field1: val1, field2: val2}                │
│  BITMAP:  01101001...                                 │
│  STREAM:  [(id1, {k:v}), (id2, {k:v}), ...]         │
│  GEOSPATIAL: {(member, lat, lon), ...}                │
└───────────────────────────────────────────────────────┘
```

**Redis 세션 예제**:

```redis
# 문자열
SET user:1001:name "Alice"
GET user:1001:name                    # → "Alice"

# 해시 (미니 문서처럼)
HSET user:1001 name "Alice" email "alice@example.com" age "30"
HGET user:1001 name                   # → "Alice"
HGETALL user:1001                     # → name, Alice, email, alice@example.com, age, 30

# 리스트 (메시지 큐 패턴)
LPUSH notifications:1001 "New order #42"
LPUSH notifications:1001 "Payment received"
LRANGE notifications:1001 0 -1       # → ["Payment received", "New order #42"]

# 정렬된 집합 (리더보드)
ZADD leaderboard 1500 "player:alice"
ZADD leaderboard 2300 "player:bob"
ZADD leaderboard 1800 "player:charlie"
ZREVRANGE leaderboard 0 2 WITHSCORES # → bob:2300, charlie:1800, alice:1500

# 원자적 증가 (페이지 뷰 카운터)
INCR page:views:homepage              # → 1
INCR page:views:homepage              # → 2

# TTL (세션 관리)
SET session:abc123 "{user_id: 1001}" EX 3600   # 1시간 후 만료
TTL session:abc123                              # → 3598 (남은 초)
```

**Redis 사용 사례**:
- **캐싱**: TTL로 자주 액세스되는 데이터 저장
- **세션 저장**: 자동 만료되는 사용자 세션
- **속도 제한**: 슬라이딩 윈도우를 위해 EXPIRE와 함께 INCR 사용
- **실시간 리더보드**: 순위를 위한 정렬된 집합
- **Pub/Sub 메시징**: 경량 메시지 브로커
- **분산 잠금**: 상호 배제를 위한 SETNX(SET if Not eXists)

### 4.4 Amazon DynamoDB

DynamoDB는 AWS의 완전 관리형 서버리스 키-값 및 문서 데이터베이스입니다.

**핵심 개념**:

```
┌─────────────────────────────────────────────────────────┐
│  DynamoDB 테이블                                         │
│                                                         │
│  파티션 키 (PK)  │  정렬 키 (SK)  │  속성              │
│  ────────────────┼─────────────────┼────────────        │
│  "USER#1001"     │  "PROFILE"      │  {name, email}     │
│  "USER#1001"     │  "ORDER#001"    │  {total, date}     │
│  "USER#1001"     │  "ORDER#002"    │  {total, date}     │
│  "USER#1002"     │  "PROFILE"      │  {name, email}     │
│                                                         │
│  PK → 파티션 결정 (물리적 위치)                         │
│  PK + SK → 항목을 고유하게 식별                         │
└─────────────────────────────────────────────────────────┘
```

**단일 테이블 설계**: 데이터를 많은 테이블로 정규화하는 관계형 데이터베이스와 달리, DynamoDB 모범 사례는 모든 엔티티를 단일 테이블에 넣고 복합 키를 사용하여 관계를 모델링하는 것을 권장합니다:

```
PK              SK                  속성
───────────     ───────────────     ─────────────────────
USER#1001       PROFILE             name=Alice, email=...
USER#1001       ORDER#2024-001      total=99.99, status=shipped
USER#1001       ORDER#2024-002      total=45.50, status=pending
PRODUCT#ABC     METADATA            name=Widget, price=9.99
PRODUCT#ABC     REVIEW#1001         rating=5, text="Great!"
ORDER#2024-001  ITEM#ABC            qty=2, price=9.99
```

**액세스 패턴**이 설계를 주도합니다:
- 사용자 프로필 가져오기: `PK = "USER#1001", SK = "PROFILE"`
- 사용자 주문 목록: `PK = "USER#1001", SK begins_with "ORDER#"`
- 주문 항목 가져오기: `PK = "ORDER#2024-001", SK begins_with "ITEM#"`

### 4.5 키-값 저장소의 한계

- **애드혹 쿼리 없음**: 키로만 조회 가능; "도시 X의 모든 사용자 찾기" 같은 것은 불가능
- **관계 없음**: JOIN 없음, 외래 키 없음
- **애플리케이션 수준 일관성**: 애플리케이션이 키 간 데이터 일관성을 관리해야 함
- **데이터 모델링 복잡성**: 비정규화와 단일 테이블 설계는 신중한 사전 계획 필요

---

## 5. 문서 저장소

### 5.1 데이터 모델

문서 저장소는 값을 구조화되고 쿼리 가능한 문서(일반적으로 JSON 또는 BSON)로 만들어 키-값 모델을 확장합니다:

```json
{
  "_id": "order_1001",
  "customer": {
    "name": "Alice Chen",
    "email": "alice@example.com",
    "address": {
      "street": "123 Main St",
      "city": "San Francisco",
      "state": "CA",
      "zip": "94105"
    }
  },
  "items": [
    {"product_id": "P001", "name": "Widget", "qty": 2, "price": 9.99},
    {"product_id": "P002", "name": "Gadget", "qty": 1, "price": 24.99}
  ],
  "total": 44.97,
  "status": "shipped",
  "created_at": "2024-11-15T10:30:00Z",
  "tags": ["electronics", "priority"]
}
```

**주요 특성**:
- **자체 설명**: 각 문서가 자체 구조를 포함
- **중첩 구조**: 객체 내 객체, 객체 배열
- **유연한 스키마**: 동일 컬렉션의 다른 문서가 다른 필드를 가질 수 있음
- **풍부한 쿼리**: 키-값 저장소와 달리 중첩 필드를 포함한 모든 필드로 쿼리 가능
- **인덱스**: 중첩 객체 내 필드를 포함한 모든 필드에 보조 인덱스

### 5.2 JSON vs BSON

| 기능 | JSON | BSON |
|------|------|------|
| **형식** | 텍스트 기반 | 바이너리 |
| **가독성** | 사람이 읽을 수 있음 | 기계 최적화 |
| **데이터 타입** | String, Number, Boolean, null, Object, Array | Date, ObjectId, Decimal128, Binary 포함 20+ 타입 |
| **크기** | 텍스트에 컴팩트 | 약간 큰 편(타입 접두사) |
| **파싱** | 느림(텍스트 파싱) | 빠름(직접 메모리 매핑) |
| **사용처** | CouchDB, Elasticsearch | MongoDB |

### 5.3 MongoDB

MongoDB는 BSON을 저장 형식으로 사용하는 가장 인기 있는 문서 데이터베이스입니다.

**핵심 개념**:

```
관계형            MongoDB
─────────         ─────────
Database    →     Database
Table       →     Collection
Row         →     Document
Column      →     Field
JOIN        →     Embedding / $lookup
Primary Key →     _id field
Index       →     Index
```

**CRUD 연산**:

```javascript
// INSERT — 고객 데이터가 주문 문서 안에 직접 임베딩됨.
// 이것은 의도적인 비정규화(denormalization): 고객의 이름과 이메일을 복제하여
// 주문을 표시할 때 두 번째 쿼리(JOIN에 해당)가 필요 없게 만듦.
// 트레이드오프: Alice가 이메일을 변경하면 모든 주문 문서를 업데이트해야 함.
db.orders.insertOne({
  customer: { name: "Alice", email: "alice@example.com" },
  items: [
    { product: "Widget", qty: 2, price: 9.99 },
    { product: "Gadget", qty: 1, price: 24.99 }
  ],
  total: 44.97,
  status: "pending",
  created_at: new Date()
});

// FIND (쿼리)
// 점 표기법(dot notation) "customer.name"은 임베딩된 문서 안으로 접근 —
// MongoDB가 중첩 필드(nested field)를 네이티브로 인덱싱하기 때문에 가능.
// 관계형 DB에서는 orders와 customers 테이블 사이의 JOIN이 필요할 것임.
db.orders.find({
  "customer.name": "Alice",
  "status": "pending"
});

// total > 100인 주문 찾기, 날짜순 정렬.
// sort + limit 패턴은 {total: 1, created_at: -1}에 복합 인덱스가 있을 때
// 효율적 — MongoDB가 인덱스를 사용하여 메모리 내 정렬을 피함.
db.orders.find({ total: { $gt: 100 } })
         .sort({ created_at: -1 })
         .limit(10);

// 배열 내부 쿼리: "items.product"는 items 배열의 모든 요소에서 매칭.
// MongoDB는 배열 필드에 대해 자동으로 "멀티키 인덱스(multikey index)"를
// 생성하므로, items가 배열임에도 인덱스를 사용할 수 있음.
db.orders.find({ "items.product": "Widget" });

// UPDATE
// $set은 지정된 필드만 수정; 다른 필드는 그대로 유지.
// $currentDate는 자동으로 타임스탬프를 설정 — 감사 추적(audit trail)에 유용.
db.orders.updateOne(
  { _id: ObjectId("...") },
  {
    $set: { status: "shipped" },
    $currentDate: { updated_at: true }
  }
);

// $push는 배열에 제자리(in-place)에서 추가 — 전체 문서를 읽기-수정-쓰기할 필요 없음.
// 이것은 문서 수준에서 원자적(atomic) (MongoDB는 단일 문서 원자성을 보장).
db.orders.updateOne(
  { _id: ObjectId("...") },
  { $push: { items: { product: "Doohickey", qty: 3, price: 5.99 } } }
);

// DELETE — deleteMany는 한 번의 작업으로 모든 매칭 문서를 제거.
// 관계형 FK처럼 연쇄 삭제(cascading delete)가 없으므로 관련 정리는 앱이 처리해야 함.
db.orders.deleteMany({ status: "cancelled" });
```

**집계 파이프라인**: MongoDB의 복잡한 데이터 처리 프레임워크:

```javascript
// 최근 30일간 제품 카테고리별 수익.
// 집계 파이프라인은 UNIX 파이프처럼 순차적 단계를 통해 문서를 처리:
// 각 단계가 데이터를 변환하고 다음 단계로 전달.
db.orders.aggregate([
  // 단계 1: 초기에 필터링 — 파이프라인 상단의 $match는 인덱스를 사용하며
  // 후속 모든 단계의 데이터 볼륨을 줄임 (SQL의 WHERE와 유사).
  { $match: {
    created_at: { $gte: new Date(Date.now() - 30*24*60*60*1000) },
    status: { $ne: "cancelled" }
  }},
  // 단계 2: $unwind는 items 배열을 "펼침" — 각 배열 요소가 자체 문서가 됨.
  // 개별 제품별로 그룹화해야 하지만 items가 주문 문서 안에 임베딩되어 있기
  // 때문에(비정규화) 이 작업이 필요.
  { $unwind: "$items" },
  // 단계 3: 제품별로 그룹화하고 집계값 계산.
  // items가 임베딩되어 있기(별도 컬렉션이 아닌) 때문에 JOIN 없이
  // 계산 가능 — 비정규화(denormalization)의 트레이드오프가 여기서 보상됨.
  { $group: {
    _id: "$items.product",
    total_revenue: { $sum: { $multiply: ["$items.qty", "$items.price"] } },
    total_units: { $sum: "$items.qty" },
    order_count: { $sum: 1 }
  }},
  // 단계 4: 수익 내림차순 정렬
  { $sort: { total_revenue: -1 } },
  // 단계 5: 상위 10개로 제한
  { $limit: 10 }
]);
```

### 5.4 스키마 설계 패턴

문서 데이터베이스에서 주요 설계 결정은 **임베딩 vs 참조**입니다:

**임베딩** (비정규화):

```json
// 주문을 사용자 문서 안에 임베딩 — 한 번의 읽기로 사용자와 모든 주문을 가져옴.
// JOIN에 해당하는 작업을 피함.
// 트레이드오프: 주문이 추가될 때마다 문서가 커지며, MongoDB의 16MB 문서
// 크기 제한이 있음. 임베딩 항목 수가 제한적일 때(one-to-few) 적합.
{
  "_id": "user_1001",
  "name": "Alice",
  "orders": [
    { "order_id": "O001", "total": 44.97, "items": ["..."] },
    { "order_id": "O002", "total": 89.50, "items": ["..."] }
  ]
}
```

**참조** (정규화):

```json
// Users 컬렉션 — 사용자 문서는 작고 안정적으로 유지됨.
{ "_id": "user_1001", "name": "Alice" }

// Orders 컬렉션 — 주문이 외래 키(foreign key)처럼 ID로 사용자를 참조.
// 사용자의 주문을 가져오려면 두 번째 쿼리(또는 $lookup)가 필요.
// 사용자당 주문 수가 무제한일 때 이 방식이 더 적합.
{ "_id": "O001", "user_id": "user_1001", "total": 44.97, "items": ["..."] }
{ "_id": "O002", "user_id": "user_1001", "total": 89.50, "items": ["..."] }
```

**결정 기준**:

| 요인 | 임베드 | 참조 |
|------|--------|------|
| 데이터가 함께 읽히는가? | 예 → 임베드 | 아니오 → 참조 |
| 데이터 크기가 무제한인가? | 아니오 → 임베드 | 예 → 참조 (16MB 문서 제한) |
| 데이터가 독립적으로 업데이트되는가? | 아니오 → 임베드 | 예 → 참조 |
| 카디널리티 | One-to-few | One-to-many / Many-to-many |

**일반적인 패턴**:

| 패턴 | 설명 | 예 |
|------|------|-----|
| **Subset** | 가장 많이 사용되는 필드를 임베드, 나머지는 참조 | 제품 요약은 임베드, 전체 스펙은 참조 |
| **Extended Reference** | 참조된 문서에서 자주 액세스되는 필드의 복사본 저장 | 주문이 고객 이름 + 이메일 저장(전체 프로필 아님) |
| **Computed** | 미리 계산하고 파생 값 저장 | 매번 항목에서 계산하지 않고 주문 총액 저장 |
| **Bucket** | 시계열 데이터를 버킷으로 그룹화 | 시간당 센서 판독값을 하나의 문서로 |
| **Outlier** | 임베딩 제한을 초과하는 문서를 다르게 처리 | "has_overflow" 플래그 지정하고 초과분을 오버플로 컬렉션에 저장 |

### 5.5 MongoDB의 인덱싱

```javascript
// 단일 필드 인덱스
db.orders.createIndex({ status: 1 });

// 복합 인덱스
db.orders.createIndex({ "customer.email": 1, created_at: -1 });

// 멀티키 인덱스 (배열 필드에)
db.orders.createIndex({ "items.product": 1 });

// 텍스트 인덱스 (전문 검색)
db.orders.createIndex({ "items.product": "text", notes: "text" });

// TTL 인덱스 (만료 후 자동 삭제)
db.sessions.createIndex({ created_at: 1 }, { expireAfterSeconds: 3600 });

// 고유 인덱스
db.users.createIndex({ email: 1 }, { unique: true });
```

### 5.6 문서 저장소의 한계

- **다중 문서 ACID 트랜잭션 없음** (역사적으로; MongoDB는 v4.0에서 추가했지만 성능 오버헤드 있음)
- **데이터 중복**: 비정규화는 업데이트가 많은 문서를 터치해야 할 수 있음을 의미
- **무제한 문서 증가**: 제한 없이 증가하는 문서는 성능을 저하시킴
- **내장 참조 무결성 없음**: 애플리케이션이 외래 키 같은 제약 조건을 강제해야 함

---

## 6. 와이드 컬럼 저장소

### 6.1 데이터 모델

와이드 컬럼 저장소(컬럼 패밀리 저장소라고도 함)는 행이 아닌 컬럼으로 데이터를 구성합니다. 컬럼형 분석 데이터베이스(Parquet나 ClickHouse 같은)와 혼동해서는 안 됩니다; 와이드 컬럼 저장소는 운영 워크로드를 위해 설계되었습니다.

```
┌────────────────────────────────────────────────────────────────┐
│  와이드 컬럼 저장소                                             │
│                                                                │
│  행 키        │ 컬럼 패밀리: "profile"  │ CF: "activity"       │
│  ─────────────┼───────────────────────────┼────────────────     │
│  user:1001    │ name: Alice               │ last_login: ...     │
│               │ email: alice@ex.com       │ page_views: 1547    │
│               │ city: San Francisco       │                     │
│  ─────────────┼───────────────────────────┼────────────────     │
│  user:1002    │ name: Bob                 │ last_login: ...     │
│               │ email: bob@ex.com         │ page_views: 892     │
│               │ phone: +1-555-0123        │ last_purchase: ...  │
│               │                           │                     │
│  참고: user:1001에는 "phone" 컬럼이 없음,                       │
│        user:1002에는 추가 "last_purchase" 컬럼이 있음.          │
│        각 행은 다른 컬럼을 가질 수 있음!                        │
└────────────────────────────────────────────────────────────────┘
```

**주요 개념**:
- **행 키(Row Key)**: 기본 식별자, 노드 간 데이터 분산 결정
- **컬럼 패밀리(Column Family)**: 관련 컬럼 그룹, 테이블 생성 시 정의
- **컬럼(Column)**: 컬럼 패밀리 내의 이름-값 쌍
- **희소 저장**: 행이 동일한 컬럼을 가질 필요 없음; 빈 컬럼은 저장 공간을 소비하지 않음
- **타임스탬프**: 각 셀(행 키 + 컬럼)은 여러 타임스탬프가 찍힌 버전을 저장 가능

### 6.2 Cassandra

Apache Cassandra는 단일 장애 지점이 없는 고가용성을 위해 설계된 분산 와이드 컬럼 저장소입니다.

**아키텍처**:

```
                  ┌─────────────────────────────┐
                  │      Cassandra Ring          │
                  │                              │
                  │     N1 ─── N2                │
                  │    /         \               │
                  │  N6           N3             │
                  │    \         /               │
                  │     N5 ─── N4                │
                  │                              │
                  │  모든 노드가 동등            │
                  │  (master/slave 없음)        │
                  │  일관된 해싱으로             │
                  │  데이터 분산                 │
                  └─────────────────────────────┘
```

**CQL (Cassandra Query Language)**:

```sql
-- 키스페이스 생성 (데이터베이스와 유사).
-- NetworkTopologyStrategy에서 데이터센터당 3개의 복제본을 설정하면
-- DC에서 노드 2개가 장애가 나도 모든 파티션의 복사본 1개는 생존.
CREATE KEYSPACE ecommerce
WITH replication = {
  'class': 'NetworkTopologyStrategy',
  'dc1': 3, 'dc2': 3
};

-- 복합 기본 키로 테이블 생성.
-- 파티션 키(partition key) 선택은 Cassandra에서 가장 중요한 설계 결정:
-- 데이터 분산과 쿼리 효율성 모두를 결정함.
CREATE TABLE ecommerce.orders (
  customer_id UUID,
  order_date TIMESTAMP,
  order_id UUID,
  total DECIMAL,
  status TEXT,
  items LIST<FROZEN<item_type>>,
  PRIMARY KEY ((customer_id), order_date, order_id)
) WITH CLUSTERING ORDER BY (order_date DESC);

-- PRIMARY KEY 구조 분석:
--   파티션 키: (customer_id) — Cassandra가 이 값을 해싱하여 어느 노드가
--     데이터를 저장할지 결정. 한 고객의 모든 주문이 동일한 노드에 위치하므로
--     "고객 X의 모든 주문 가져오기"가 단일 노드 읽기(빠름)가 됨.
--   클러스터링 키: order_date, order_id — 파티션 내에서 행이 이 컬럼 순서대로
--     디스크에 정렬 저장됨. DESC 순서는 가장 최근 주문이 물리적으로 먼저
--     위치하므로 "최근 N개 주문"이 순차 디스크 읽기가 됨.

-- 데이터 삽입
INSERT INTO ecommerce.orders (customer_id, order_date, order_id, total, status)
VALUES (uuid(), '2024-11-15', uuid(), 44.97, 'shipped');

-- 파티션 키로 쿼리 (빠름 — 코디네이터가 customer_id를 해싱하여 정확한
-- 노드를 찾음; 다른 노드는 접촉하지 않음). O(1) 노드 조회.
SELECT * FROM ecommerce.orders
WHERE customer_id = 550e8400-e29b-41d4-a716-446655440000;

-- 클러스터링 키 범위로 쿼리 (빠름 — order_date가 클러스터링 키이므로
-- 이 파티션 내의 행들이 디스크에서 날짜순으로 정렬되어 있음. Cassandra는
-- 연속된 바이트 범위를 읽음 — 랜덤 탐색이 아닌 순차 스캔).
SELECT * FROM ecommerce.orders
WHERE customer_id = 550e8400-e29b-41d4-a716-446655440000
  AND order_date >= '2024-01-01'
  AND order_date < '2025-01-01';

-- 파티션 키 없이 쿼리 (느림 — Cassandra는 'pending' 주문이 어느 노드에
-- 있는지 알 수 없으므로 모든 노드에 브로드캐스트하여 모든 파티션을 스캔해야 함.
-- 이 전체 클러스터 스캔을 인정하기 위해 ALLOW FILTERING이 필요함).
-- 안티패턴: 프로덕션에서 피하세요! 이 쿼리가 필요하면
-- status를 파티션 키로 하는 별도 테이블을 생성하세요.
SELECT * FROM ecommerce.orders WHERE status = 'pending' ALLOW FILTERING;
```

**데이터 모델링 원칙**: Cassandra에서는 엔티티가 아닌 쿼리를 중심으로 테이블을 모델링합니다. N개의 다른 쿼리가 있다면 N개의 다른 테이블이 필요할 수 있습니다(각각 하나의 쿼리에 최적화).

### 6.3 HBase

Apache HBase는 HDFS(Hadoop Distributed File System) 위에 구축된 와이드 컬럼 저장소로, Google의 Bigtable에서 영감을 받았습니다.

**Cassandra와의 주요 차이점**:

| 기능 | Cassandra | HBase |
|------|-----------|-------|
| **아키텍처** | Peer-to-peer (마스터 없음) | Master-RegionServer |
| **일관성** | 조정 가능 (기본 AP) | 강함 (CP) |
| **쓰기 경로** | Log-structured merge tree | Log-structured merge tree |
| **저장소** | 자체 저장 엔진 | HDFS |
| **쿼리 언어** | CQL (SQL-like) | Java API / HBase Shell |
| **최적 용도** | 쓰기 중심, 다중 데이터센터 | HDFS 데이터에 대한 무작위 읽기/쓰기 |

### 6.4 와이드 컬럼 저장소 사용 사례

- **시계열 데이터**: IoT 센서 판독값, 메트릭, 로그 (행 키 = device_id, 클러스터링 키 = timestamp)
- **이벤트 로깅**: 사용자 또는 서비스별로 분할된 애플리케이션 이벤트 저장
- **콘텐츠 관리**: 기사, 댓글, 메타데이터 저장
- **추천 엔진**: 사용자-항목 상호작용 행렬 저장
- **메시징 시스템**: 대화별로 분할된 메시지 저장

### 6.5 한계

- **JOIN 없음**: 쿼리를 위한 모든 데이터가 하나의 테이블에 있어야 함(비정규화)
- **제한된 보조 인덱스**: 비-키 컬럼으로 쿼리하는 것은 비용이 큼
- **복잡한 데이터 모델링**: 효과적인 파티션 키 설계는 액세스 패턴에 대한 깊은 이해 필요
- **다중 파티션 트랜잭션 없음** (Cassandra는 Paxos를 사용한 경량 트랜잭션을 추가했지만 느림)

---

## 7. 그래프 데이터베이스

### 7.1 속성 그래프 모델

그래프 데이터베이스는 데이터를 노드와 엣지의 네트워크로 저장하여 복잡하고 상호 연결된 관계를 가진 데이터에 이상적입니다.

```
┌──────────────────────────────────────────────────────────────┐
│  속성 그래프 모델                                             │
│                                                              │
│  NODE (Vertex)                    EDGE (Relationship)        │
│  ┌─────────────────┐             ────────────────────        │
│  │ Label: Person   │             Type: FOLLOWS               │
│  │ Properties:     │             Properties:                 │
│  │   name: "Alice" │──FOLLOWS──▶   since: "2023-01"          │
│  │   age: 30       │             Direction: Alice → Bob      │
│  └─────────────────┘                                         │
│                                                              │
│  노드는 다음을 가짐:              엣지는 다음을 가짐:        │
│  - 하나 이상의 레이블            - 정확히 하나의 타입        │
│  - 0개 이상의 속성               - 방향                      │
│  - 고유 ID                       - 0개 이상의 속성           │
│                                  - 시작 노드와 끝 노드       │
└──────────────────────────────────────────────────────────────┘
```

**그래프 예제**:

```
(Alice:Person)──[:FOLLOWS]──▶(Bob:Person)
     │                            │
     │                            │
  [:LIKES]                    [:WROTE]
     │                            │
     ▼                            ▼
(Neo4j:Product)              (Review:Review)
  {name: "Neo4j"}             {rating: 5,
                               text: "Great DB!"}
     ▲
     │
  [:MADE_BY]
     │
(Neo4j Inc:Company)
  {founded: 2007}
```

### 7.2 왜 그래프인가?

소셜 네트워크 쿼리를 고려하세요: "나와 같은 제품을 좋아하는 친구의 친구를 찾기."

**SQL에서** (관계형):

```sql
-- 같은 제품을 좋아하는 친구의 친구
SELECT DISTINCT fof.name
FROM users me
JOIN friendships f1 ON me.id = f1.user_id
JOIN friendships f2 ON f1.friend_id = f2.user_id
JOIN user_likes ul1 ON me.id = ul1.user_id
JOIN user_likes ul2 ON f2.friend_id = ul2.user_id
WHERE me.name = 'Alice'
  AND ul1.product_id = ul2.product_id
  AND f2.friend_id != me.id;

-- 이것은 여러 JOIN이 필요하고, 성능은
-- 순회 깊이에 따라 지수적으로 저하됩니다.
```

**Cypher에서** (Neo4j):

```cypher
// 동일한 쿼리, 그래프 순회로 자연스럽게 표현
MATCH (me:Person {name: "Alice"})-[:FRIEND]->()-[:FRIEND]->(fof:Person),
      (me)-[:LIKES]->(product:Product)<-[:LIKES]-(fof)
WHERE fof <> me
RETURN DISTINCT fof.name;
```

**성능 비교**: 관계 중심 쿼리의 경우, 그래프 데이터베이스는 데이터셋 크기에 관계없이 일정한 성능을 유지하지만(순회가 로컬이기 때문), 관계형 데이터베이스는 테이블이 커지면서 성능이 저하됩니다(JOIN이 더 큰 인덱스 구조를 스캔하기 때문).

```
쿼리 시간
    │
    │  Relational (JOIN 기반)
    │  ╱
    │ ╱
    │╱                    Graph DB (순회 기반)
    │─────────────────────────────────────────
    │
    └──────────────────────────────────────── 데이터 크기
```

### 7.3 Cypher 쿼리 언어

Cypher는 Neo4j의 선언적 그래프 쿼리 언어입니다. 구문은 ASCII 아트를 사용하여 그래프 패턴을 표현합니다.

**패턴 구문**:
```
(node)              -- 노드
(n:Label)           -- 레이블이 있는 노드
(n:Label {prop: v}) -- 속성 필터가 있는 노드
-[r:TYPE]->         -- 방향성 있는 관계
-[r:TYPE*1..3]->    -- 가변 길이 경로 (1~3 홉)
```

**CREATE 연산**:

```cypher
// 노드 생성
CREATE (alice:Person {name: "Alice", age: 30})
CREATE (bob:Person {name: "Bob", age: 28})
CREATE (widget:Product {name: "Widget", price: 9.99})

// 관계 생성
MATCH (alice:Person {name: "Alice"}), (bob:Person {name: "Bob"})
CREATE (alice)-[:FOLLOWS {since: date("2023-01-15")}]->(bob);

MATCH (alice:Person {name: "Alice"}), (w:Product {name: "Widget"})
CREATE (alice)-[:PURCHASED {date: date("2024-03-20"), qty: 2}]->(w);
```

**READ 연산**:

```cypher
// Alice가 팔로우하는 모든 사람 찾기
MATCH (alice:Person {name: "Alice"})-[:FOLLOWS]->(friend)
RETURN friend.name;

// 두 사람 사이의 최단 경로 찾기
MATCH path = shortestPath(
  (alice:Person {name: "Alice"})-[:FOLLOWS*]-(bob:Person {name: "Bob"})
)
RETURN path;

// 추천: Alice와 같은 제품을 구매한 사람들이 구매한 제품
MATCH (alice:Person {name: "Alice"})-[:PURCHASED]->(p:Product)<-[:PURCHASED]-(other),
      (other)-[:PURCHASED]->(rec:Product)
WHERE NOT (alice)-[:PURCHASED]->(rec)
RETURN rec.name, COUNT(*) AS score
ORDER BY score DESC
LIMIT 5;

// PageRank 스타일 영향력: 누가 가장 많은 팔로워를 가지는가?
MATCH (p:Person)<-[:FOLLOWS]-(follower)
RETURN p.name, COUNT(follower) AS followers
ORDER BY followers DESC
LIMIT 10;

// 커뮤니티 감지: 상호 팔로우 클러스터 찾기
MATCH (a:Person)-[:FOLLOWS]->(b:Person)-[:FOLLOWS]->(a)
RETURN a.name, b.name;
```

**UPDATE 및 DELETE**:

```cypher
// 속성 업데이트
MATCH (alice:Person {name: "Alice"})
SET alice.age = 31;

// 레이블 추가
MATCH (alice:Person {name: "Alice"})
SET alice:PremiumUser;

// 관계 삭제
MATCH (alice:Person {name: "Alice"})-[r:FOLLOWS]->(bob:Person {name: "Bob"})
DELETE r;

// 노드와 모든 관계 삭제
MATCH (n:Person {name: "Eve"})
DETACH DELETE n;
```

### 7.4 그래프 사용 사례

| 사용 사례 | 그래프가 뛰어난 이유 |
|-----------|---------------------|
| **소셜 네트워크** | 친구의 친구 쿼리, 영향력, 커뮤니티 감지 |
| **추천 엔진** | 그래프 순회를 통한 협업 필터링 |
| **사기 탐지** | 거래 네트워크에서 의심스러운 패턴 식별 |
| **지식 그래프** | 엔티티와 관계 표현 (Google Knowledge Graph, Wikidata) |
| **네트워크/IT 운영** | 네트워크 토폴로지 모델링, 종속성 추적, 근본 원인 분석 |
| **공급망** | 제조, 배송, 유통을 통한 제품 추적 |
| **액세스 제어** | 권한 계층 구조 및 상속 모델링 |
| **생물학** | 단백질 상호작용 네트워크, 유전자 조절 네트워크 |

### 7.5 그래프 데이터베이스의 한계

- **집계에 적합하지 않음**: SUM, AVG, GROUP BY는 그래프가 빛나는 곳이 아님
- **저장 오버헤드**: 모든 엣지에 대한 관계 포인터 저장은 관계형 외래 키보다 더 많은 공간 사용
- **제한된 수평 확장**: 고도로 연결된 그래프를 노드 간에 분할하는 것은 NP-hard 문제(그래프 분할)
- **작은 생태계**: 관계형 또는 문서 데이터베이스에 비해 도구, 개발자, 커뮤니티 지원이 적음
- **쓰기 처리량**: 일반적으로 키-값 또는 와이드 컬럼 저장소보다 낮음

---

## 8. 비교 매트릭스: 어떤 모델을 사용할지

### 8.1 결정 매트릭스

| 기준 | 키-값 | 문서 | 와이드 컬럼 | 그래프 | 관계형 |
|------|-------|------|------------|-------|--------|
| **스키마 유연성** | 높음 (불투명 값) | 높음 (JSON) | 중간 (컬럼 패밀리 고정) | 높음 (속성 그래프) | 낮음 (고정 스키마) |
| **쿼리 복잡성** | 키 조회만 | 필드에 대한 풍부한 쿼리 | 파티션 키 기반 | 그래프 순회 | 전체 SQL |
| **관계** | 없음 | 임베디드/참조 | 비정규화 | 일등급 | JOIN |
| **쓰기 처리량** | 매우 높음 | 높음 | 매우 높음 | 중간 | 중간 |
| **읽기 지연** | 밀리초 미만 | 낮음 | 낮음 (파티션 키) | 가변 (깊이) | 가변 |
| **수평 확장** | 우수 | 좋음 | 우수 | 제한적 | 나쁨 |
| **ACID 트랜잭션** | 제한적 | 문서당 (일부에서 다중 문서) | 제한적 | 노드당 (일부는 다중 지원) | 전체 |
| **집계** | 없음 | 집계 파이프라인 | 제한적 | 제한적 | 전체 SQL |

### 8.2 결정 플로우차트

```
                         주요 요구사항은 무엇인가?
                                   │
                ┌──────────────────┼──────────────────┐
                │                  │                  │
          고속               구조화된           복잡한
          단순 조회           데이터에 대한      관계
                │            풍부한 쿼리             │
                │                  │                  │
           키-값              문서              그래프 DB
           (Redis,            (MongoDB,          (Neo4j,
            DynamoDB)          CouchDB)           Neptune)
                                   │
                              대규모 쓰기
                              필요?
                              ┌────┴────┐
                              │         │
                             예        아니오
                              │         │
                         와이드 컬럼    문서
                         (Cassandra)  (MongoDB)
```

### 8.3 구체적 권장사항

| 시나리오 | 권장 모델 | 이유 |
|----------|----------|------|
| 사용자 세션 | 키-값 (Redis) | TTL이 있는 단순 GET/SET |
| 제품 카탈로그 | 문서 (MongoDB) | 제품 카테고리별 유연한 속성 |
| IoT 시계열 | 와이드 컬럼 (Cassandra) | 대규모 쓰기 처리량, 시간순 파티션 |
| 소셜 그래프 | 그래프 (Neo4j) | 관계 순회가 핵심 작업 |
| 금융 거래 | 관계형 (PostgreSQL) | ACID 준수 필수 |
| 콘텐츠 관리 | 문서 (MongoDB) | 중첩되고 가변적인 구조 |
| 실시간 분석 | 와이드 컬럼 + 키-값 | 저장용 Cassandra, 캐싱용 Redis |
| 사기 탐지 | 그래프 (Neo4j) | 거래 네트워크에서 패턴 매칭 |
| 쇼핑 카트 | 키-값 (Redis) 또는 문서 | 빠른 읽기/쓰기, 유연한 구조 |
| 추천 엔진 | 그래프 + 문서 | 관계용 그래프, 항목 메타데이터용 문서 |

---

## 9. 폴리글랏 지속성

### 9.1 정의

폴리글랏 지속성(Polyglot Persistence)은 애플리케이션의 각 구성 요소의 특정 액세스 패턴과 요구사항에 따라 다른 데이터 저장 기술을 사용하는 관행입니다.

이 용어는 Martin Fowler와 Pramod Sadalage가 2011년에 만들었습니다.

### 9.2 아키텍처 예제

전자상거래 플랫폼을 고려하세요:

```
┌─────────────────────────────────────────────────────────────────┐
│                    전자상거래 플랫폼                             │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐        │
│  │  Product      │  │  Order       │  │  User          │        │
│  │  Service      │  │  Service     │  │  Service       │        │
│  └──────┬───────┘  └──────┬───────┘  └───────┬────────┘        │
│         │                 │                   │                  │
│         ▼                 ▼                   ▼                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐        │
│  │  MongoDB     │  │  PostgreSQL  │  │  PostgreSQL    │        │
│  │  (Catalog)   │  │  (Orders)    │  │  (Users)       │        │
│  │              │  │  ACID txns   │  │  Auth + RBAC   │        │
│  └──────────────┘  └──────────────┘  └────────────────┘        │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐        │
│  │  Redis       │  │  Neo4j       │  │  Elasticsearch │        │
│  │  (Cache +    │  │  (Recom-     │  │  (Search)      │        │
│  │   Sessions)  │  │   mendations)│  │                │        │
│  └──────────────┘  └──────────────┘  └────────────────┘        │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │  Cassandra   │  │  S3 / Blob   │                             │
│  │  (Analytics  │  │  (Images,    │                             │
│  │   Events)    │  │   Files)     │                             │
│  └──────────────┘  └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

### 9.3 이점

- **최적화된 성능**: 각 서비스가 액세스 패턴에 가장 적합한 데이터베이스 사용
- **독립적 확장**: 워크로드에 따라 각 데이터베이스를 독립적으로 확장
- **기술 유연성**: 팀이 특정 문제에 최적의 도구 채택 가능

### 9.4 과제

- **운영 복잡성**: 여러 데이터베이스 기술은 다양한 전문지식 필요
- **데이터 일관성**: 데이터베이스 간 데이터 동기화가 어려움
- **모니터링 및 디버깅**: 다른 모니터링 도구, 다른 로그 형식
- **교차 저장소 쿼리**: MongoDB와 PostgreSQL의 데이터를 조인하려면 애플리케이션 수준 로직 필요
- **트랜잭션 경계**: 단일 비즈니스 작업이 여러 데이터베이스에 걸쳐 있을 수 있음 (예: "주문하기"는 PostgreSQL에 주문을 쓰고 Redis에 캐시를 씀)

### 9.5 과제 완화

| 과제 | 완화 전략 |
|------|----------|
| 데이터 일관성 | 최종 일관성을 위한 이벤트 기반 아키텍처 (Kafka/RabbitMQ) |
| 교차 저장소 쿼리 | 애플리케이션/게이트웨이 레이어에서 API 구성 |
| 운영 복잡성 | 공유 인프라를 가진 플랫폼 팀 (Kubernetes, Terraform) |
| 트랜잭션 경계 | 분산 트랜잭션을 위한 Saga 패턴 또는 outbox 패턴 |
| 모니터링 | 통합 관찰성 (OpenTelemetry, Grafana) |

### 9.6 폴리글랏 지속성을 사용하지 말아야 할 때

- **작은 팀**: 운영 오버헤드가 정당화되지 않음
- **단순한 애플리케이션**: 단일 PostgreSQL 인스턴스가 대부분의 워크로드를 잘 처리
- **엄격한 일관성 요구사항**: 여러 데이터베이스는 전역 일관성을 거의 불가능하게 만듦
- **초기 단계 스타트업**: 조기 최적화; 하나의 데이터베이스로 시작하고 나중에 분할

> **경험 법칙**: 모든 것에 PostgreSQL로 시작하세요. 특정하고 측정 가능한 한계(성능, 규모, 데이터 모델 불일치)에 도달하면, 그 특정 사용 사례를 위한 특수 데이터베이스 도입을 고려하세요.

---

## 10. 연습문제

### 연습문제 1: CAP 정리 분석

다음 각 시스템에 대해 CP 또는 AP로 분류하세요. 답을 정당화하세요.

1. 계좌 간 송금을 처리하는 은행 시스템.
2. 대략적인 카운트를 표시하는 소셜 미디어 "좋아요" 카운터.
3. DNS 시스템.
4. 분산 구성 저장소 (etcd 또는 ZooKeeper 같은).
5. 쇼핑 카트 서비스.

### 연습문제 2: 데이터 모델 선택

다음 각 시나리오에 대해 가장 적절한 NoSQL 데이터 모델(키-값, 문서, 와이드 컬럼, 그래프)을 선택하고 기본 스키마를 설계하세요.

1. **실시간 멀티플레이어 게임**: 100,000명의 동시 플레이어에 대한 플레이어 위치, 점수, 게임 상태 저장. 업데이트는 플레이어당 초당 60회 발생.

2. **레시피 웹사이트**: 가변 재료, 단계, 영양 정보, 사용자 평가 및 댓글이 있는 레시피 저장. 사용자가 재료로 검색할 수 있어야 함.

3. **족보 애플리케이션**: 수세기에 걸친 부모-자녀 관계, 결혼, 역사적 기록이 있는 가계도 저장.

4. **IoT 차량 관리**: 5초마다 샘플링된 50,000대 차량의 GPS 좌표, 속도, 연료 레벨 저장.

### 연습문제 3: Redis 설계

다음 요구사항을 가진 속도 제한기를 위한 Redis 데이터 모델을 설계하세요:
- 사용자당 분당 100개의 API 요청 허용
- 각 응답과 함께 남은 요청 수 반환
- 각 분 윈도우 시작 시 카운터 재설정

Redis 명령을 사용하여 `check_rate_limit(user_id)` 함수의 의사 코드를 작성하세요.

### 연습문제 4: MongoDB 스키마 설계

블로그 플랫폼을 구축하고 있습니다. 다음 액세스 패턴을 고려하여 MongoDB 스키마를 설계하세요:
1. 모든 댓글과 함께 블로그 게시물 표시
2. 특정 작성자의 최근 게시물 10개 표시
3. 모든 작성자의 최근 게시물 10개 표시
4. 게시물당 총 댓글 수 계산

고려사항:
- 게시물은 0~10,000개의 댓글을 가질 수 있음
- 댓글은 답글을 가질 수 있음 (최대 3단계 깊이)
- 작성자는 이름, 소개, 아바타가 있는 프로필을 가짐

JSON 문서 예제로 스키마를 제공하고 임베딩 vs 참조 결정을 설명하세요.

### 연습문제 5: Cassandra 데이터 모델링

다음 쿼리를 가진 메시징 애플리케이션을 위한 Cassandra 스키마를 설계하세요:
1. 대화의 모든 메시지를 타임스탬프순으로 가져오기 (최신순)
2. 사용자의 모든 대화를 마지막 활동순으로 가져오기
3. 사용자의 대화당 읽지 않은 메시지 수 가져오기

CQL CREATE TABLE 문을 작성하고 파티션 키 및 클러스터링 키 선택을 설명하세요.

### 연습문제 6: Cypher 쿼리

다음 그래프가 주어졌을 때:

```
(:Person {name: "Alice"})-[:WORKS_AT]->(:Company {name: "TechCorp"})
(:Person {name: "Bob"})-[:WORKS_AT]->(:Company {name: "TechCorp"})
(:Person {name: "Charlie"})-[:WORKS_AT]->(:Company {name: "DataInc"})
(:Person {name: "Alice"})-[:KNOWS]-(:Person {name: "Bob"})
(:Person {name: "Bob"})-[:KNOWS]-(:Person {name: "Charlie"})
(:Person {name: "Alice"})-[:KNOWS]-(:Person {name: "Diana"})
(:Person {name: "Diana"})-[:WORKS_AT]->(:Company {name: "DataInc"})
(:Person {name: "Alice"})-[:SKILL]->(:Technology {name: "Python"})
(:Person {name: "Bob"})-[:SKILL]->(:Technology {name: "Python"})
(:Person {name: "Bob"})-[:SKILL]->(:Technology {name: "Java"})
(:Person {name: "Charlie"})-[:SKILL]->(:Technology {name: "Python"})
(:Person {name: "Diana"})-[:SKILL]->(:Technology {name: "Java"})
```

다음에 대한 Cypher 쿼리를 작성하세요:
1. "DataInc"에서 일하는 사람을 아는 모든 사람 찾기 (DataInc에서 직접 일하지 않는).
2. Bob과 최소 2개의 기술을 공유하는 모든 사람 찾기.
3. Alice와 Charlie 사이의 최단 경로 찾기.
4. Alice의 연결이 일하는 곳을 기반으로 Alice에게 회사 추천 (자신의 회사 제외).

### 연습문제 7: 폴리글랏 지속성 설계

차량 공유 플랫폼(Uber/Lyft와 유사)을 설계하고 있습니다. 애플리케이션에는 다음이 필요합니다:
- 실시간 운전자 위치 추적 (2초마다 업데이트)
- 승차 예약 및 결제 처리
- 평점이 있는 운전자 및 승객 프로필
- 경로 계산 및 ETA 추정
- 승차 기록 및 분석
- 사기 탐지 (승차 요청의 의심스러운 패턴 식별)

폴리글랏 지속성 아키텍처를 설계하세요:
1. 각 데이터 도메인과 액세스 패턴 식별.
2. 각 도메인에 대한 데이터베이스 기술 선택. 선택을 정당화.
3. 서비스 간 데이터 흐름을 보여주는 아키텍처 다이어그램 그리기.
4. 주요 데이터 일관성 과제 식별 및 솔루션 제안.

### 연습문제 8: CAP 정리 증명 확장

섹션 2.3의 Gilbert-Lynch 증명 스케치를 확장하여 **3-노드** 시스템(N1, N2, N3)도 네트워크가 {N1}과 {N2, N3}의 두 그룹으로 분할될 때 C, A, P를 동시에 만족할 수 없음을 보이세요.

구체적으로:
1. 초기 상태 정의.
2. 파티션 설명.
3. 소수 파티션(N1)에 대한 쓰기 표시.
4. 다수 파티션(N2 또는 N3)에서의 읽기 표시.
5. 모순 식별.

### 연습문제 9: 일관성 모델 분류

아래 각 시나리오에 대해 필요한 최소 일관성 모델 결정 (가장 강함에서 가장 약함으로: 선형화 가능성, 순차 일관성, 인과 일관성, read-your-writes, 최종 일관성):

1. 사용자가 프로필 사진을 업데이트하고 즉시 프로필 페이지를 봄.
2. 그룹 채팅에서 Alice가 Bob의 메시지에 답장하면, 모든 사람이 Alice의 답장 전에 Bob의 메시지를 봐야 함.
3. 재고 시스템이 절대 제품을 과다 판매해서는 안 됨.
4. 약간 부정확할 수 있지만 결국 모든 좋아요를 반영해야 하는 "좋아요" 카운터.
5. 사용자가 항상 보낸 편지함 폴더에서 보낸 메시지를 보는 이메일 시스템.

### 연습문제 10: 비교 분석

다음을 사용하여 대학 등록 시스템(학생, 과정, 등록, 교수, 학과)을 모델링하는 방법을 비교하는 500단어 에세이를 작성하세요:
1. 관계형 데이터베이스 (PostgreSQL)
2. 문서 데이터베이스 (MongoDB)
3. 그래프 데이터베이스 (Neo4j)

각 모델에 대해:
- 스키마 그리기/설명
- 두 가지 대표적인 쿼리 작성
- 이 사용 사례에 대한 모델의 장점 하나와 단점 하나 나열

---

## 11. 참고문헌

1. Brewer, E. (2000). "Towards Robust Distributed Systems" (CAP Theorem keynote). ACM PODC.
2. Gilbert, S. & Lynch, N. (2002). "Brewer's Conjecture and the Feasibility of Consistent, Available, Partition-Tolerant Web Services." ACM SIGACT News.
3. DeCandia, G. et al. (2007). "Dynamo: Amazon's Highly Available Key-Value Store." SOSP.
4. Chang, F. et al. (2006). "Bigtable: A Distributed Storage System for Structured Data." OSDI.
5. Cattell, R. (2011). "Scalable SQL and NoSQL Data Stores." ACM SIGMOD Record.
6. Sadalage, P. & Fowler, M. (2012). *NoSQL Distilled*. Addison-Wesley.
7. Robinson, I., Webber, J., & Eifrem, E. (2015). *Graph Databases*. O'Reilly Media.
8. Abadi, D. (2012). "Consistency Tradeoffs in Modern Distributed Database System Design." IEEE Computer.
9. MongoDB Manual. "Data Modeling Introduction." https://www.mongodb.com/docs/manual/core/data-modeling-introduction/
10. Apache Cassandra Documentation. https://cassandra.apache.org/doc/latest/

---

**이전**: [12. 동시성 제어](./12_Concurrency_Control.md) | **다음**: [14. 분산 데이터베이스](./14_Distributed_Databases.md)

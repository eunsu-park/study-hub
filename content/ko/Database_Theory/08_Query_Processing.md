# Lesson 08: 쿼리 처리(Query Processing)

**이전**: [07_Advanced_Normalization.md](./07_Advanced_Normalization.md) | **다음**: [09_Indexing.md](./09_Indexing.md)

---

> **주제(Topic)**: Database Theory
> **레슨(Lesson)**: 8 of 16
> **선행 학습(Prerequisites)**: 관계 대수(Relational algebra, Lesson 03), SQL 기초, 디스크 I/O 이해
> **목표(Objective)**: DBMS가 SQL 쿼리를 효율적인 실행 계획으로 변환하는 방법을 이해하고, 선택(selection)과 조인(join) 알고리즘의 비용 모델을 숙달하며, 쿼리 최적화 기법을 파악합니다

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 쿼리 처리 파이프라인(파싱, 변환, 최적화, 실행)을 설명하고 각 단계의 역할을 설명합니다.
2. 디스크 블록 비용 모델을 사용하여 선택 알고리즘(선형 스캔, 기본 인덱스, 보조 인덱스)의 I/O 비용을 계산합니다.
3. 조인 알고리즘(중첩 루프 조인(Nested Loop Join), 블록 중첩 루프 조인(Block Nested Loop Join), 정렬-합병 조인(Sort-Merge Join), 해시 조인(Hash Join))의 비용 특성을 비교하고 주어진 시나리오에 적합한 알고리즘을 선택합니다.
4. 쿼리 최적화기(Query Optimizer)가 통계와 비용 추정을 사용하여 대안 쿼리 계획을 열거하고 평가하는 방법을 설명합니다.
5. 대수적 동치 규칙(Algebraic Equivalence Rules)을 적용하여 쿼리 트리를 변환하고 더 효율적인 실행 계획을 생성합니다.
6. 쿼리 실행 계획(예: EXPLAIN 출력)을 해석하여 SQL 쿼리의 성능 병목 지점을 진단합니다.

---

## 1. 소개(Introduction)

SQL 쿼리를 작성할 때, 데이터베이스는 작성된 그대로 실행하지 않습니다. SQL 문과 실제 디스크 액세스 사이에는 **파싱(parsing)**, **최적화(optimization)**, **실행(execution)**의 정교한 파이프라인이 있습니다. 이 파이프라인을 이해하는 것은 효율적인 쿼리를 작성하고 성능 문제를 진단하는 데 중요합니다.

### 1.1 쿼리 처리 파이프라인(The Query Processing Pipeline)

```
SQL Query
    │
    ▼
┌─────────────────┐
│    Parser        │ → 구문 검사, 파스 트리
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Translator      │ → 관계 대수 표현식
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Optimizer       │ → 최선의 실행 계획 선택
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Execution       │ → 계획 실행, 결과 반환
│  Engine          │
└─────────────────┘
```

### 1.2 예제: 간단한 쿼리의 여정(Example: A Simple Query's Journey)

```sql
SELECT e.name, d.dept_name
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
WHERE e.salary > 80000;
```

1. **파서(Parser)**: 구문 검사, 테이블/열 이름 해석, 파스 트리 생성
2. **변환기(Translator)**: 관계 대수로 변환: π_{name, dept_name}(σ_{salary > 80000}(employees ⋈_{dept_id} departments))
3. **최적화기(Optimizer)**: 많은 동등한 계획을 고려:
   - 먼저 필터링 후 조인? 아니면 먼저 조인 후 필터링?
   - salary에 인덱스 사용? dept_id에?
   - 중첩 루프 조인? 해시 조인? 정렬-병합 조인?
4. **실행 엔진(Execution engine)**: 반복자 모델을 사용하여 선택된 계획 실행

---

## 2. 파싱과 변환(Parsing and Translation)

### 2.1 파싱(Parsing)

파서는 다음을 수행합니다:

1. **어휘 분석(Lexical analysis)**: 쿼리를 토큰으로 분해 (키워드, 식별자, 연산자, 리터럴)
2. **구문 분석(Syntax analysis)**: 쿼리가 SQL 문법 규칙을 따르는지 확인, 파스 트리 생성
3. **의미 분석(Semantic analysis)**: 테이블과 열이 존재하는지, 타입이 호환되는지, 사용자가 권한을 가지는지 확인

**파스 트리** (우리 예제):

```
         SELECT
        /      \
   ProjectList  FROM
   /      \      |
 e.name  d.dept_name  JoinClause
                        /    \
                  employees  departments
                       |
                  ON e.dept_id = d.dept_id
                       |
                  WHERE e.salary > 80000
```

### 2.2 관계 대수로 변환(Translation to Relational Algebra)

파서 출력은 초기 관계 대수 표현식 (또는 **쿼리 트리(query tree)**라고 하는 동등한 내부 표현)으로 변환됩니다:

```
π_{name, dept_name}
    │
    σ_{salary > 80000}
    │
    ⋈_{dept_id}
   / \
  e   d
```

이 초기 표현식은 **논리적으로 정확**하지만 반드시 **효율적이지는 않습니다**. 최적화기의 역할은 동등하지만 더 빠른 계획을 찾는 것입니다.

---

## 3. 쿼리 평가 계획과 반복자 모델(Query Evaluation Plans and the Iterator Model)

### 3.1 쿼리 평가 계획(Query Evaluation Plan)

**쿼리 평가 계획(query evaluation plan)** (또는 실행 계획)은 다음을 명시합니다:
- 수행할 관계 대수 연산
- 각 연산에 사용할 **알고리즘**
- 연산이 실행되는 **순서**
- 연산 간 데이터 흐름 방식

### 3.2 반복자(Volcano/Pipeline) 모델(The Iterator (Volcano/Pipeline) Model)

대부분의 현대 데이터베이스는 **반복자 모델(iterator model)** (Goetz Graefe의 Volcano 쿼리 처리 시스템에서 따와 Volcano 모델이라고도 함)을 사용합니다:

모든 연산자는 세 가지 메서드를 구현합니다:

```
open()   → 연산자 초기화. 자식 반복자 열기, 버퍼 할당.
next()   → 결과의 다음 튜플 반환. 필요에 따라 자식의 next() 호출.
close()  → 정리. 버퍼 해제, 자식 반복자 닫기.
```

**핵심 통찰**: 연산자들이 트리로 구성됩니다. 루트가 `next()`를 호출하면, 이것이 리프(테이블 스캔)까지 계단식으로 내려갑니다. 튜플들은 한 번에 하나씩 **위로** 흐릅니다.

```
         π_{name, dept_name}     ← 루트가 next() 호출
              │
         σ_{salary > 80000}     ← 필터링, 일치하는 튜플을 위로 전달
              │
         ⋈_{dept_id}            ← 조인된 튜플 생성
            / \
      Scan(e)  Scan(d)          ← 디스크에서 튜플 읽기
```

### 3.3 구체화 vs 파이프라이닝(Materialization vs Pipelining)

**구체화(Materialization)**: 각 연산자가 **전체** 결과를 생성하여 임시 릴레이션에 저장한 다음 부모에 전달합니다. 간단하지만 많은 임시 저장 공간이 필요합니다.

**파이프라이닝(Pipelining)**: 튜플들이 완전히 구체화되지 않고 연산자들을 통해 흐릅니다. 한 튜플이 생성되자마자 다음 연산자로 전달됩니다. 훨씬 더 메모리 효율적입니다.

```
구체화:
  Scan(e) → [전체 임시 테이블] → σ → [전체 임시 테이블] → ⋈ → [전체 임시 테이블] → π

파이프라이닝:
  Scan(e) → σ → ⋈ → π
  (튜플별로, 전체 임시 테이블 없음)
```

파이프라이닝이 선호되지만 항상 가능하지는 않습니다. 일부 연산은 **블로킹(blocking)**입니다 — 출력을 생성하기 전에 모든 입력을 소비해야 합니다:
- **정렬(Sorting)** (정렬하려면 모든 튜플을 봐야 함)
- **해시 조인 빌드 단계(Hash join build phase)** (전체 해시 테이블을 빌드해야 함)
- **집계(Aggregation)** (모든 그룹을 처리해야 함)

### 3.4 풀 vs 푸시 모델(Pull vs Push Model)

위에서 설명한 반복자 모델은 **풀(pull)** 모델 (또는 수요 주도)입니다: 부모가 `next()`를 호출하여 자식으로부터 튜플을 가져옵니다.

현대 시스템은 점점 더 **푸시(push)** 모델 (또는 데이터 주도)을 사용합니다: 자식이 부모에게 튜플을 푸시합니다. 이는 더 캐시 친화적이고 컴파일에 적합할 수 있습니다.

```
풀 (Volcano):                    푸시:
  부모가 child.next() 호출          자식이 parent.consume(tuple) 호출
  자식이 하나의 튜플 반환            부모가 즉시 처리
  부모가 처리                       더 캐시 친화적
```

일부 시스템 (예: HyPer, Umbra)은 쿼리를 데이터를 연산자들을 통해 푸시하는 타이트한 루프로 컴파일하여 거의 손으로 코딩한 것과 같은 성능을 달성합니다.

---

## 4. 비용 추정(Cost Estimation)

### 4.1 비용 지표(Cost Metrics)

쿼리 처리의 주요 비용:

| 비용 구성 요소 | 기호 | 설명 |
|---------------|--------|-------------|
| **디스크 I/O** | tT, tS | 전송 시간(순차 읽기)과 탐색 시간 |
| **CPU** | — | 비교, 해싱, 계산 |
| **메모리** | M | 사용 가능한 버퍼 페이지 |
| **네트워크** | — | 분산 쿼리용 |

전통적 시스템에서는 **디스크 I/O가 지배적**입니다. 다음과 같은 디스크의 경우:
- 탐색 시간(Seek time, tS) ≈ 4 ms
- 블록당 전송 시간(Transfer time per block, tT) ≈ 0.1 ms

단일 랜덤 I/O는 ~4.1 ms가 소요되는 반면, 순차 읽기는 블록당 ~0.1 ms가 소요됩니다. 이 40:1 비율은 순차 액세스 패턴이 왜 그렇게 중요한지 설명합니다.

### 4.2 표기법(Notation)

| 기호 | 의미 |
|--------|---------|
| n_r | 릴레이션 r의 튜플 수 |
| b_r | r의 튜플을 포함하는 디스크 블록 수 |
| l_r | r의 튜플 크기(바이트) |
| f_r | 블로킹 팩터: 블록당 튜플 = ⌊B / l_r⌋ |
| B | 블록(페이지) 크기(바이트) |
| V(A, r) | r의 속성 A의 고유 값 수 |
| M | 메모리의 사용 가능한 버퍼 페이지 수 |

관계: b_r = ⌈n_r / f_r⌉

### 4.3 예제 카탈로그 통계(Example Catalog Statistics)

```
employees (e):
    n_e = 10,000 튜플
    l_e = 200 바이트
    B   = 4,096 바이트 (4 KB 페이지)
    f_e = ⌊4096 / 200⌋ = 20 튜플/블록
    b_e = ⌈10000 / 20⌉ = 500 블록
    V(dept_id, e) = 50 고유 부서
    V(salary, e) = 2,000 고유 급여 값

departments (d):
    n_d = 50 튜플
    l_d = 100 바이트
    f_d = ⌊4096 / 100⌋ = 40 튜플/블록
    b_d = ⌈50 / 40⌉ = 2 블록
```

---

## 5. 선택 구현(Selection Implementation)

선택(Selection, σ)은 술어를 만족하는 튜플을 필터링합니다. 구현 전략은 사용 가능한 인덱스에 크게 의존합니다.

### 5.1 알고리즘 A1: 선형 스캔(전체 테이블 스캔) (Linear Scan (Full Table Scan))

릴레이션의 모든 블록을 스캔하여 각 튜플을 술어에 대해 테스트합니다.

```
ALGORITHM: LinearScan(r, predicate)
FOR EACH block b in r DO
    FOR EACH tuple t in b DO
        IF t satisfies predicate THEN
            output t
        END IF
    END FOR
END FOR
```

**비용**: b_r 블록 전송 + 1 탐색

우리 예제: 500 전송 + 1 탐색 = 500 × 0.1ms + 4ms = **54 ms**

**사용 시기**: 항상 적용 가능. 인덱스가 없거나 선택도가 매우 낮을 때 (대부분의 튜플이 자격) 사용됩니다.

### 5.2 알고리즘 A2: 이진 검색(Binary Search)

파일이 선택 속성으로 정렬되어 있고 술어가 동등 조건이면:

```
ALGORITHM: BinarySearch(r, A = v)
이진 검색을 사용하여 A = v를 포함하는 첫 번째 블록 찾기
앞으로 스캔하여 모든 일치하는 튜플 찾기
```

**비용**: 검색을 위한 ⌈log₂(b_r)⌉ 탐색 및 전송 + 중복 값을 위한 추가 블록

키에 대한 동등 조건: ⌈log₂(500)⌉ = 9 블록 액세스 = 9 × (4ms + 0.1ms) = **37 ms**

### 5.3 알고리즘 A3: 기본 인덱스, 키에 대한 동등 조건(Primary Index, Equality on Key)

선택 속성 (키인)에 기본 B⁺-트리 인덱스가 존재하면:

```
비용 = (h_i + 1) × (tS + tT)
```

여기서 h_i는 B⁺-트리의 높이 (일반적으로 2-4).

h_i = 3인 경우: 4 × 4.1ms = **16.4 ms** (3 인덱스 레벨 + 1 데이터 블록)

### 5.4 알고리즘 A4: 기본 인덱스, 비키에 대한 동등 조건(Primary Index, Equality on Non-Key)

여러 튜플이 일치할 수 있습니다. 이들은 연속적입니다 (파일이 이 속성으로 정렬되어 있으므로):

```
비용 = h_i × (tS + tT) + tS + tT × b
```

여기서 b는 일치하는 튜플을 포함하는 블록 수입니다.

### 5.5 알고리즘 A5: 보조 인덱스, 동등 조건(Secondary Index, Equality)

**후보 키에 대해** (최대 하나의 일치):
```
비용 = (h_i + 1) × (tS + tT)
```

키 속성에 대한 기본 인덱스와 동일합니다.

**비키 속성에 대해** (여러 일치):
```
비용 = (h_i + n) × (tS + tT)
```

여기서 n은 일치하는 튜플 수입니다. 각 일치하는 튜플은 **다른 블록**에 있을 수 있으므로 (기본 인덱스처럼 연속적이지 않음), 각각 별도의 탐색이 필요합니다.

이는 낮은 선택도 술어에 대해 **매우 비쌀** 수 있습니다. n = 500이면, 비용은 (3 + 500) × 4.1ms = **2,062 ms** — 전체 테이블 스캔 (54 ms)보다 훨씬 나쁩니다!

### 5.6 범위 술어를 사용한 선택(Selection with Range Predicates)

`salary > 80000`와 같은 술어의 경우:

| 방법 | 비용 |
|--------|------|
| 선형 스캔 | b_r (항상 작동) |
| 기본 인덱스 (B⁺-트리) | h_i + b/2 (평균적으로 리프 레벨의 절반 스캔) |
| 보조 인덱스 (B⁺-트리) | h_i + 범위의 리프 페이지 + 일치하는 레코드 포인터 |

### 5.7 복잡한 술어를 사용한 선택(Selection with Complex Predicates)

**연언 선택(Conjunctive selection)** (σ_{θ₁ ∧ θ₂ ∧ ... ∧ θₙ}):

1. 한 조건에 인덱스가 있으면, 그것을 사용하고 나머지 조건을 필터로 적용
2. 여러 조건에 인덱스가 있으면, **인덱스 교집합(index intersection)** 사용: 각 인덱스에서 레코드 포인터를 가져와 교집합을 구한 다음 일치하는 레코드 검색
3. 여러 속성에 대한 복합 인덱스 (사용 가능하면 이상적)

**선언 선택(Disjunctive selection)** (σ_{θ₁ ∨ θ₂ ∨ ... ∨ θₙ}):

1. **모든** 조건에 인덱스가 있으면, **인덱스 합집합(index union)** 사용: 각 인덱스에서 포인터를 가져와 합집합을 구함
2. 어떤 조건에 인덱스가 없으면, 선형 스캔을 사용해야 함 (하나의 누락된 인덱스가 전체 접근 방식을 무효화)

### 5.8 비교 요약(Comparison Summary)

| 알고리즘 | 조건 | 비용 (블록) |
|-----------|-----------|---------------|
| 선형 스캔 | 항상 | b_r |
| 이진 검색 | 정렬된 파일, 동등 조건 | ⌈log₂(b_r)⌉ |
| 기본 B⁺-트리, 키 | 키에 대한 인덱스 | h_i + 1 |
| 기본 B⁺-트리, 비키 | 비키에 대한 인덱스 | h_i + 일치하는 블록 |
| 보조 B⁺-트리, 키 | 키에 대한 인덱스 | h_i + 1 |
| 보조 B⁺-트리, 비키 | 비키에 대한 인덱스 | h_i + n (각 일치 = 1 탐색!) |

---

## 6. 조인 알고리즘(Join Algorithms)

조인은 일반적으로 쿼리 처리에서 가장 비용이 많이 드는 연산입니다. 조인 알고리즘의 선택이 성능에 극적인 영향을 미칩니다.

### 6.1 표기법(Notation)

릴레이션 r (외부)과 s (내부)를 조인합니다:
- b_r, b_s = 블록 수
- n_r, n_s = 튜플 수
- M = 사용 가능한 메모리 페이지

### 6.2 알고리즘 J1: 중첩 루프 조인(Nested Loop Join, NLJ)

가장 간단한 조인 알고리즘. r의 각 튜플에 대해 s의 모든 것을 스캔하여 일치를 찾습니다.

```
ALGORITHM: NestedLoopJoin(r, s, θ)
FOR EACH tuple t_r IN r DO
    FOR EACH tuple t_s IN s DO
        IF (t_r, t_s) satisfies θ THEN
            output (t_r ⋈ t_s)
        END IF
    END FOR
END FOR
```

**비용 (최악의 경우 — 각 릴레이션에 단일 버퍼 페이지)**:

```
비용 = n_r × b_s + b_r   블록 전송
     = n_r + b_r          탐색
```

r의 n_r 튜플 각각에 대해 s의 모든 b_s 블록을 스캔합니다. 추가로 r 자체의 b_r 블록 읽기.

**예제**: employees (외부)를 departments (내부)와 조인:
- n_r = 10,000, b_s = 2, b_r = 500
- 전송: 10,000 × 2 + 500 = 20,500
- 탐색: 10,000 + 500 = 10,500
- 시간: 20,500 × 0.1ms + 10,500 × 4ms = **44,050 ms ≈ 44초**

**최적화**: 항상 **더 작은** 릴레이션을 내부(s)로 배치하세요. 교환하면:
- n_r = 50, b_s = 500, b_r = 2
- 전송: 50 × 500 + 2 = 25,002
- 이는 전송에서는 더 나쁘지만 탐색에서는 더 좋습니다.

실제로 튜플 수준 중첩 루프는 거의 사용되지 않습니다. 블록 수준이 훨씬 더 좋습니다.

### 6.3 알고리즘 J2: 블록 중첩 루프 조인(Block Nested Loop Join, BNLJ)

튜플별 반복 대신 블록별로 반복합니다.

```
ALGORITHM: BlockNestedLoopJoin(r, s, θ)
FOR EACH block B_r OF r DO
    FOR EACH block B_s OF s DO
        FOR EACH tuple t_r IN B_r DO
            FOR EACH tuple t_s IN B_s DO
                IF (t_r, t_s) satisfies θ THEN
                    output (t_r ⋈ t_s)
                END IF
            END FOR
        END FOR
    END FOR
END FOR
```

**비용**:

```
블록 전송 = b_r × b_s + b_r
탐색     = 2 × b_r
```

r의 각 블록은 한 번 읽힙니다. r의 각 블록에 대해 s의 전부가 스캔됩니다 (b_s 블록). s는 b_r번 읽힙니다.

**예제**: 동일한 테이블:
- 전송: 500 × 2 + 500 = 1,500
- 탐색: 2 × 500 = 1,000
- 시간: 1,500 × 0.1ms + 1,000 × 4ms = **4,150 ms ≈ 4.2초**

튜플 수준 NLJ보다 10배 개선!

**M 버퍼 페이지를 사용한 추가 최적화**:

외부 릴레이션에 (M - 2) 페이지, 내부에 1 페이지, 출력에 1 페이지 사용:

```
블록 전송 = ⌈b_r / (M-2)⌉ × b_s + b_r
탐색     = 2 × ⌈b_r / (M-2)⌉
```

M = 52인 경우 (외부에 50 페이지, 내부에 1, 출력에 1):
- 외부 청크: ⌈500 / 50⌉ = 10
- 전송: 10 × 2 + 500 = 520
- 탐색: 2 × 10 = 20
- 시간: 520 × 0.1ms + 20 × 4ms = **132 ms**

전체 외부가 메모리에 맞으면 (b_r ≤ M - 2), 비용은 **b_r + b_s** 전송과 **2** 탐색뿐입니다 — 단일 패스!

### 6.4 알고리즘 J3: 인덱스 중첩 루프 조인(Indexed Nested Loop Join)

내부 릴레이션의 조인 속성에 인덱스가 있으면, 스캔 대신 사용합니다.

```
ALGORITHM: IndexedNestedLoopJoin(r, s, θ)
FOR EACH tuple t_r IN r DO
    s의 인덱스를 사용하여 t_r과 일치하는 튜플 찾기
    FOR EACH matching t_s DO
        output (t_r ⋈ t_s)
    END FOR
END FOR
```

**비용**:

```
비용 = b_r + n_r × c
```

여기서 c는 s에 대한 단일 인덱스 조회 비용입니다 (일반적으로 B⁺-트리의 키에 대한 동등 조건의 경우 h_i + 1).

**예제**: departments.dept_id에 인덱스 (h_i = 2):
- c = 2 + 1 = 3 (인덱스 순회 + 1 데이터 블록)
- 비용: 500 + 10,000 × 3 = 30,500 블록 액세스
- 하지만 탐색을 고려하면: 인덱스가 메모리에 있으면 BNLJ보다 훨씬 더 좋음

인덱스가 버퍼 캐시에 있으면 (작은 인덱스의 경우 일반적):
- c ≈ 1 (데이터 블록만)
- 비용: 500 + 10,000 × 1 = 10,500 전송

### 6.5 알고리즘 J4: 정렬-병합 조인(Sort-Merge Join)

조인 속성으로 두 릴레이션을 정렬한 다음 병합합니다.

```
ALGORITHM: SortMergeJoin(r, s, join_attr)
Phase 1: r을 join_attr로 정렬 (외부 병합 정렬)
Phase 2: s를 join_attr로 정렬 (외부 병합 정렬)
Phase 3: 병합
    p_r ← 정렬된 r의 첫 번째 튜플
    p_s ← 정렬된 s의 첫 번째 튜플
    WHILE 릴레이션이 소진되지 않음 DO
        IF p_r[join_attr] = p_s[join_attr] THEN
            모든 일치하는 조합 출력
            동등 그룹을 지나 두 포인터 모두 전진
        ELSE IF p_r[join_attr] < p_s[join_attr] THEN
            p_r 전진
        ELSE
            p_s 전진
        END IF
    END WHILE
```

**비용**:

```
정렬 비용 = O(b × log_M(b)) 각 릴레이션에 대해 (외부 병합 정렬)
병합 비용 = b_r + b_s (정렬된 두 릴레이션을 통한 단일 패스)

총 = sort(r) + sort(s) + b_r + b_s
```

b 블록과 M 메모리 페이지를 가진 릴레이션의 외부 병합 정렬 비용:
- 초기 정렬 후 런 수: ⌈b / M⌉
- 병합 패스 수: ⌈log_{M-1}(⌈b/M⌉)⌉
- 각 패스는 모든 블록을 읽고 씁니다: 패스당 2 × b
- 총 정렬 비용: 2 × b × (1 + ⌈log_{M-1}(⌈b/M⌉)⌉) 블록 전송

**예제** (M = 52):
- employees 정렬: ⌈500/52⌉ = 10 런, ⌈log₅₁(10)⌉ = 1 병합 패스
  - 비용: 2 × 500 × (1 + 1) = 2,000 전송
- departments 정렬: 이미 메모리에 맞음 (2 블록 < 52)
  - 비용: 2 × 2 = 4 전송
- 병합: 500 + 2 = 502 전송
- **총: 2,506 전송**

**정렬-병합이 뛰어난 경우**:
- 두 릴레이션이 이미 정렬되어 있음 (정렬 단계 건너뛰기!)
- 해시 조인이 메모리를 다 쓰는 큰 릴레이션
- 비동등 조인 (정렬-병합은 θ-조인을 처리할 수 있지만 해시 조인은 불가능)

### 6.6 알고리즘 J5: 해시 조인(Hash Join)

더 작은 릴레이션에 해시 테이블을 빌드한 다음 더 큰 것으로 프로브합니다.

```
ALGORITHM: HashJoin(r, s, join_attr)
Phase 1 (빌드): 더 작은 릴레이션(s라고 하자)을 메모리에 해시
    hash_table ← {}
    FOR EACH tuple t_s IN s DO
        bucket ← hash(t_s[join_attr])
        t_s를 hash_table[bucket]에 삽입
    END FOR

Phase 2 (프로브): 더 큰 릴레이션 스캔, 해시 테이블 프로브
    FOR EACH tuple t_r IN r DO
        bucket ← hash(t_r[join_attr])
        FOR EACH t_s IN hash_table[bucket] DO
            IF t_r[join_attr] = t_s[join_attr] THEN
                output (t_r ⋈ t_s)
            END IF
        END FOR
    END FOR
```

**비용 (빌드 릴레이션이 메모리에 맞는 경우)**:

```
비용 = b_s + b_r  블록 전송 (두 릴레이션을 한 번씩 읽기)
     = 2          탐색
```

이것은 최적입니다! 각 릴레이션을 정확히 한 번씩 읽습니다.

**예제**: departments (2 블록)가 메모리에 맞음:
- 비용: 2 + 500 = 502 전송, 2 탐색
- 시간: 502 × 0.1ms + 2 × 4ms = **58.2 ms**

**Grace 해시 조인 (빌드가 메모리에 맞지 않을 때)**:

더 작은 릴레이션이 메모리에 맞지 않으면, 파티셔닝 사용:

```
Phase 1 (파티션): r과 s를 모두 M-1 파티션으로 해시
    각 파티션은 디스크에 기록됨

Phase 2 (빌드 & 프로브): 각 파티션 i에 대해:
    s의 파티션 i를 해시 테이블에 로드
    r의 파티션 i를 스캔, 해시 테이블 프로브
```

**비용**:

```
파티셔닝: 2 × (b_r + b_s)    전송 (읽기 + 쓰기 모두)
빌드 & 프로브: b_r + b_s     전송 (두 파티션 읽기)
총: 3 × (b_r + b_s)          전송
```

**요구사항**: 더 작은 릴레이션의 각 파티션이 메모리에 맞아야 함:
```
b_s / (M - 1) ≤ M - 2
⟹ b_s ≤ (M - 1)(M - 2) ≈ M²
```

따라서 해시 조인은 더 작은 릴레이션이 최대 약 M² 블록을 가지면 작동합니다.

### 6.7 비용 비교(Cost Comparison)

| 알고리즘 | 블록 전송 | 탐색 | 최선의 경우 |
|-----------|:-:|:-:|-----------|
| 중첩 루프 | n_r × b_s + b_r | n_r + b_r | 절대 (최악의 경우) |
| 블록 중첩 루프 | ⌈b_r/(M-2)⌉ × b_s + b_r | 2⌈b_r/(M-2)⌉ | 인덱스 없음, 작은 M |
| 인덱스 NL | b_r + n_r × c | b_r + n_r | 내부 조인 속성에 인덱스 |
| 정렬-병합 | 정렬 비용 + b_r + b_s | 많은 탐색 | 이미 정렬됨, 또는 θ-조인 |
| 해시 조인 (인메모리) | b_r + b_s | 2 | 더 작은 릴레이션이 메모리에 맞음 |
| Grace 해시 조인 | 3(b_r + b_s) | 중간 | 큰 릴레이션, M² 충분 |

**우리 예제에 대한 실제 비교** (employees ⋈ departments, M = 52):

| 알고리즘 | 전송 | 시간 (대략) |
|-----------|-----------|---------------|
| 튜플 NLJ | 20,500 | 44초 |
| 블록 NLJ (M=52) | 520 | 132 ms |
| 정렬-병합 | 2,506 | ~260 ms |
| 해시 조인 (인메모리) | 502 | 58 ms |

더 작은 릴레이션이 메모리에 맞을 때 해시 조인이 결정적으로 승리합니다.

---

## 7. 쿼리 최적화(Query Optimization)

### 7.1 개요(Overview)

최적화기는 초기 쿼리 계획을 동등하지만 더 효율적인 것으로 변환합니다. 두 가지 주요 접근 방식:

1. **휴리스틱(규칙 기반) 최적화(Heuristic (rule-based) optimization)**: "거의 항상" 유익한 변환 규칙 적용
2. **비용 기반 최적화(Cost-based optimization)**: 대안 계획 나열, 각 비용 추정, 가장 저렴한 것 선택

대부분의 실제 시스템은 두 가지의 조합을 사용합니다.

### 7.2 관계 대수의 동등 규칙(Equivalence Rules for Relational Algebra)

이러한 규칙은 최적화기가 한 표현식을 동등한 것으로 변환할 수 있게 합니다:

#### 규칙 1: 선택의 연쇄(Cascade of Selections)

```
σ_{θ₁ ∧ θ₂}(r) = σ_{θ₁}(σ_{θ₂}(r))
```

연언은 순차적 선택으로 분할될 수 있습니다.

#### 규칙 2: 선택의 교환성(Commutativity of Selection)

```
σ_{θ₁}(σ_{θ₂}(r)) = σ_{θ₂}(σ_{θ₁}(r))
```

선택의 순서는 중요하지 않습니다.

#### 규칙 3: 투영의 연쇄(Cascade of Projections)

```
π_{L₁}(π_{L₂}(...(π_{Lₙ}(r)))) = π_{L₁}(r)
```

가장 바깥쪽 투영만 중요합니다 (L₁ ⊆ L₂ ⊆ ... ⊆ Lₙ인 한).

#### 규칙 4: 조인의 교환성(Commutativity of Join)

```
r ⋈ s = s ⋈ r
```

#### 규칙 5: 조인의 결합성(Associativity of Join)

```
(r ⋈ s) ⋈ t = r ⋈ (s ⋈ t)
```

이는 다방향 조인에 중요합니다. n개 테이블의 경우, (2(n-1))! / (n-1)! 다른 조인 순서가 있습니다 (카탈란 수). 5개 테이블의 경우, 14개 순서. 10개 테이블: 4,862개.

#### 규칙 6: 조인을 통한 선택 푸시(Push Selection Through Join)

```
σ_{θ}(r ⋈ s) = σ_{θ}(r) ⋈ s     (θ가 r의 속성만 포함하는 경우)
```

이것이 가장 중요한 단일 최적화입니다: **일찍 필터링하여 중간 결과 크기 감소**.

#### 규칙 7: 집합 연산을 통한 선택 푸시(Push Selection Through Set Operations)

```
σ_{θ}(r ∪ s) = σ_{θ}(r) ∪ σ_{θ}(s)
σ_{θ}(r ∩ s) = σ_{θ}(r) ∩ s     (또는 r ∩ σ_{θ}(s))
σ_{θ}(r - s) = σ_{θ}(r) - s
```

#### 규칙 8: 조인을 통한 투영 푸시(Push Projection Through Join)

```
π_{L}(r ⋈_{θ} s) = π_{L}(π_{L₁}(r) ⋈_{θ} π_{L₂}(s))
```

여기서 L₁ = L 또는 θ에 필요한 r의 속성, L₂ = L 또는 θ에 필요한 s의 속성.

### 7.3 휴리스틱 최적화(Heuristic Optimization)

일반 전략:

1. 연언 선택 **분해** (규칙 1)
2. 선택을 가능한 한 아래로 **푸시** (규칙 6, 7)
3. 투영을 가능한 한 아래로 **푸시** (규칙 8)
4. **조인 순서 선택**: 가장 선택적인 조인을 먼저 배치
5. 파이프라인으로 실행될 수 있는 **하위 트리 식별**

#### 예제: 휴리스틱 최적화

원본:

```
π_{e.name, d.dept_name}(σ_{e.salary > 80000 ∧ d.building = 'Watson'}(employees ⋈ departments))
```

**1단계**: 선택 분해
```
π_{e.name, d.dept_name}(σ_{e.salary > 80000}(σ_{d.building = 'Watson'}(employees ⋈ departments)))
```

**2단계**: 선택 푸시 다운
```
π_{e.name, d.dept_name}(σ_{e.salary > 80000}(employees) ⋈ σ_{d.building = 'Watson'}(departments))
```

**3단계**: 투영 푸시 다운
```
π_{e.name, d.dept_name}(
    π_{e.name, e.dept_id}(σ_{e.salary > 80000}(employees))
    ⋈
    π_{d.dept_id, d.dept_name}(σ_{d.building = 'Watson'}(departments))
)
```

**전후 비교:**

```
전: 모든 직원을 모든 부서와 조인, 그 다음 필터링.
  비용: 10,000 × 50 = 500,000 중간 튜플

후: 직원 필터링 (1,000개 남음) 및 부서 (5개 남음), 그 다음 조인.
  비용: 1,000 × 5 = 5,000 중간 튜플 — 100배 감소!
```

### 7.4 비용 기반 최적화(Cost-Based Optimization)

휴리스틱 최적화는 좋지만 충분하지 않습니다. 최적화기는 각 계획의 **실제 비용**을 추정하여 최선의 것을 선택해야 합니다.

#### 선택도 추정(Selectivity Estimation)

술어의 **선택도(selectivity)**는 그것을 만족하는 튜플의 분율을 추정합니다:

| 술어 | 추정 선택도 |
|-----------|----------------------|
| A = v (동등 조건) | 1 / V(A, r) |
| A > v (범위, 균등 분포) | (max(A) - v) / (max(A) - min(A)) |
| A ≥ v₁ AND A ≤ v₂ | (v₂ - v₁) / (max(A) - min(A)) |
| θ₁ ∧ θ₂ (연언, 독립) | sel(θ₁) × sel(θ₂) |
| θ₁ ∨ θ₂ (선언, 독립) | sel(θ₁) + sel(θ₂) - sel(θ₁) × sel(θ₂) |
| NOT θ | 1 - sel(θ) |

**예제**: σ_{salary > 80000}(employees)의 크기 추정

급여가 30,000부터 150,000까지의 범위 (균등 분포):
```
sel = (150,000 - 80,000) / (150,000 - 30,000) = 70,000 / 120,000 ≈ 0.583
추정 튜플 = 10,000 × 0.583 ≈ 5,833
```

#### 조인 크기 추정(Join Size Estimation)

속성 A에 대한 자연 조인 r ⋈ s의 경우:

```
추정 크기 = (n_r × n_s) / max(V(A, r), V(A, s))
```

**예제**: dept_id에 대한 employees ⋈ departments:
```
크기 = (10,000 × 50) / max(50, 50) = 500,000 / 50 = 10,000
```

이것은 의미가 있습니다: 각 직원은 하나의 부서에 있으므로, 조인은 직원당 하나의 튜플을 생성합니다.

#### 히스토그램(Histograms)

균등 분포 가정은 종종 부정확합니다. 실제 데이터베이스는 **히스토그램** — 값의 분포에 대한 통계를 유지합니다:

**등폭 히스토그램(Equi-width histogram)**: 값 범위를 동일 폭 버킷으로 나누고, 버킷당 튜플을 세헤아립니다.

```
급여 히스토그램 (5 버킷):
  [30K-54K):  2,500명 직원
  [54K-78K):  3,000명 직원
  [78K-102K): 2,500명 직원
  [102K-126K): 1,500명 직원
  [126K-150K]: 500명 직원
```

이 히스토그램으로 σ_{salary > 80000}을 추정하면:
```
(102K-80K)/(102K-78K) × 2,500 + 1,500 + 500 = (22/24) × 2,500 + 2,000 ≈ 4,292
```

균등 추정 5,833보다 훨씬 더 정확합니다!

**등깊이(등높이) 히스토그램(Equi-depth (equi-height) histogram)**: 각 버킷이 대략 동일한 수의 튜플을 가집니다. 치우친 분포에 더 좋습니다.

### 7.5 조인 순서 최적화(Join Ordering Optimization)

다방향 조인의 경우, 순서가 엄청나게 중요합니다. 다음을 고려하세요:

```sql
SELECT *
FROM r1 JOIN r2 ON ... JOIN r3 ON ... JOIN r4 ON ...
```

가능한 순서 (4개 테이블):
1. ((r1 ⋈ r2) ⋈ r3) ⋈ r4
2. (r1 ⋈ (r2 ⋈ r3)) ⋈ r4
3. (r1 ⋈ r2) ⋈ (r3 ⋈ r4)
4. ... (훨씬 더 많음)

최적화기는 **동적 프로그래밍**을 사용하여 최선의 순서를 찾습니다:

```
ALGORITHM: FindBestJoinOrder({R₁, R₂, ..., Rₙ})

FOR each single relation Rᵢ DO
    bestPlan({Rᵢ}) ← Rᵢ에 대한 액세스 경로
END FOR

FOR size = 2 TO n DO
    FOR each subset S of size 'size' DO
        bestPlan(S) ← S를 비어 있지 않은 S₁ ∪ S₂로 분할하는
                       모든 방법에 대한 MIN:
                       cost(bestPlan(S₁) ⋈ bestPlan(S₂))
    END FOR
END FOR

RETURN bestPlan({R₁, R₂, ..., Rₙ})
```

이것은 모든 가능한 조인 트리(왼쪽-깊은 트리뿐만 아니라 덤불 트리도 포함)를 고려합니다. 복잡도: O(3ⁿ) — 지수적이지만 최대 ~15-20개 테이블까지의 쿼리에 실용적입니다.

더 큰 쿼리의 경우, 휴리스틱 또는 탐욕 알고리즘이 대신 사용됩니다.

### 7.6 왼쪽-깊은 vs 덤불 조인 트리(Left-Deep vs Bushy Join Trees)

```
왼쪽-깊은 트리:              덤불 트리:

        ⋈                        ⋈
       / \                      / \
      ⋈   R₄                  ⋈   ⋈
     / \                      / \ / \
    ⋈   R₃                  R₁ R₂ R₃ R₄
   / \
  R₁  R₂
```

**왼쪽-깊은 트리**는 많은 최적화기가 선호합니다:
1. 각 조인 단계의 내부 릴레이션이 파이프라이닝을 사용할 수 있습니다 (구체화 없음)
2. 인덱스 중첩 루프 조인이 자연스럽게 작동합니다 (내부 = 인덱스된 테이블)
3. 검색 공간이 더 작습니다: n! 순서 vs 덤불 트리의 경우 지수적으로 더 많음

---

## 8. 통계 및 카탈로그 정보(Statistics and Catalog Information)

### 8.1 카탈로그에 저장되는 것(What the Catalog Stores)

시스템 카탈로그(메타데이터)는 비용 추정을 위한 통계를 유지합니다:

```sql
-- PostgreSQL 카탈로그 테이블:
pg_class     -- 테이블/인덱스 통계 (n_r, b_r 등)
pg_statistic -- 열 수준 통계 (히스토그램, 고유 값, 상관관계)
pg_stats     -- 통계의 사람이 읽을 수 있는 뷰
```

주요 통계:
- **n_r** (reltuples): 테이블의 행 수
- **b_r** (relpages): 디스크 페이지 수
- **V(A, r)** (n_distinct): 열당 고유 값 수
- **히스토그램(Histograms)**: 열당 값 분포
- **상관관계(Correlation)**: 물리적 순서가 논리적 순서와 얼마나 잘 일치하는지 (범위 스캔에 중요)
- **가장 일반적인 값(Most common values, MCV)**: 가장 빈번한 값의 목록과 그 빈도
- **NULL 분율(NULL fraction)**: 열당 NULL 값의 분율

### 8.2 통계 업데이트(Updating Statistics)

데이터가 변경됨에 따라 통계가 오래됩니다. 데이터베이스는 통계를 새로 고칠 명령을 제공합니다:

```sql
-- PostgreSQL
ANALYZE employees;              -- 하나의 테이블에 대한 통계 업데이트
ANALYZE;                         -- 모든 테이블에 대한 통계 업데이트
ALTER TABLE employees SET (autovacuum_analyze_threshold = 50);

-- MySQL
ANALYZE TABLE employees;

-- SQL Server
UPDATE STATISTICS employees;
```

PostgreSQL의 **autovacuum** 프로세스는 충분한 행이 변경되면 자동으로 통계를 업데이트합니다 (기본값: 테이블의 10%).

### 8.3 오래된 통계의 영향(Impact of Stale Statistics)

오래된 통계는 **나쁜 계획**으로 이어집니다:

```
시나리오: 통계가 수집되었을 때 테이블이 1,000개 행을 가졌습니다.
          이제 1,000,000개 행을 가집니다.

최적화기 생각: "작은 테이블, 중첩 루프 조인이 괜찮습니다."
현실: "거대한 테이블, 해시 조인이 1000배 더 빠를 것입니다."
```

이것은 프로덕션 시스템에서 갑작스러운 쿼리 성능 저하의 가장 일반적인 원인 중 하나입니다.

---

## 9. 쿼리 실행 엔진 아키텍처(Query Execution Engine Architecture)

### 9.1 구성 요소(Components)

```
┌──────────────────────────────────────────────────────────┐
│                    Query Executor                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │
│  │ Plan Cache  │  │ Iterator   │  │ Expression         │ │
│  │ (prepared   │  │ Operators  │  │ Evaluator          │ │
│  │  statements)│  │            │  │ (predicates,       │ │
│  │             │  │ - SeqScan  │  │  projections,      │ │
│  │             │  │ - IdxScan  │  │  aggregations)     │ │
│  │             │  │ - NestLoop │  │                    │ │
│  │             │  │ - HashJoin │  │                    │ │
│  │             │  │ - SortMrg  │  │                    │ │
│  │             │  │ - HashAgg  │  │                    │ │
│  │             │  │ - Sort     │  │                    │ │
│  └────────────┘  └────────────┘  └────────────────────┘ │
│                         │                                 │
│              ┌──────────┴──────────┐                     │
│              │  Buffer Manager     │                     │
│              │  (page cache)       │                     │
│              └──────────┬──────────┘                     │
│                         │                                 │
│              ┌──────────┴──────────┐                     │
│              │  Storage Engine     │                     │
│              │  (disk I/O)         │                     │
│              └─────────────────────┘                     │
└──────────────────────────────────────────────────────────┘
```

### 9.2 계획 캐싱(Plan Caching)

파싱과 최적화는 비용이 많이 듭니다. 데이터베이스는 이 작업을 반복하지 않도록 실행 계획을 캐시합니다:

```sql
-- PostgreSQL: 준비된 문은 계획을 캐시합니다
PREPARE find_emp(int) AS
    SELECT * FROM employees WHERE emp_id = $1;

EXECUTE find_emp(42);   -- 첫 실행: 파싱 + 최적화 + 실행
EXECUTE find_emp(99);   -- 후속: 캐시된 계획 재사용
```

**계획 무효화**: 캐시된 계획은 다음의 경우 무효가 됩니다:
- 테이블 구조 변경 (ALTER TABLE)
- 통계 업데이트 (ANALYZE)
- 인덱스 생성 또는 삭제

### 9.3 실행 계획 읽기(Reading Execution Plans)

대부분의 데이터베이스는 실행 계획을 보는 명령을 제공합니다:

```sql
-- PostgreSQL
EXPLAIN ANALYZE
SELECT e.name, d.dept_name
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
WHERE e.salary > 80000;
```

출력 (예):

```
Hash Join  (cost=1.12..25.47 rows=167 width=64) (actual time=0.05..0.31 rows=150 loops=1)
  Hash Cond: (e.dept_id = d.dept_id)
  ->  Seq Scan on employees e  (cost=0.00..22.50 rows=167 width=40) (actual time=0.01..0.15 rows=150 loops=1)
        Filter: (salary > 80000)
        Rows Removed by Filter: 850
  ->  Hash  (cost=1.05..1.05 rows=50 width=28) (actual time=0.02..0.02 rows=50 loops=1)
        Buckets: 1024  Batches: 1  Memory Usage: 12kB
        ->  Seq Scan on departments d  (cost=0.00..1.05 rows=50 width=28) (actual time=0.00..0.01 rows=50 loops=1)
Planning Time: 0.15 ms
Execution Time: 0.38 ms
```

**이 계획 읽기 (아래에서 위로)**:
1. departments의 순차 스캔 (50개 행) → 해시 테이블 빌드 (12 KB)
2. salary > 80000 필터를 사용한 employees의 순차 스캔 (1000개 중 150개 행 통과)
3. dept_id를 사용한 해시 조인
4. 총: 0.38 ms 실행 시간

### 9.4 적응형 쿼리 실행(Adaptive Query Execution)

현대 데이터베이스는 런타임 중에 실행 계획을 조정할 수 있습니다:

- **PostgreSQL**: 준비된 문에 대해 일반 vs 맞춤 계획을 사용합니다. 5번 실행 후, 비교하고 전환할 수 있습니다.
- **Oracle**: 적응형 커서 공유 — 캐시된 계획이 특정 매개변수 값에 대해 성능이 나쁠 때 감지
- **Spark SQL**: 적응형 쿼리 실행(Adaptive Query Execution, AQE) — 실제 파티션 크기에 기반하여 쿼리 중간에 재최적화

---

## 10. 고급 주제(Advanced Topics)

### 10.1 병렬 쿼리 실행(Parallel Query Execution)

현대 데이터베이스는 여러 CPU 코어에 걸쳐 쿼리 실행을 병렬화합니다:

```
              Gather
             /  |  \
     Worker1  Worker2  Worker3
        |        |        |
     Scan(p1) Scan(p2) Scan(p3)  ← 병렬 순차 스캔
```

병렬화 가능한 연산:
- 스캔 (테이블을 범위로 나눔)
- 필터 (각 워커가 자신의 파티션을 필터링)
- 해시 조인 (병렬 빌드 + 병렬 프로브)
- 집계 (워커당 부분 집계, 그 다음 병합)
- 정렬 (병렬 정렬, 그 다음 병합)

### 10.2 열 기반 실행(Columnar Execution)

전통적인 행 저장소: 전체 행을 읽으며, 일부 열만 필요해도.

열 저장소: 각 열을 별도로 저장, 필요한 열만 읽음.

```
행 저장소:                     열 저장소:
[id=1, name=Alice, sal=80K]   id:   [1, 2, 3, ...]
[id=2, name=Bob,   sal=90K]   name: [Alice, Bob, ...]
[id=3, name=Carol, sal=75K]   sal:  [80K, 90K, 75K, ...]
```

분석을 위한 열 저장소의 장점:
- 필요한 열만 읽기 (더 적은 I/O)
- 더 나은 압축 (유사한 값이 함께)
- CPU 친화적 (열 배열에 대한 SIMD 연산)
- 사용처: DuckDB, ClickHouse, Redshift, BigQuery

### 10.3 Just-In-Time (JIT) 컴파일(Just-In-Time (JIT) Compilation)

쿼리 계획을 해석하는 대신 (각 튜플에 대해 가상 함수 호출), 네이티브 머신 코드로 컴파일:

```
전통적 (해석):
  각 튜플에 대해:
    가상 함수 호출: 술어 평가
    가상 함수 호출: 열 투영
    가상 함수 호출: 조인을 위한 해시

JIT 컴파일:
  각 튜플에 대해:
    if tuple.salary > 80000:    // 인라인, 가상 디스패치 없음
      hash = tuple.dept_id % N   // 인라인
      emit(tuple.name, ...)      // 인라인
```

JIT 컴파일은 해석 오버헤드를 제거하며, 복잡한 표현식과 큰 데이터셋에 특히 유익합니다.

PostgreSQL은 표현식 평가 및 튜플 변형을 위한 JIT 컴파일을 (LLVM 사용) 지원합니다.

---

## 11. 연습문제(Exercises)

### 연습문제 1: 비용 계산(Cost Calculation)

주어진:
- employees: n = 10,000, b = 500, emp_id에 인덱스 (B⁺-트리, 높이 3)
- departments: n = 200, b = 10
- 메모리: M = 12 페이지

dept_id로 employees와 departments를 조인하는 비용 (블록 전송)을 계산하세요:

1. 블록 중첩 루프 조인 (employees가 외부)
2. 블록 중첩 루프 조인 (departments가 외부)
3. 해시 조인 (departments가 빌드)

<details>
<summary>해답</summary>

1. **BNLJ (employees 외부)**:
   - 외부 청크: ⌈500 / (12-2)⌉ = ⌈500/10⌉ = 50
   - 비용: 50 × 10 + 500 = 1,000 전송

2. **BNLJ (departments 외부)**:
   - 외부 청크: ⌈10 / (12-2)⌉ = ⌈10/10⌉ = 1
   - 비용: 1 × 500 + 10 = 510 전송

3. **해시 조인 (departments가 빌드)**:
   - departments (10 블록)가 12 페이지 메모리에 맞음
   - 비용: 10 + 500 = 510 전송

해시 조인과 departments-외부 BNLJ는 비슷합니다. 해시 조인은 더 적은 탐색 (2 vs. 2)을 가집니다. 실제로 해시 조인은 더 나은 캐시 동작으로 인해 선호됩니다.
</details>

### 연습문제 2: 선택도 추정(Selectivity Estimation)

주어진: 10,000개 행의 employees 테이블.
- salary: min=30,000, max=150,000, V(salary) = 2,000
- dept_id: V(dept_id) = 50
- city: V(city) = 100

다음이 반환하는 튜플 수를 추정하세요:

1. σ_{salary = 75000}(employees)
2. σ_{salary > 100000}(employees)
3. σ_{dept_id = 5 ∧ city = 'Boston'}(employees)

<details>
<summary>해답</summary>

1. **salary = 75000**: sel = 1/V(salary) = 1/2000. 결과: 10,000/2,000 = **5 튜플**

2. **salary > 100000**: sel = (150,000 - 100,000)/(150,000 - 30,000) = 50,000/120,000 ≈ 0.417. 결과: 10,000 × 0.417 ≈ **4,167 튜플**

3. **dept_id = 5 AND city = 'Boston'** (독립성 가정):
   - sel(dept_id = 5) = 1/50
   - sel(city = 'Boston') = 1/100
   - 결합: (1/50) × (1/100) = 1/5,000
   - 결과: 10,000 / 5,000 = **2 튜플**
</details>

### 연습문제 3: 휴리스틱 최적화(Heuristic Optimization)

다음 쿼리 트리를 휴리스틱 규칙을 사용하여 최적화하세요:

```sql
SELECT p.product_name, c.category_name
FROM products p, categories c, order_items oi
WHERE p.category_id = c.category_id
  AND p.product_id = oi.product_id
  AND oi.quantity > 10
  AND c.category_name = 'Electronics';
```

초기 및 최적화된 쿼리 트리를 그리세요.

<details>
<summary>해답</summary>

**초기 (최적화되지 않은) 트리:**

```
π_{product_name, category_name}
    │
σ_{p.cat_id=c.cat_id ∧ p.prod_id=oi.prod_id ∧ oi.qty>10 ∧ c.cat_name='Electronics'}
    │
    ×  (카티전 곱)
   / \
  ×   oi
 / \
p   c
```

**최적화된 트리 (선택 푸시 다운, 카티전 곱 대신 조인 사용):**

```
π_{product_name, category_name}
    │
    ⋈_{p.cat_id = c.cat_id}
   / \
  ⋈_{p.prod_id = oi.prod_id}    σ_{cat_name='Electronics'}(c)
 / \
p   σ_{qty > 10}(oi)
```

**적용된 최적화:**
1. 연언 선택 분해
2. σ_{qty > 10}을 order_items로 푸시 (조인 전)
3. σ_{cat_name = 'Electronics'}을 categories로 푸시 (조인 전)
4. 카티전 곱을 목표 조인으로 교체
5. 일찍 투영 (명확성을 위해 표시하지 않았지만, 필요한 열만 통과)

핵심 이득: categories가 ~1개 행 ('Electronics')으로 필터링되고, order_items가 하위 집합 (qty > 10)으로 필터링된 후, 조인이 발생합니다.
</details>

### 연습문제 4: 조인 알고리즘 선택(Join Algorithm Selection)

각 시나리오에 대해 최적화기가 어떤 조인 알고리즘을 선택할 것 같은지 결정하세요.

1. 100개 행 룩업 테이블을 1000만 개 행 팩트 테이블과 조인. 팩트 테이블의 조인 열에 인덱스 존재.
2. 두 100만 개 행 테이블을 조인, 둘 다 정렬되지 않음, 충분한 메모리 (1GB 버퍼 풀).
3. 두 100만 개 행 테이블을 범위 조건으로 조인 (r.date BETWEEN s.start_date AND s.end_date).
4. 두 테이블을 조인하는데 둘 다 이미 조인 열로 정렬되어 있음.

<details>
<summary>해답</summary>

1. **인덱스 중첩 루프 조인.** 룩업 테이블 (100개 행)이 외부; 각 행에 대해 팩트 테이블의 인덱스를 사용하여 일치를 찾습니다. 비용: 100개 인덱스 조회, 각 O(log n). 1000만 개 행을 스캔하는 것보다 훨씬 빠릅니다.

2. **해시 조인.** 충분한 메모리로, 한 테이블의 해시 테이블이 메모리에 완전히 맞습니다. 비용: 두 테이블을 한 번씩 읽기 (최적). 인덱스 불필요, 정렬 불필요.

3. **정렬-병합 조인** 또는 **블록 중첩 루프 조인.** 해시 조인은 범위 조건에 작동하지 않습니다 (범위를 해시할 수 없음). 날짜 열에 대한 정렬-병합은 효율적인 범위 매칭을 허용합니다. 블록 NLJ는 정렬이 너무 비싸면 대안입니다.

4. **정렬-병합 조인 (정렬 단계 건너뛰기).** 두 테이블이 이미 정렬되어 있으므로, 병합 단계 비용은 b_r + b_s입니다 — 각 테이블을 통한 단일 패스. 이것은 최적입니다.
</details>

### 연습문제 5: 실행 계획 읽기(Reading Execution Plans)

이 PostgreSQL EXPLAIN 출력이 주어졌을 때, 아래 질문에 답하세요:

```
Nested Loop  (cost=0.29..8.33 rows=1 width=64)
  ->  Index Scan using idx_emp_id on employees  (cost=0.29..4.30 rows=1 width=40)
        Index Cond: (emp_id = 42)
  ->  Seq Scan on departments  (cost=0.00..1.62 rows=1 width=24)
        Filter: (dept_id = employees.dept_id)
        Rows Removed by Filter: 49
```

1. 어떤 조인 알고리즘이 사용되나요?
2. 어떤 테이블이 외부 (드라이빙) 테이블인가요?
3. 최적화기가 왜 employees에 인덱스 스캔을 사용하나요?
4. 왜 departments에 순차 스캔이 사용되나요?
5. 추정 총 비용은 얼마인가요?

<details>
<summary>해답</summary>

1. **중첩 루프 조인.**

2. **employees가 외부 테이블입니다** (중첩 루프 아래에 먼저 나열됨). 루프를 구동합니다.

3. **emp_id = 42가 매우 선택적인 동등 술어이기 때문입니다.** emp_id의 인덱스는 정확히 1개 행 (rows=1)을 찾습니다. 전체 employees 테이블을 읽는 것은 낭비입니다.

4. **departments가 작기 때문입니다 (50개 행, ~2 페이지).** 각 외부 행 (이 경우 1개만)에 대해, 전체 departments 테이블이 스캔됩니다. 외부 행이 1개만 있으므로, 순차 스캔은 한 번만 실행됩니다. 아주 작은 테이블에 대한 단일 프로브의 경우 인덱스 조회가 더 빠르지 않을 수 있습니다.

5. **총 추정 비용: 8.33** (PostgreSQL의 비용 단위로, 1.0 ≈ 순차 페이지 읽기). 이는 매우 저렴합니다 — 본질적으로 1개 인덱스 조회 + 1개 작은 테이블 스캔.
</details>

### 연습문제 6: 동등 규칙(Equivalence Rules)

동등 규칙을 사용하여 이 두 표현식이 동일한 결과를 생성함을 보이세요:

**표현식 A:**
```
σ_{dept='CS'}(employees ⋈ departments)
```

**표현식 B:**
```
employees ⋈ σ_{dept='CS'}(departments)
```

어느 것이 더 효율적이고 왜 그런가요?

<details>
<summary>해답</summary>

**동등성 증명:**

규칙 6 (조인을 통한 선택 푸시)에 의해, 술어 dept='CS'가 departments의 속성만 포함하면:

```
σ_{dept='CS'}(employees ⋈ departments) = employees ⋈ σ_{dept='CS'}(departments)
```

이것이 유효한 이유:
1. 조인은 모든 일치하는 (employee, department) 쌍을 생성합니다
2. 선택은 그 다음 dept='CS'로 필터링합니다
3. 동등하게, 먼저 departments를 필터링하여 CS 부서만 얻은 다음 조인할 수 있습니다

**표현식 B가 더 효율적**입니다:
- 표현식 A: 모든 직원을 모든 부서와 조인 (10,000 × 50 조합 평가), 그 다음 필터링. 조인은 10,000개 행을 생성한 다음, 필터가 ~200개만 유지 (50개 중 1/50이 CS에 있다면).
- 표현식 B: 먼저 departments를 필터링 (50 → 1개 행), 그 다음 조인. 조인은 1개 부서 행과만 직원을 일치시키면 됩니다. 훨씬 더 적은 작업.

중간 결과의 크기:
- A: 10,000 중간 행 → 필터 → 200 최종 행
- B: 1 중간 행 × employees → 200 최종 행 직접
</details>

### 연습문제 7: 비용 기반 최적화(Cost-Based Optimization)

세 테이블과 그 통계가 주어졌습니다:

```
orders (o):     n = 100,000,  b = 5,000
customers (c):  n = 10,000,   b = 500
products (p):   n = 1,000,    b = 50
```

조인 술어: o.cust_id = c.cust_id AND o.prod_id = p.prod_id

해시 조인과 M = 100 페이지를 가정합니다. 이 두 조인 순서를 비교하세요:

**계획 A**: (orders ⋈ customers) ⋈ products
**계획 B**: (orders ⋈ products) ⋈ customers

<details>
<summary>해답</summary>

**계획 A: (orders ⋈ customers) ⋈ products**

1단계: orders ⋈ customers (해시 조인, customers가 빌드)
- 빌드: 500 블록 (customers가 100 페이지에 맞나? 아니요, 500 > 100. Grace 해시 조인 필요.)
- Grace 해시 조인 비용: 3 × (5,000 + 500) = 16,500 전송
- 결과 크기: 100,000개 행 (각 주문이 하나의 고객을 가짐)
- 결과 블록: ~5,000 (orders와 유사)

2단계: result ⋈ products (해시 조인, products가 빌드)
- 빌드: 50 블록 (products가 100 페이지에 맞음. 인메모리 해시 조인.)
- 비용: 5,000 + 50 = 5,050 전송
- 계획 A 총: 16,500 + 5,050 = **21,550 전송**

**계획 B: (orders ⋈ products) ⋈ customers**

1단계: orders ⋈ products (해시 조인, products가 빌드)
- 빌드: 50 블록 (products가 메모리에 맞음!)
- 인메모리 해시 조인 비용: 5,000 + 50 = 5,050 전송
- 결과 크기: 100,000개 행 (각 주문이 하나의 제품을 가짐)
- 결과 블록: ~5,000

2단계: result ⋈ customers (해시 조인, customers가 빌드)
- 빌드: 500 블록 (100 페이지에 맞지 않음. Grace 해시 조인.)
- 비용: 3 × (5,000 + 500) = 16,500 전송
- 계획 B 총: 5,050 + 16,500 = **21,550 전송**

흥미롭게도, 총 전송 수는 동일합니다! 하지만 계획 B가 약간 더 나은 이유:
1. 1단계가 인메모리 해시 조인 사용 (더 적은 탐색, 더 나은 캐시)
2. 1단계의 중간 결과가 2단계로 파이프라인될 수 있습니다

더 똑똑한 접근법: 두 작은 테이블 (products: 50, customers: 500)에 해시 테이블을 빌드한 다음, orders를 한 번 스캔:

**계획 C**: orders를 한 번 스캔, 두 해시 테이블 모두 프로브
- 비용: 5,000 + 500 + 50 = 5,550 전송 (두 해시 테이블이 메모리에 맞으면 — 550 페이지가 필요하며, M=100을 초과)

M=600이면, 계획 C가 최적일 것입니다.
</details>

---

## 12. 요약(Summary)

| 개념 | 핵심 포인트 |
|---------|-----------|
| **쿼리 처리 파이프라인(Query Processing Pipeline)** | 파싱 → 최적화 → 실행 |
| **반복자 모델(Iterator Model)** | open/next/close 인터페이스; 튜플이 연산자 트리를 통해 위로 흐름 |
| **파이프라이닝(Pipelining)** | 중간 결과 구체화 회피 |
| **선택 알고리즘(Selection algorithms)** | 선형 스캔, 이진 검색, 인덱스 스캔; 선택은 선택도에 의존 |
| **조인 알고리즘(Join algorithms)** | NLJ, 블록 NLJ, 인덱스 NLJ, 정렬-병합, 해시 조인 |
| **해시 조인(Hash join)** | 빌드 릴레이션이 메모리에 맞을 때 최적: 비용 = b_r + b_s |
| **정렬-병합 조인(Sort-merge join)** | 사전 정렬된 데이터와 범위 조인에 최선 |
| **휴리스틱 최적화(Heuristic optimization)** | 선택 푸시 다운, 투영 푸시 다운, 조인 재정렬 |
| **비용 기반 최적화(Cost-based optimization)** | 통계를 사용하여 비용 추정; 조인 순서를 위한 동적 프로그래밍 |
| **통계(Statistics)** | 히스토그램, 고유 값, 상관관계 — 좋은 계획에 필수적 |
| **계획 캐싱(Plan caching)** | 준비된 문에 대한 반복 파싱/최적화 회피 |

쿼리 처리는 데이터베이스 이론이 시스템 엔지니어링과 만나는 곳입니다. 이러한 개념을 이해하면 더 나은 쿼리를 작성하고, 적절한 인덱스를 생성하고 (다음 레슨에서 다룸), 실행 계획을 읽어 성능 문제를 진단하는 데 도움이 됩니다.

---

**이전**: [07_Advanced_Normalization.md](./07_Advanced_Normalization.md) | **다음**: [09_Indexing.md](./09_Indexing.md)

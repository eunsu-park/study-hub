# 트랜잭션

**이전**: [함수와 프로시저](./10_Functions_and_Procedures.md) | **다음**: [트리거](./12_Triggers.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 트랜잭션(Transaction)의 개념과 원자성(Atomicity)이 데이터 무결성에 중요한 이유를 설명할 수 있습니다
2. ACID 네 가지 속성과 신뢰할 수 있는 데이터베이스 운영에서의 역할을 설명할 수 있습니다
3. BEGIN, COMMIT, ROLLBACK을 사용하여 트랜잭션 경계를 제어할 수 있습니다
4. 트랜잭션 내 부분 롤백을 위한 SAVEPOINT를 생성하고 관리할 수 있습니다
5. SQL의 네 가지 격리 수준(Isolation Level)을 비교하고 각각이 방지하는 동시성 이상 현상을 파악할 수 있습니다
6. 더티 리드(Dirty Read), 비반복 읽기(Non-repeatable Read), 팬텀 리드(Phantom Read) 같은 동시성 문제를 진단할 수 있습니다
7. 행 수준 잠금(Row-level Lock)과 테이블 수준 잠금(Table-level Lock)을 적용하여 임계 구역을 보호할 수 있습니다
8. 일관된 잠금 순서를 사용하여 교착상태(Deadlock)를 감지하고 방지할 수 있습니다

---

트랜잭션(Transaction)은 모든 신뢰할 수 있는 데이터베이스 애플리케이션의 근간입니다. 은행 계좌 간 송금, 이커머스 시스템의 주문 처리, IoT 기기의 센서 데이터 기록 등 어떤 작업이든 트랜잭션은 연산 그룹이 완전히 성공하거나 완전히 실패하도록 보장하여 데이터베이스를 일관된 상태로 유지합니다. 트랜잭션을 마스터하는 것은 동시 접근과 예기치 못한 장애 상황에서도 올바르게 동작하는 애플리케이션을 구축하는 데 필수적입니다.

## 1. 트랜잭션 개념

트랜잭션은 하나의 논리적 작업 단위를 구성하는 연산들의 집합입니다.

```
┌──────────────────────────────────────────────────────────┐
│                     계좌 이체 트랜잭션                    │
├──────────────────────────────────────────────────────────┤
│  1. A 계좌에서 10만원 차감                               │
│  2. B 계좌에 10만원 추가                                 │
│  → 둘 다 성공하거나, 둘 다 실패해야 함                  │
└──────────────────────────────────────────────────────────┘
```

---

## 2. ACID 속성

| 속성 | 영문 | 설명 |
|------|------|------|
| 원자성 | Atomicity | 전부 성공 또는 전부 실패 |
| 일관성 | Consistency | 트랜잭션 전후로 데이터 일관성 유지 |
| 격리성 | Isolation | 동시 실행 트랜잭션 간 간섭 방지 |
| 지속성 | Durability | 완료된 트랜잭션은 영구 저장 |

---

## 3. 기본 트랜잭션 명령

### BEGIN / COMMIT / ROLLBACK

```sql
-- 트랜잭션 시작
BEGIN;
-- 또는
START TRANSACTION;

-- 작업 수행
UPDATE accounts SET balance = balance - 100000 WHERE id = 1;
UPDATE accounts SET balance = balance + 100000 WHERE id = 2;

-- 커밋 (변경사항 확정)
COMMIT;

-- 또는 롤백 (변경사항 취소)
ROLLBACK;
```

### 실습 예제

```sql
-- 테이블 생성
CREATE TABLE accounts (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    balance NUMERIC(12, 2) DEFAULT 0
);

INSERT INTO accounts (name, balance) VALUES
('김철수', 1000000),
('이영희', 500000);

-- 이체 트랜잭션
BEGIN;

UPDATE accounts SET balance = balance - 100000 WHERE name = '김철수';
UPDATE accounts SET balance = balance + 100000 WHERE name = '이영희';

-- 확인
SELECT * FROM accounts;

-- 확정 또는 취소
COMMIT;  -- 또는 ROLLBACK;
```

---

## 4. 자동 커밋 (Autocommit)

psql은 기본적으로 자동 커밋 모드입니다.

```sql
-- 자동 커밋 모드에서 각 문장은 개별 트랜잭션
INSERT INTO accounts (name, balance) VALUES ('박민수', 300000);
-- 즉시 커밋됨

-- 자동 커밋 비활성화
\set AUTOCOMMIT off

-- 이제 명시적 COMMIT 필요
INSERT INTO accounts (name, balance) VALUES ('최지영', 400000);
COMMIT;

-- 자동 커밋 다시 활성화
\set AUTOCOMMIT on
```

---

## 5. SAVEPOINT

트랜잭션 내에서 부분 롤백 지점을 만듭니다.

```sql
BEGIN;

INSERT INTO accounts (name, balance) VALUES ('신규1', 100000);
SAVEPOINT sp1;

INSERT INTO accounts (name, balance) VALUES ('신규2', 200000);
SAVEPOINT sp2;

INSERT INTO accounts (name, balance) VALUES ('신규3', 300000);

-- sp2로 롤백 (신규3만 취소)
ROLLBACK TO SAVEPOINT sp2;

-- sp1으로 롤백 (신규2, 신규3 취소)
ROLLBACK TO SAVEPOINT sp1;

-- 전체 커밋 (신규1만 저장)
COMMIT;
```

### SAVEPOINT 관리

```sql
-- SAVEPOINT 해제
RELEASE SAVEPOINT sp1;

-- SAVEPOINT 덮어쓰기 (같은 이름으로 재생성)
SAVEPOINT mypoint;
-- ... 작업 ...
SAVEPOINT mypoint;  -- 새 지점으로 대체
```

---

## 6. 트랜잭션 격리 수준

동시에 실행되는 트랜잭션 간의 격리 정도를 결정합니다.

### 격리 수준 종류

| 수준 | Dirty Read | Non-repeatable Read | Phantom Read |
|------|------------|---------------------|--------------|
| READ UNCOMMITTED | 가능 | 가능 | 가능 |
| READ COMMITTED | 방지 | 가능 | 가능 |
| REPEATABLE READ | 방지 | 방지 | 가능* |
| SERIALIZABLE | 방지 | 방지 | 방지 |

*PostgreSQL의 REPEATABLE READ는 Phantom Read도 방지

### PostgreSQL 기본값

PostgreSQL의 기본 격리 수준은 **READ COMMITTED**입니다.

### 격리 수준 설정

```sql
-- 트랜잭션별 설정
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
-- 또는
BEGIN;
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;

-- 세션 전체 설정
SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL SERIALIZABLE;

-- 현재 격리 수준 확인
SHOW transaction_isolation;
```

---

## 7. 동시성 문제

### Dirty Read (더티 리드)

커밋되지 않은 데이터를 읽는 문제 → PostgreSQL에서는 발생하지 않음

### Non-repeatable Read (비반복 읽기)

같은 트랜잭션 내에서 같은 데이터를 두 번 읽었을 때 다른 값이 나오는 문제

```sql
-- 트랜잭션 A
BEGIN;
SELECT balance FROM accounts WHERE id = 1;  -- 1000000

-- 트랜잭션 B가 업데이트하고 커밋
-- UPDATE accounts SET balance = 900000 WHERE id = 1; COMMIT;

SELECT balance FROM accounts WHERE id = 1;  -- 900000 (다른 값!)
COMMIT;
```

### Phantom Read (팬텀 리드)

같은 조건으로 조회했을 때 행의 개수가 달라지는 문제

```sql
-- 트랜잭션 A
BEGIN;
SELECT COUNT(*) FROM accounts WHERE balance > 500000;  -- 2개

-- 트랜잭션 B가 새 행 삽입하고 커밋
-- INSERT INTO accounts VALUES (...); COMMIT;

SELECT COUNT(*) FROM accounts WHERE balance > 500000;  -- 3개 (유령 행!)
COMMIT;
```

---

## 8. 격리 수준별 동작

### READ COMMITTED (기본)

```sql
BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;

-- 다른 트랜잭션이 커밋한 변경사항을 즉시 볼 수 있음
SELECT * FROM accounts;  -- 최신 커밋된 데이터

COMMIT;
```

### REPEATABLE READ

```sql
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;

-- 트랜잭션 시작 시점의 스냅샷을 봄
SELECT * FROM accounts;

-- 다른 트랜잭션이 커밋해도 같은 결과
SELECT * FROM accounts;  -- 동일

COMMIT;
```

### SERIALIZABLE

```sql
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;

-- 가장 엄격한 격리
-- 직렬화 충돌 시 오류 발생 가능
SELECT * FROM accounts WHERE balance > 500000;
UPDATE accounts SET balance = balance + 10000 WHERE id = 1;

COMMIT;
-- ERROR: could not serialize access due to concurrent update
-- (다른 트랜잭션과 충돌 시)
```

---

## 9. 잠금 (Locking)

### 행 수준 잠금

```sql
-- SELECT FOR UPDATE: 조회하면서 잠금
BEGIN;
SELECT * FROM accounts WHERE id = 1 FOR UPDATE;
-- 다른 트랜잭션은 이 행을 수정/삭제 불가

UPDATE accounts SET balance = balance - 100000 WHERE id = 1;
COMMIT;

-- SELECT FOR SHARE: 공유 잠금 (읽기는 허용, 쓰기 불가)
SELECT * FROM accounts WHERE id = 1 FOR SHARE;
```

### 잠금 옵션

```sql
-- 대기하지 않고 실패
SELECT * FROM accounts WHERE id = 1 FOR UPDATE NOWAIT;

-- 지정된 시간만 대기
SELECT * FROM accounts WHERE id = 1 FOR UPDATE SKIP LOCKED;
```

### 테이블 수준 잠금

```sql
-- 명시적 테이블 잠금 (드물게 사용)
LOCK TABLE accounts IN EXCLUSIVE MODE;
```

---

## 10. 교착상태 (Deadlock)

두 트랜잭션이 서로의 잠금을 기다리는 상태

```sql
-- 트랜잭션 A
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
-- id=1 잠금

-- 트랜잭션 B
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 2;
-- id=2 잠금

-- 트랜잭션 A
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
-- id=2 대기...

-- 트랜잭션 B
UPDATE accounts SET balance = balance + 100 WHERE id = 1;
-- id=1 대기... → 교착상태!

-- PostgreSQL이 자동으로 한 트랜잭션을 중단시킴
-- ERROR: deadlock detected
```

### 교착상태 방지

```sql
-- 항상 같은 순서로 잠금 획득
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;  -- 항상 작은 id 먼저
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
```

---

## 10.5 교착상태 실습 및 SERIALIZABLE 격리(Deadlock Lab and SERIALIZABLE Isolation)

앞 섹션에서 교착상태와 격리 수준을 소개했습니다. 이 섹션에서는 더 깊이 들어갑니다: 교착상태를 의도적으로 만들고, 감지하고, 디버깅하는 방법을 배우며, PostgreSQL의 SERIALIZABLE 격리 수준이 직렬화 가능 스냅샷 격리(Serializable Snapshot Isolation, SSI)를 사용하여 REPEATABLE READ가 잡지 못하는 이상 현상을 방지하는 방법을 이해합니다.

### 교착상태 실습: 단계별로 교착상태 재현하기

같은 데이터베이스에 연결된 **두 개의** psql 터미널을 엽니다. 의도적으로 교착상태를 만들겠습니다.

```sql
-- 설정: 테스트 테이블 생성
CREATE TABLE deadlock_lab (
    id INTEGER PRIMARY KEY,
    value TEXT,
    counter INTEGER DEFAULT 0
);

INSERT INTO deadlock_lab VALUES (1, 'Alice', 100), (2, 'Bob', 200);
```

**터미널 1:**

```sql
-- 단계 1: 트랜잭션 시작 후 id=1 행 잠금
-- Why: id=1에 대한 행 수준 배타적 잠금(exclusive lock)을 획득
BEGIN;
UPDATE deadlock_lab SET counter = counter + 10 WHERE id = 1;
-- id=1이 터미널 1에 의해 잠김
-- 아직 COMMIT하지 않음 — 터미널 2가 id=2를 잠글 때까지 대기
```

**터미널 2:**

```sql
-- 단계 2: 트랜잭션 시작 후 id=2 행 잠금
-- Why: id=2에 대한 행 수준 배타적 잠금을 획득
BEGIN;
UPDATE deadlock_lab SET counter = counter + 20 WHERE id = 2;
-- id=2가 터미널 2에 의해 잠김
```

**터미널 1:**

```sql
-- 단계 3: id=2 잠금 시도 (터미널 2가 보유 중)
-- Why: 터미널 1이 터미널 2가 id=2를 해제하기를 기다림
UPDATE deadlock_lab SET counter = counter + 10 WHERE id = 2;
-- 차단됨(BLOCK) — 터미널 2의 id=2 잠금을 기다리는 중
```

**터미널 2:**

```sql
-- 단계 4: id=1 잠금 시도 (터미널 1이 보유 중)
-- Why: 터미널 2가 터미널 1이 id=1을 해제하기를 기다림
-- 순환이 생성됨: T1이 T2를 기다림, T2가 T1을 기다림 → 교착상태!
UPDATE deadlock_lab SET counter = counter + 20 WHERE id = 1;

-- ~1초 이내에 PostgreSQL이 교착상태 감지:
-- ERROR:  deadlock detected
-- DETAIL: Process 12345 waits for ShareLock on transaction 67890;
--         blocked by process 12346.
--         Process 12346 waits for ShareLock on transaction 67891;
--         blocked by process 12345.
-- HINT:  See server log for query details.

-- PostgreSQL이 이 트랜잭션(희생자)을 중단시킴
ROLLBACK;  -- 정리
```

**터미널 1:**

```sql
-- 터미널 1의 id=2 UPDATE가 이제 성공 (터미널 2가 잠금을 해제했으므로)
COMMIT;
```

### PostgreSQL 교착상태 감지 내부 구조

```
┌──────────────────────────────────────────────────────────────────────┐
│              PostgreSQL 교착상태 감지                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PostgreSQL은 대기-그래프(wait-for graph)를 사용하여 교착상태를 감지: │
│                                                                      │
│      T1 ──기다림──► T2                                               │
│      ▲                │                                              │
│      └──기다림────────┘    ← 순환 = 교착상태                          │
│                                                                      │
│  감지 트리거:                                                        │
│  - 프로세스가 deadlock_timeout(1초)보다 오래 잠금을 기다림            │
│  - PostgreSQL이 대기-그래프를 구성                                    │
│  - 순환이 발견되면 하나의 트랜잭션을 중단 (희생자)                   │
│                                                                      │
│  주요 설정:                                                          │
│  - deadlock_timeout = 1s  (기본값; DL 확인 전 대기 시간)             │
│  - lock_timeout = 0       (0 = 영구 대기; 대기 제한 설정 가능)       │
│  - log_lock_waits = off   (on으로 설정하면 DL 타임아웃 초과 대기 기록)│
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 교착상태 감지 설정

```sql
-- 현재 설정 확인
SHOW deadlock_timeout;     -- 기본값: 1s
SHOW lock_timeout;         -- 기본값: 0 (제한 없음)
SHOW log_lock_waits;       -- 기본값: off

-- lock_timeout 설정으로 영구 대기 대신 빠르게 실패
-- Why: 웹 애플리케이션에서 잠금을 30초 이상 기다리는 것은
-- 즉시 실패 후 재시도하는 것보다 더 나쁨
SET lock_timeout = '5s';

-- log_lock_waits 설정으로 잠금 경합 모니터링
-- Why: 어떤 쿼리가 잠금에 의해 자주 차단되는지 파악
ALTER SYSTEM SET log_lock_waits = on;
SELECT pg_reload_conf();

-- 세션별 타임아웃 (테스트에 유용)
BEGIN;
SET LOCAL lock_timeout = '2s';
-- 2초 이내에 잠금을 획득하지 못하면 다음과 같이 실패:
-- ERROR: canceling statement due to lock timeout
SELECT * FROM deadlock_lab WHERE id = 1 FOR UPDATE;
COMMIT;
```

### 교착상태 방지: 일관된 잠금 순서

```sql
-- 나쁜 예: 일관되지 않은 잠금 순서가 교착상태를 유발
-- 트랜잭션 A: UPDATE ... WHERE id = 1; UPDATE ... WHERE id = 2;
-- 트랜잭션 B: UPDATE ... WHERE id = 2; UPDATE ... WHERE id = 1;

-- 좋은 예: 항상 같은 순서로 잠금 (예: id 오름차순)
-- Why: 모든 트랜잭션이 같은 순서로 행을 잠그면,
-- 대기-그래프에서 순환이 절대 형성될 수 없음

-- 일관된 잠금 순서를 사용한 이체 함수
CREATE OR REPLACE FUNCTION safe_transfer(
    from_id INTEGER,
    to_id INTEGER,
    amount INTEGER
) RETURNS VOID AS $$
DECLARE
    -- Why: 잠금을 획득하기 전에 잠금 순서를 결정
    first_id  INTEGER := LEAST(from_id, to_id);
    second_id INTEGER := GREATEST(from_id, to_id);
BEGIN
    -- 이체 방향과 관계없이 항상 작은 id를 먼저 잠금
    -- Why: 모든 트랜잭션이 같은 순서로 잠금을 획득하도록 보장
    PERFORM 1 FROM deadlock_lab WHERE id = first_id FOR UPDATE;
    PERFORM 1 FROM deadlock_lab WHERE id = second_id FOR UPDATE;

    -- 이제 안전하게 이체 수행
    UPDATE deadlock_lab SET counter = counter - amount WHERE id = from_id;
    UPDATE deadlock_lab SET counter = counter + amount WHERE id = to_id;
END;
$$ LANGUAGE plpgsql;

-- 두 호출 모두 id=1을 먼저 잠그고, 그 다음 id=2 — 교착상태 불가능
SELECT safe_transfer(1, 2, 50);  -- 잠금: 1 → 2
SELECT safe_transfer(2, 1, 30);  -- 잠금: 1 → 2 (2 → 1이 아님!)
```

### SERIALIZABLE 격리: 직렬화 가능 스냅샷 격리(SSI)

PostgreSQL은 SERIALIZABLE 격리 수준을 전통적인 2단계 잠금(2PL) 없이 직렬화 충돌을 감지하는 **직렬화 가능 스냅샷 격리(Serializable Snapshot Isolation, SSI)** 기법으로 구현합니다.

```
┌──────────────────────────────────────────────────────────────────────┐
│           SSI vs 전통적 2PL                                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  전통적 2PL (SQL Server, MySQL/InnoDB 사용):                         │
│  - 읽기 잠금(공유)과 쓰기 잠금(배타적) 획득                           │
│  - 읽기가 쓰기를 차단, 쓰기가 읽기를 차단                             │
│  - 직렬화 가능성을 보장하지만 동시성 감소                             │
│  - 읽기 집중 워크로드에서 높은 잠금 경합                              │
│                                                                      │
│  PostgreSQL SSI:                                                     │
│  - MVCC 스냅샷 사용 (읽기가 쓰기를 절대 차단하지 않음)                │
│  - 트랜잭션 간 읽기-쓰기 의존성 추적                                  │
│  - "위험한 구조" 감지 (잠재적 직렬화 이상)                            │
│  - 충돌 감지 시 하나의 트랜잭션 중단                                  │
│  - 읽기 집중 워크로드에 훨씬 유리                                     │
│                                                                      │
│  트레이드오프: SSI는 오탐(false positive)이 발생할 수 있음             │
│  (실제로는 안전한 트랜잭션을 중단), 하지만 실제 이상을                │
│  놓치지는 않음.                                                       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 쓰기 편향 이상(Write Skew Anomaly): REPEATABLE READ가 충분하지 않은 이유

쓰기 편향(Write Skew)은 REPEATABLE READ가 방지할 수 없지만 SERIALIZABLE은 방지할 수 있는 동시성 이상입니다. 대표 예시: 병원에는 항상 최소 한 명의 당직 의사가 있어야 합니다.

```sql
-- 설정
CREATE TABLE doctors_on_call (
    id INTEGER PRIMARY KEY,
    name TEXT,
    on_call BOOLEAN DEFAULT true
);

INSERT INTO doctors_on_call VALUES (1, '김의사', true), (2, '이의사', true);
```

**REPEATABLE READ에서 (쓰기 편향 발생):**

```sql
-- 터미널 1 (김의사가 당직 해제 요청):
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
-- 확인: 다른 당직 의사가 최소 한 명 있는가?
-- Why: 이 스냅샷에서는 김의사와 이의사 모두 당직 중
SELECT COUNT(*) FROM doctors_on_call WHERE on_call = true AND id != 1;
-- 결과: 1 (이의사가 당직 중) — 당직 해제해도 안전

-- 터미널 2 (이의사가 당직 해제 요청 — 동시에):
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
-- 확인: 다른 당직 의사가 최소 한 명 있는가?
-- Why: 이 스냅샷에서는 둘 다 여전히 당직 중
SELECT COUNT(*) FROM doctors_on_call WHERE on_call = true AND id != 2;
-- 결과: 1 (김의사가 당직 중) — 당직 해제해도 안전

-- 터미널 1:
UPDATE doctors_on_call SET on_call = false WHERE id = 1;
COMMIT;  -- 성공!

-- 터미널 2:
UPDATE doctors_on_call SET on_call = false WHERE id = 2;
COMMIT;  -- 역시 성공!

-- 문제: 두 의사 모두 당직 해제됨 — 아무도 근무하지 않음!
-- Why: 각 트랜잭션은 다른 의사가 당직 중인 스냅샷을 봄.
-- REPEATABLE READ는 이 교차 트랜잭션 의존성을 감지할 수 없음.
```

**SERIALIZABLE에서 (쓰기 편향 방지):**

```sql
-- 터미널 1:
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SELECT COUNT(*) FROM doctors_on_call WHERE on_call = true AND id != 1;
-- 결과: 1

-- 터미널 2:
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SELECT COUNT(*) FROM doctors_on_call WHERE on_call = true AND id != 2;
-- 결과: 1

-- 터미널 1:
UPDATE doctors_on_call SET on_call = false WHERE id = 1;
COMMIT;  -- 성공

-- 터미널 2:
UPDATE doctors_on_call SET on_call = false WHERE id = 2;
COMMIT;
-- ERROR: could not serialize access due to read/write dependencies
--        among transactions
-- DETAIL: Reason code: Canceled on identification as a pivot, ...

-- Why: PostgreSQL의 SSI가 T2의 읽기(당직 상태)가
-- T1의 쓰기에 의해 무효화되었음을 감지. 시스템이 올바르게
-- 쓰기 편향 이상을 방지함.
```

### SERIALIZABLE 사용 시점

```sql
-- SERIALIZABLE은 오버헤드를 추가함. 다음 경우에 사용:
-- 1. 정확성 > 처리량 (금융 시스템, 재고)
-- 2. 비즈니스 규칙이 여러 행에 걸침 (당직 예시처럼)
-- 3. 수동 잠금 없이 진정한 직렬화 가능성 보장이 필요

-- SERIALIZABLE 사용 시 최선의 방법:
-- - 트랜잭션을 짧게 유지
-- - 직렬화 실패 시 재시도 준비
-- - 애플리케이션 코드에서 재시도 루프 사용:

-- 애플리케이션 의사 코드:
-- max_retries = 3
-- for attempt in range(max_retries):
--     try:
--         BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE
--         ... 쿼리들 ...
--         COMMIT
--         break  # 성공
--     except SerializationFailure:
--         ROLLBACK
--         # 백오프와 함께 재시도
--         continue
```

### 잠금 경합 및 교착상태 모니터링

```sql
-- 현재 모든 잠금과 대기 상태 조회
-- Why: 프로덕션에서 잠금 경합을 디버깅하는 데 필수적
SELECT
    l.pid,
    l.locktype,
    l.mode,
    l.granted,
    a.query,
    a.state,
    age(now(), a.query_start) AS query_age
FROM pg_locks l
JOIN pg_stat_activity a ON l.pid = a.pid
WHERE NOT l.granted
ORDER BY a.query_start;

-- 교착상태 통계 조회 (PostgreSQL 14+)
-- Why: 교착상태가 얼마나 자주 발생하는지 파악하는 데 도움
SELECT
    datname,
    deadlocks,
    conflicts
FROM pg_stat_database
WHERE datname = current_database();

-- 장기 실행 트랜잭션 확인 (잠재적 잠금 보유자)
-- Why: 잊혀진 열린 트랜잭션이 잠금을 무기한 보유할 수 있음
SELECT
    pid,
    now() - xact_start AS transaction_duration,
    state,
    query
FROM pg_stat_activity
WHERE xact_start IS NOT NULL
  AND state != 'idle'
ORDER BY xact_start;
```

---

## 11. 실습 예제

### 실습 1: 기본 트랜잭션

```sql
-- 계좌 이체
CREATE OR REPLACE PROCEDURE transfer(
    from_id INTEGER,
    to_id INTEGER,
    amount NUMERIC
)
AS $$
BEGIN
    -- 출금
    UPDATE accounts SET balance = balance - amount WHERE id = from_id;

    -- 잔액 확인
    IF (SELECT balance FROM accounts WHERE id = from_id) < 0 THEN
        RAISE EXCEPTION '잔액 부족';
    END IF;

    -- 입금
    UPDATE accounts SET balance = balance + amount WHERE id = to_id;

    COMMIT;
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        RAISE;
END;
$$ LANGUAGE plpgsql;

-- 사용
CALL transfer(1, 2, 100000);
```

### 실습 2: SAVEPOINT 활용

```sql
BEGIN;

-- 기본 데이터 삽입
INSERT INTO orders (user_id, amount) VALUES (1, 50000);
SAVEPOINT order_created;

-- 재고 차감 시도
UPDATE products SET stock = stock - 1 WHERE id = 10;

-- 재고 확인
IF (SELECT stock FROM products WHERE id = 10) < 0 THEN
    ROLLBACK TO SAVEPOINT order_created;
    -- 주문은 유지하되 재고 차감만 취소
END IF;

COMMIT;
```

### 실습 3: 격리 수준 테스트

터미널 2개를 열어 테스트합니다.

```sql
-- 터미널 1
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SELECT * FROM accounts;

-- 터미널 2
UPDATE accounts SET balance = balance + 50000 WHERE id = 1;
COMMIT;

-- 터미널 1
SELECT * FROM accounts;  -- 변경 전 값 (스냅샷)
COMMIT;

SELECT * FROM accounts;  -- 이제 변경된 값 보임
```

### 실습 4: FOR UPDATE 잠금

```sql
-- 재고 확인 후 차감 (동시성 안전)
BEGIN;

-- 잠금을 걸며 조회
SELECT stock FROM products WHERE id = 1 FOR UPDATE;

-- 재고 확인 및 차감
UPDATE products
SET stock = stock - 1
WHERE id = 1 AND stock > 0;

COMMIT;
```

---

## 12. 트랜잭션 모니터링

```sql
-- 현재 실행 중인 트랜잭션 확인
SELECT
    pid,
    now() - xact_start AS duration,
    query,
    state
FROM pg_stat_activity
WHERE xact_start IS NOT NULL;

-- 잠금 대기 중인 쿼리 확인
SELECT
    blocked.pid AS blocked_pid,
    blocking.pid AS blocking_pid,
    blocked.query AS blocked_query
FROM pg_stat_activity blocked
JOIN pg_stat_activity blocking ON blocking.pid = ANY(pg_blocking_pids(blocked.pid));
```

---

**이전**: [함수와 프로시저](./10_Functions_and_Procedures.md) | **다음**: [트리거](./12_Triggers.md)

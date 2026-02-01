# 트랜잭션

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

## 다음 단계

[12_Triggers.md](./12_Triggers.md)에서 트리거를 배워봅시다!

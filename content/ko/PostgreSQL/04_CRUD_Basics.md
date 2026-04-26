# CRUD 기본

**이전**: [테이블과 데이터타입](./03_Tables_and_Data_Types.md) | **다음**: [조건과 정렬](./05_Conditions_and_Sorting.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. CRUD가 무엇의 약자인지 설명하고, 이 네 가지 연산이 데이터 조작의 기초를 이루는 이유를 설명할 수 있습니다
2. DEFAULT 값과 RETURNING을 활용하여 단일 및 다중 행을 추가하는 INSERT 문을 작성할 수 있습니다
3. 컬럼 별칭(alias), DISTINCT, 간단한 표현식을 사용한 SELECT 문을 작성할 수 있습니다
4. WHERE 절을 포함한 UPDATE 문을 작성하고 RETURNING으로 변경 내용을 확인할 수 있습니다
5. DELETE 문을 안전하게 작성하고 DELETE와 TRUNCATE의 차이를 구별할 수 있습니다
6. ON CONFLICT(DO NOTHING / DO UPDATE)를 사용하여 UPSERT 로직을 구현할 수 있습니다
7. 안전한 데이터 수정을 위한 모범 사례(SELECT 우선 확인, 트랜잭션)를 적용할 수 있습니다

---

애플리케이션과 데이터베이스 간의 거의 모든 상호작용은 네 가지 연산 중 하나로 귀결됩니다: 새 레코드 생성, 기존 레코드 읽기, 값 수정, 행 삭제입니다. SQL에서 CRUD를 익히는 것은 수학에서 사칙연산을 배우는 것과 같습니다 — 더 고급적인 모든 것들이 이 위에 쌓입니다.

네 개의 statement로 들어가기 전, [**이론과 원리**](#이론과-원리) 절을 먼저 읽으세요 — INSERT/UPDATE/DELETE가 디스크의 page에 실제로 무엇을 하는지(힌트 — UPDATE는 in-place로 덮어쓰지 않습니다), DELETE가 dead tuple을 남기는 이유, 그리고 HOT 최적화가 단순 update를 어떻게 저렴하게 유지하는지를 다룹니다.

---

## 이론과 원리

CRUD의 C, R, U, D는 모두 동일한 8 KB heap page에 닿지만, 닿는 방식이 놀랄 만큼 다릅니다. MVCC 아래에서 UPDATE는 기존 행을 *쓰는* 것이 아닙니다 — 새 버전의 INSERT + 옛 버전에 플래그를 다는 것입니다. DELETE는 공간을 해제하지 않습니다 — 행의 `xmax`만 표시하고, VACUUM이 나중에 회수하도록 묘비를 남깁니다. 이 차이를 이해하는 것이, 잘 동작하는 CRUD와 조용히 데이터베이스를 부풀리는 CRUD를 가르는 분기점입니다.

이 절에서 다루는 내용:

- **(A)** INSERT가 page에 하는 일 — tuple body, line pointer, FSM(Free Space Map).
- **(B)** UPDATE가 하는 일 — 그리고 보통 두 개의 row version이 생기는 이유.
- **(C)** HOT(Heap-Only Tuple) 최적화 — UPDATE가 같은 page에 새 버전을 두고 인덱스 갱신을 *통째로 건너뛸 수 있는* 조건.
- **(D)** DELETE가 하는 일 — 그리고 공간 회수에서 VACUUM의 역할.

### A. INSERT — 본질적으로 append-only

`INSERT INTO t (...) VALUES (...)`을 실행하면 PostgreSQL은 다음 순서를 거칩니다.

1. **충분한 free space가 있는 page를 찾기** — **Free Space Map**(FSM)을 참조. FSM은 각 page의 free space 양을 요약한 작은 보조 파일(`<oid>_fsm`)입니다.
2. **그 page에 content lock을 획득**.
3. **메모리에서 tuple을 구성** — `xmin = current_xid`, `xmax = 0`인 23바이트 header, 그리고 03번 레슨 §B.3에 따라 배치된 컬럼 데이터.
4. **tuple을 `pd_upper - tuple_size`에 두고**, line pointer를 `pd_lower`에 둠. 두 포인터 갱신.
5. **변경을 기술하는 WAL 레코드 작성**(`XLOG_HEAP_INSERT`).
6. **`shared_buffers`에서 page를 dirty로 표시**. 실제 디스크 쓰기는 background writer 또는 checkpointer를 통해 나중에 일어남.
7. **테이블의 모든 인덱스를 갱신**해서 새 행의 `(page_no, line_pointer_index)`를 가리키도록 함.

공간이 있는 page가 없으면 PostgreSQL이 파일을 8 KB(또는 그 이상 — PG 16에서 `extend_table_with_multiple_blocks` 도입)만큼 확장합니다.

#### A.1 INSERT가 빠른 이유

3-6단계는 모두 in-memory이거나 sequential WAL 쓰기입니다. 7단계(인덱스 갱신)가 지배적인 비용입니다 — 인덱스가 5개인 테이블은 page 갱신을 1번이 아니라 6번 치릅니다. "인덱스를 적게 두라"가 단순한 저장 규칙이 아니라 write-throughput 규칙인 이유입니다.

### B. UPDATE — Insert + 옛 버전을 superseded로 표시

PostgreSQL은 살아 있는 행을 절대 덮어쓰지 않습니다. `UPDATE t SET x = 5 WHERE id = 1;`은 다음을 실행합니다.

1. **기존 행을 찾기** (인덱스 또는 sequential scan).
2. **행을 lock** (`t_infomask` 비트 설정).
3. **새 컬럼 값으로 새 tuple을 구성** — `xmin = current_xid`, `xmax = 0`.
4. **새 tuple을 이 page에 자리가 있으면 같은 page에**, 없으면 다른 page에 배치 (FSM 참조).
5. **옛 tuple의 `xmax`를 `current_xid`로 설정**하고, `ctid`를 새 tuple로 향하게 함.
6. **WAL 작성** (`XLOG_HEAP_UPDATE`).
7. **변경된 indexed 컬럼마다 새 인덱스 항목 삽입** — 그리고 새 tuple이 다른 page에 산다면 변경되지 않은 indexed 컬럼들에 *대해서도* 삽입 (인덱스가 primary key가 아니라 `ctid`를 가리키기 때문).

#### B.1 두 버전의 비용

UPDATE 후, 두 row version이 *모두* 디스크에 존재합니다. 둘 다 line pointer를 가집니다. 옛 버전은 새 트랜잭션에 invisible(스냅샷이 `xmax = current_xid`를 committed-and-deleted로 보기 때문)이지만, 아직 제거할 수 없습니다 — 오래 실행 중인 트랜잭션이 여전히 필요로 할 수 있기 때문입니다. **VACUUM**이 나중에 line pointer status를 `LP_UNUSED`로 만들어 공간을 회수합니다.

이것이 **table bloat**입니다 — UPDATE가 잦은 테이블은 살아 있는 행 수가 시사하는 것보다 빠르게 디스크 크기가 커집니다. update가 많은 워크로드는 이를 통제하기 위해 autovacuum 튜닝이 필요합니다.

### C. HOT — Heap-Only Tuple

UPDATE가 다음 두 조건을 만족하면 PostgreSQL은 **HOT**이라는 fast path를 탑니다.

1. **indexed 컬럼이 하나도 변경되지 않음.**
2. **새 tuple이 옛 tuple과 같은 page에 들어맞음.**

이 경우 §B의 7단계(모든 인덱스 갱신)가 **통째로 생략됩니다**. 새 tuple은 같은 page에 배치되고, 옛 line pointer는 새 line pointer를 가리키는 "redirect"로 변환됩니다. 인덱스는 여전히 옛 line pointer를 가리키지만, 읽기는 redirect 체인을 투명하게 따라갑니다.

```
HOT update 이전:
  index → LP[3] → tuple v1 (id=1, x=4)

HOT update of x=4 → x=5 이후:
  index → LP[3] → (LP[4]로 redirect)
                  LP[4] → tuple v2 (id=1, x=5)
```

이점이 누적됩니다.

- **인덱스 갱신 0회.** 인덱스 8개인 테이블도 page 갱신은 1번뿐.
- **dead tuple은 일반 page 읽기 중 `HOT pruning`에 의해 회수**됨 — VACUUM 실행이 필요 없음. line pointer는 page-level cleanup에서 즉시 해제됨.
- **WAL 양 감소** — 인덱스 갱신이 로깅되지 않으므로.

#### C.1 fillfactor 조절

HOT은 새 tuple이 같은 page에 들어맞아야만 동작합니다. PostgreSQL은 테이블별 `fillfactor` 설정(`CREATE TABLE ... WITH (fillfactor = 80);`)을 노출합니다 — "각 page의 20%를 미래 update를 위해 비워 두라"는 뜻. update가 많은 테이블에서는 fillfactor를 100 미만으로 두면 HOT 적중률이 극적으로 올라갑니다.

### D. DELETE — 그저 묘비

`DELETE FROM t WHERE id = 1;`은 디스크 공간을 해제하지 *않습니다*. 다음을 실행합니다.

1. **행을 찾음.**
2. **tuple header의 `xmax = current_xid`로 설정.**
3. **WAL 작성** (`XLOG_HEAP_DELETE`).

그게 전부입니다. tuple body와 line pointer는 자리에 그대로 남습니다. 새 트랜잭션은 그 행을 건너뛸 것이지만(스냅샷이 검사할 때 `xmax`가 committed로 보임), 바이트는 여전히 디스크에 있습니다.

#### D.1 VACUUM이 하는 일

VACUUM이 page를 훑을 때, 각 tuple의 visibility를 클러스터의 가장 오래된 active transaction(`OldestXmin`)과 대조합니다. `xmax`가 `OldestXmin`보다 오래되었다면, 그 tuple은 *모든* 현재 및 미래 트랜잭션에 invisible — 제거 안전. VACUUM은:

1. **page를 compact해서 tuple body 해제.**
2. **line pointer를 `LP_UNUSED`로 표시** — 이 page에 대한 미래 INSERT가 재사용할 수 있도록.
3. **page를 FSM에 기록** — 미래 INSERT가 찾을 수 있도록.
4. **회수된 line pointer를 가리키던 인덱스 항목 제거** — 비싼 부분, VACUUM이 각 인덱스를 스캔해야 함.

`VACUUM FULL`은 더 공격적입니다 — 테이블 전체를 dead tuple 없는 새 파일로 다시 쓰고 atomic하게 교체. `ACCESS EXCLUSIVE` lock을 잡고 느릴 수 있지만, 디스크를 실제로 OS에 반환합니다.

#### D.2 TRUNCATE — 우회

`TRUNCATE TABLE t;`는 개념적으로 `DELETE FROM t;`이지만 dead tuple을 만들지 않습니다. 새 빈 파일을 할당하고 atomic하게 교체하는 방식으로 구현. 테이블 크기와 무관하게 O(1)이지만, `WHERE` filter가 불가능하고 강한 lock을 잡습니다.

### 이론에서 아래 SQL로

이어지는 각 절은 위 메커니즘이 구체화된 형태입니다:

- **`INSERT`** — §A의 순서를 실행. multi-row `INSERT ... VALUES`는 WAL flush를 행 사이에 amortize.
- **`UPDATE`** — §B를 실행. UPDATE가 indexed 컬럼을 건드리지 않으면 §C HOT이 발동.
- **`DELETE`** — §D를 실행. 공간은 `VACUUM` 또는 `VACUUM FULL`만이 회수.
- **`SELECT`** — §A-D의 visibility 규칙(`xmin`/`xmax` vs snapshot)을 사용해 어떤 row version이 트랜잭션에 보이는지 결정.
- **`RETURNING` 절** — `INSERT`, `UPDATE`, `DELETE`가 영향받은 행을 한 번의 round-trip에 반환하게 함. 특히 serial key의 `INSERT ... RETURNING id`에 유용.

---

## 0. 실습 준비

```sql
-- 실습용 테이블 생성
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    age INTEGER,
    city VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 1. INSERT - 데이터 삽입

### 단일 행 삽입

```sql
-- 모든 컬럼 지정
INSERT INTO users (name, email, age, city)
VALUES ('김철수', 'kim@email.com', 30, '서울');

-- 일부 컬럼만 지정 (나머지는 DEFAULT 또는 NULL)
INSERT INTO users (name, email)
VALUES ('이영희', 'lee@email.com');
```

### 다중 행 삽입

```sql
INSERT INTO users (name, email, age, city) VALUES
('박민수', 'park@email.com', 25, '부산'),
('최지영', 'choi@email.com', 28, '대전'),
('정수진', 'jung@email.com', 35, '서울');
```

### DEFAULT 값 사용

```sql
-- 특정 컬럼에 DEFAULT 사용
INSERT INTO users (name, email, age, city, created_at)
VALUES ('홍길동', 'hong@email.com', 40, '인천', DEFAULT);

-- 모든 컬럼 DEFAULT (id만 자동 생성)
INSERT INTO users DEFAULT VALUES;  -- 에러: NOT NULL 컬럼 때문
```

### RETURNING - 삽입된 데이터 반환

```sql
-- RETURNING은 INSERT 후 별도 SELECT를 불필요하게 함 — 생성된 값(id, 타임스탬프)을
-- 같은 왕복(round-trip)으로 반환하여 지연 시간을 50% 절감
INSERT INTO users (name, email, age, city)
VALUES ('신짱구', 'shin@email.com', 5, '떡잎마을')
RETURNING id;

-- 여러 컬럼 반환
INSERT INTO users (name, email, age, city)
VALUES ('김미영', 'mikim@email.com', 32, '서울')
RETURNING id, name, created_at;

-- 모든 컬럼 반환
INSERT INTO users (name, email)
VALUES ('테스트', 'test@email.com')
RETURNING *;
```

---

## 2. SELECT - 데이터 조회

### 모든 데이터 조회

```sql
-- 모든 컬럼
SELECT * FROM users;

-- 특정 컬럼만
SELECT name, email FROM users;
```

### 컬럼 별칭 (Alias)

```sql
SELECT
    name AS 이름,
    email AS 이메일,
    age AS 나이
FROM users;

-- AS 생략 가능
SELECT name 이름, email 이메일 FROM users;
```

### 중복 제거 (DISTINCT)

```sql
-- 중복 도시 제거
SELECT DISTINCT city FROM users;

-- 여러 컬럼 조합의 중복 제거
SELECT DISTINCT city, age FROM users;
```

### 계산 및 표현식

```sql
-- 계산
SELECT name, age, age + 10 AS age_after_10_years FROM users;

-- 문자열 연결
SELECT name || ' (' || email || ')' AS user_info FROM users;

-- CONCAT 함수
SELECT CONCAT(name, ' - ', city) AS name_city FROM users;
```

### 조건 조회 (간단히)

```sql
-- WHERE 절 (자세한 내용은 다음 장)
SELECT * FROM users WHERE city = '서울';
SELECT * FROM users WHERE age >= 30;
```

---

## 3. UPDATE - 데이터 수정

### 기본 UPDATE

```sql
-- 특정 행 수정
UPDATE users
SET age = 31
WHERE name = '김철수';

-- 여러 컬럼 수정
UPDATE users
SET age = 26, city = '대구'
WHERE email = 'park@email.com';
```

### 조건 없는 UPDATE (주의!)

```sql
-- 모든 행이 수정됨!
UPDATE users SET city = '서울';  -- 위험!

-- 항상 WHERE 절 확인
```

### 계산을 이용한 UPDATE

```sql
-- 모든 사용자 나이 1 증가
UPDATE users SET age = age + 1;

-- 특정 조건 사용자만
UPDATE users SET age = age + 1 WHERE city = '서울';
```

### RETURNING으로 수정된 데이터 확인

```sql
UPDATE users
SET age = 32
WHERE name = '이영희'
RETURNING *;

UPDATE users
SET city = '광주'
WHERE age < 30
RETURNING id, name, city;
```

### NULL로 설정

```sql
UPDATE users
SET city = NULL
WHERE name = '테스트';
```

---

## 4. DELETE - 데이터 삭제

### 기본 DELETE

```sql
-- 특정 행 삭제
DELETE FROM users WHERE name = '테스트';

-- 여러 조건
DELETE FROM users WHERE city IS NULL AND age IS NULL;
```

### 조건 없는 DELETE (주의!)

```sql
-- 모든 데이터 삭제!
DELETE FROM users;  -- 위험!

-- 테이블은 남아있음
```

### RETURNING으로 삭제된 데이터 확인

```sql
DELETE FROM users
WHERE email = 'test@email.com'
RETURNING *;
```

### TRUNCATE - 테이블 비우기

```sql
-- TRUNCATE는 행 단위 WAL 로깅 우회 — 페이지를 직접 해제하여
-- 대용량 테이블 비우기에 DELETE보다 수십~수백 배 빠름.
-- 트레이드오프: 행별 트리거 미실행, RETURNING 사용 불가
TRUNCATE TABLE users;

-- SERIAL 재시작
TRUNCATE TABLE users RESTART IDENTITY;

-- 관련 테이블도 함께 (외래키)
TRUNCATE TABLE users CASCADE;
```

### DELETE vs TRUNCATE

| 특징 | DELETE | TRUNCATE |
|------|--------|----------|
| WHERE 조건 | 가능 | 불가능 |
| 속도 | 느림 | 빠름 |
| 트랜잭션 롤백 | 가능 | 제한적 |
| RETURNING | 가능 | 불가능 |
| 트리거 실행 | 실행됨 | 실행 안됨 |
| SERIAL 리셋 | 안됨 | 선택 가능 |

---

## 5. UPSERT (ON CONFLICT)

삽입 시 충돌이 발생하면 업데이트하는 기능입니다.

### 충돌 시 무시

```sql
-- ON CONFLICT DO NOTHING은 멱등성(idempotent) 삽입에 이상적 — 동일한 요청을
-- 재시도해도(예: 메시지 큐에서) 중복 행이나 에러가 발생하지 않음
INSERT INTO users (name, email, age, city)
VALUES ('김철수', 'kim@email.com', 35, '부산')
ON CONFLICT (email) DO NOTHING;
```

### 충돌 시 업데이트

```sql
-- 이미 존재하면 업데이트
INSERT INTO users (name, email, age, city)
VALUES ('김철수', 'kim@email.com', 35, '부산')
ON CONFLICT (email)
DO UPDATE SET
    age = EXCLUDED.age,
    city = EXCLUDED.city;
```

### EXCLUDED 키워드

`EXCLUDED`는 삽입하려고 했던 데이터를 참조합니다.

```sql
INSERT INTO users (name, email, age, city)
VALUES ('김철수', 'kim@email.com', 35, '부산')
ON CONFLICT (email)
DO UPDATE SET
    age = EXCLUDED.age,           -- 새 값 (35)
    city = users.city,            -- 기존 값 유지
    name = EXCLUDED.name;         -- 새 값 (김철수)
```

### 조건부 UPSERT

```sql
INSERT INTO users (name, email, age, city)
VALUES ('김철수', 'kim@email.com', 35, '부산')
ON CONFLICT (email)
DO UPDATE SET
    age = EXCLUDED.age,
    city = EXCLUDED.city
WHERE users.age < EXCLUDED.age;  -- 기존 나이보다 클 때만 업데이트
```

---

## 6. 서브쿼리를 이용한 INSERT

### SELECT 결과 삽입

```sql
-- 다른 테이블에서 복사
CREATE TABLE users_backup AS SELECT * FROM users;

-- 또는
INSERT INTO users_backup SELECT * FROM users;

-- 조건부 복사
INSERT INTO users_backup
SELECT * FROM users WHERE city = '서울';
```

### 계산된 값 삽입

```sql
INSERT INTO statistics (city, user_count)
SELECT city, COUNT(*) FROM users GROUP BY city;
```

---

## 7. 실습 예제

### 실습 데이터 준비

```sql
-- 테이블 초기화
TRUNCATE TABLE users RESTART IDENTITY;

-- 샘플 데이터 삽입
INSERT INTO users (name, email, age, city) VALUES
('김철수', 'kim@email.com', 30, '서울'),
('이영희', 'lee@email.com', 25, '부산'),
('박민수', 'park@email.com', 35, '서울'),
('최지영', 'choi@email.com', 28, '대전'),
('정수진', 'jung@email.com', 32, '서울'),
('홍길동', 'hong@email.com', 40, '인천'),
('강동원', 'kang@email.com', 27, '부산'),
('손예진', 'son@email.com', 33, '서울');
```

### 실습 1: 기본 CRUD

```sql
-- 1. 새 사용자 추가
INSERT INTO users (name, email, age, city)
VALUES ('신규회원', 'new@email.com', 22, '광주')
RETURNING *;

-- 2. 서울 사용자 조회
SELECT * FROM users WHERE city = '서울';

-- 3. 나이 30 이상 사용자의 도시를 '수도권'으로 변경
UPDATE users
SET city = '수도권'
WHERE age >= 30
RETURNING name, age, city;

-- 4. 광주 사용자 삭제
DELETE FROM users
WHERE city = '광주'
RETURNING *;
```

### 실습 2: UPSERT

```sql
-- 이메일이 이미 존재하면 나이와 도시 업데이트
INSERT INTO users (name, email, age, city)
VALUES ('김철수', 'kim@email.com', 31, '경기')
ON CONFLICT (email)
DO UPDATE SET
    age = EXCLUDED.age,
    city = EXCLUDED.city
RETURNING *;

-- 존재하지 않는 이메일이면 새로 삽입
INSERT INTO users (name, email, age, city)
VALUES ('새회원', 'newuser@email.com', 29, '제주')
ON CONFLICT (email)
DO UPDATE SET age = EXCLUDED.age, city = EXCLUDED.city
RETURNING *;
```

### 실습 3: 대량 데이터 처리

```sql
-- 백업 테이블 생성 및 데이터 복사
CREATE TABLE users_backup AS
SELECT * FROM users WHERE 1=0;  -- 구조만 복사

INSERT INTO users_backup
SELECT * FROM users;

-- 특정 조건 사용자만 백업
INSERT INTO users_backup
SELECT * FROM users WHERE city IN ('서울', '부산');

-- 백업 확인
SELECT COUNT(*) FROM users_backup;
```

---

## 8. 주의사항 및 팁

### SQL Injection 방지

```sql
-- 나쁜 예 (문자열 직접 연결)
-- "SELECT * FROM users WHERE name = '" + userInput + "'"

-- 좋은 예 (파라미터 바인딩 사용 - 애플리케이션에서)
-- "SELECT * FROM users WHERE name = $1"
```

### UPDATE/DELETE 전 확인

```sql
-- 1. 먼저 SELECT로 대상 확인
SELECT * FROM users WHERE city = '서울';

-- 2. 확인 후 UPDATE/DELETE 실행
UPDATE users SET age = age + 1 WHERE city = '서울';
```

### 트랜잭션 활용

```sql
-- 중요한 작업은 트랜잭션으로
BEGIN;
UPDATE users SET age = age + 1 WHERE city = '서울';
-- 결과 확인 후
COMMIT;  -- 또는 ROLLBACK;
```

---

---

**이전**: [테이블과 데이터타입](./03_Tables_and_Data_Types.md) | **다음**: [조건과 정렬](./05_Conditions_and_Sorting.md)

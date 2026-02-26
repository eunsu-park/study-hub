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

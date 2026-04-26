# 함수와 프로시저

**이전**: [뷰와 인덱스](./09_Views_and_Indexes.md) | **다음**: [트랜잭션](./11_Transactions.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 문자열·숫자·날짜에 대한 PostgreSQL 내장 함수(Built-in Function)를 활용한다
2. `CREATE FUNCTION ... LANGUAGE SQL`을 사용해 간단한 SQL 함수를 생성한다
3. 변수·IF-ELSE 분기·CASE·반복문을 포함한 PL/pgSQL 함수를 작성한다
4. `RETURNS TABLE`, `RETURNS SETOF`, OUT 파라미터를 사용해 함수에서 테이블 데이터를 반환한다
5. `BEGIN ... EXCEPTION` 블록과 RAISE를 사용해 예외 처리를 구현한다
6. 저장 프로시저(PROCEDURE)를 생성하고 함수와의 차이점을 설명한다
7. `CREATE OR REPLACE`를 적용해 기존 함수를 삭제하지 않고 수정한다
8. 사용자 정의 함수와 프로시저를 목록 조회·내용 확인·삭제한다

---

내장 함수는 가장 일반적인 변환을 처리해 주지만, 모든 애플리케이션은 결국 데이터베이스 내부에 위치해야 하는 커스텀 로직을 필요로 합니다. 사용자 정의 함수와 프로시저를 사용하면 세금 계산, 등급 분류, 입력 유효성 검사 같은 비즈니스 규칙을 데이터 바로 옆에 캡슐화할 수 있습니다. 이를 통해 왕복 통신 횟수를 줄이고, 어떤 클라이언트가 접속하더라도 동일한 로직이 일관되게 적용되도록 보장합니다.

함수를 작성하기 전, [**이론과 원리**](#이론과-원리) 절을 먼저 읽으세요 — PL/pgSQL이 어떻게 parsing되고 캐시되는지, volatility 클래스(`IMMUTABLE`/`STABLE`/`VOLATILE`)가 planner에게 무엇을 말하는지, 그리고 procedure가 *왜* 단순히 "값을 반환하지 않는 함수"가 아닌지를 다룹니다.

---

## 이론과 원리

`CREATE FUNCTION` 선언은 단순한 inline SQL의 sugar 그 이상입니다. 함수 body는 parsing되고 캐시되며, 선언한 volatility 클래스는 planner가 호출을 fold할 수 있는 방식을 바꾸고, 언어(SQL vs PL/pgSQL vs C)는 body가 planning 시점에 inline될 수 있는지를 결정합니다. `CREATE PROCEDURE`는 비슷해 보이지만 트랜잭션 의미론이 근본적으로 다릅니다 — body 중간에 `COMMIT`과 `BEGIN`을 할 수 있는 반면, 함수는 그럴 수 없습니다. 이 네 가지 — 언어, volatility, inlining, 트랜잭션 scope — 를 이해하면 "함수를 추가했더니 느려졌다"가 예측 가능한 cost 분석으로 바뀝니다.

이 절에서 다루는 내용:

- **(A)** 함수 언어 — SQL 함수 vs PL/pgSQL, 각각이 inline되는 시점과 자체 scope를 갖는 시점.
- **(B)** Volatility 클래스 — `IMMUTABLE`, `STABLE`, `VOLATILE`과 각각이 enable하는 planner 최적화.
- **(C)** PL/pgSQL — parse, plan caching, 그리고 statement당 plan 재사용.
- **(D)** Function vs procedure — 트랜잭션 통제의 차이.

### A. 함수 언어와 inlining

PostgreSQL 함수는 여러 언어로 작성 가능. 사용자 코드에서 가장 흔한 둘:

#### A.1 SQL 함수

`CREATE FUNCTION add_one(x int) RETURNS int LANGUAGE sql AS 'SELECT x + 1';`

body는 (다중 statement일 수 있는) SQL block. 함수가 **충분히 단순**하면 — 단일 SELECT, IMMUTABLE/STABLE, 복잡한 매개변수 없음 — planner가 **inline**할 수 있음 — view처럼 planning 중에 모든 호출이 body로 치환됨. 그러면 runtime에 함수 호출 overhead가 없음.

SQL 함수가 항상 inlinable한 것은 아님. 다중 statement body, VOLATILE 표시, 비일상적 컨텍스트의 set-returning 함수 — 이들 중 어느 것이라도 inlining을 막고 진짜 per-call 실행을 강제.

#### A.2 PL/pgSQL 함수

`CREATE FUNCTION ... LANGUAGE plpgsql AS $$ DECLARE ... BEGIN ... END $$;`

PL/pgSQL은 SQL 위에 layered된 절차적 언어. body에 변수, 제어 흐름(`IF`, `LOOP`, `FOR`, `WHILE`), 예외 처리, 명시적 `RETURN`이 있음. PL/pgSQL 함수는 **절대 inline되지 않음** — 모든 호출이 자체 변수 scope를 가진 진짜 호출이며, body는 1번 parsing되지만 매번 실행됨.

비용 — 호출당 함수 호출 overhead(마이크로초 단위지만, 행마다 호출되면 누적). 이점 — 순수 SQL로 표현할 수 없는 모든 것.

### B. Volatility 클래스

모든 함수는 세 **volatility** 클래스 중 하나를 선언(또는 기본값). planner에게 어떤 최적화가 안전한지 알리는 메타데이터.

#### B.1 IMMUTABLE — 같은 입력, 같은 출력, 영원히

`abs(x)`, `length(s)`, `pi()`. planner가 자유롭게:

- 모든 인자가 상수이면 호출을 planning 시점에 **constant-fold**. `WHERE x = abs(-5)`가 한 번에 `WHERE x = 5`가 됨.
- **expression 인덱스에 함수 사용** (`CREATE INDEX ON t (immutable_func(col))`). PostgreSQL은 STABLE이나 VOLATILE 함수에 인덱스를 만들기를 거부 — 인덱스가 조용히 stale해지기 때문.

#### B.2 STABLE — 같은 입력, 같은 출력, 한 statement 내에서

`now()`, `current_user`, `current_setting('foo')`. 한 쿼리 안에서 `now()`를 10번 호출하면 같은 값을 반환(PostgreSQL이 보장). 쿼리 사이에서는 그렇지 않음.

planner가:

- **bound로서 index scan에 함수 사용 가능** (`WHERE x > now()`). 함수는 스캔 시작 시 1번 호출되고, 결과는 스캔 동안 상수로 처리.
- statement 사이에서는 constant-fold **불가**.

#### B.3 VOLATILE — 무엇이든 가능

`random()`, `nextval('seq')`, `INSERT`나 `UPDATE`를 수행하는 모든 함수. planner는 매번, source 순서대로 호출해야 하며, iteration 경계를 가로질러 옮길 수 없음.

#### B.4 잘못 선언하면 조용히 깨지는 이유

`now()`를 IMMUTABLE로 표시하면 planner가 "오늘"을 prepared statement에서 절대 변하지 않는 값으로 constant-fold할 수 있음 — 같은 쿼리, 연결의 남은 생애 동안 같은 답. `random()`을 STABLE로 표시하면 planner가 한 SELECT 전체에 걸쳐 random 값 하나를 재사용할 수 있음. PostgreSQL은 당신의 선언을 신뢰함 — 잘못된 선언은 에러가 아니라 잘못된 답을 만듦.

기본값은 VOLATILE — 안전하지만 비관적. 사용자 정의 함수를 정의할 때는 항상 올바른 클래스를 생각.

### C. PL/pgSQL — parse, plan, cache

PL/pgSQL 함수 body는 세 단계로 처리됨.

#### C.1 첫 호출 — full parse

세션이 함수를 처음 호출할 때, PL/pgSQL이 body 전체를 내부 트리로 parsing. 모든 embedded SQL statement(`SELECT`, `INSERT`, `EXECUTE`)는 노드가 됨.

#### C.2 각 SQL statement의 첫 실행 — plan과 캐시

각 embedded SQL statement는 처음 실행될 때 planning되고, plan은 **세션 생애 동안 캐시**됨. 함수의 다음 호출은 캐시된 plan을 재사용 — 재-planning 없음.

이로써 PL/pgSQL은 반복 호출에 빠르지만 미묘한 함정을 만듦 — 캐시된 plan은 *첫* 호출의 매개변수 값으로 빌드(PostgreSQL은 부분 generic-plan 로직 사용). 후속 호출의 매개변수 selectivity가 매우 다르면 캐시된 plan이 그것들에 나쁠 수 있음. PostgreSQL은 cost 추정에 따라 몇 번의 호출 후 custom과 generic plan 사이에서 전환하지만, 병리적 사례가 존재.

#### C.3 `EXECUTE` — 명시적 재-planning

PL/pgSQL 안에서, 평범한 SQL statement는 캐시된-plan 메커니즘을 사용. 동적 형태 `EXECUTE 'SELECT ...';`는 매번 다시 parsing하고 다시 plan함. 매개변수 값이 selectivity를 급격히 바꾸는 쿼리에, 또는 쿼리 텍스트 자체가 동적으로 빌드되는 경우에 사용.

### D. Function vs Procedure — 트랜잭션 scope

PostgreSQL 11까지는 함수만 있었음. PG 11이 procedure(`CREATE PROCEDURE`)를 추가했고, 차이는 순전히 트랜잭션 통제에 관한 것:

| 특성 | Function | Procedure |
|------|----------|-----------|
| 값 반환 | 예 (`RETURNS type`) | 아니오 (`OUT` 매개변수 사용) |
| 호출 방법 | `SELECT func()` (expression 안에서) | `CALL proc()` (top-level) |
| body 안의 `COMMIT` / `ROLLBACK` | **불가** | **가능** |
| 병렬 쿼리에서 실행 가능 | `PARALLEL SAFE`로 표시되면 | 아니오 |

#### D.1 함수가 COMMIT 불가능한 이유

함수 호출은 SQL statement의 일부. statement는 트랜잭션의 일부. 함수가 호출 중에 commit하게 허용하면 호출 statement가 commit 경계를 가로지르게 됨 — MVCC visibility에 incoherent하고, 트리거에 undefined하고, 에러 시 executor가 정리 불가능.

Procedure는 expression 내부가 아니라 top level에서 호출됨. body 안에서 `COMMIT; ... BEGIN;`을 하고 새 트랜잭션을 시작할 수 있음. 여러 트랜잭션에 걸쳐 작업을 chunk해야 하는 batch job에 사용(예 — "1000행 처리, commit, 반복").

#### D.2 익명 코드 — `DO` block

`DO $$ DECLARE x int; BEGIN ... END $$;`는 익명 PL/pgSQL block을 실행. 일회성 스크립트에 유용. block은 호출하는 트랜잭션에서 실행되며, 함수처럼 commit 불가.

### 이론에서 아래 SQL로

이어지는 각 절은 위 메커니즘이 구체화된 형태입니다:

- **내장 함수** (`UPPER`, `LENGTH`, `NOW`, `RANDOM`) — 대부분 IMMUTABLE 또는 STABLE, planner가 이를 활용 (§B).
- **`CREATE FUNCTION ... LANGUAGE sql`** — 단순할 때 inlinable (§A.1).
- **`CREATE FUNCTION ... LANGUAGE plpgsql`** — 변수, 제어 흐름, 절대 inline 안 됨 (§A.2).
- **`IMMUTABLE` / `STABLE` / `VOLATILE`** — volatility 계약 선언 (§B).
- **`CREATE PROCEDURE` + `CALL`** — top-level 호출, body 중간에 `COMMIT` 가능 (§D).
- **`DO $$ ... $$`** — 익명 block, 호출 트랜잭션에서 실행 (§D.2).
- **PL/pgSQL 안의 `EXECUTE 'sql'`** — 동적 SQL을 위해 plan cache 우회 (§C.3).

---

## 1. 내장 함수

PostgreSQL은 다양한 내장 함수를 제공합니다.

### 문자열 함수

| 함수 | 설명 | 예시 | 결과 |
|------|------|------|------|
| `LENGTH()` | 문자열 길이 | `LENGTH('Hello')` | 5 |
| `UPPER()` | 대문자 변환 | `UPPER('hello')` | HELLO |
| `LOWER()` | 소문자 변환 | `LOWER('HELLO')` | hello |
| `TRIM()` | 공백 제거 | `TRIM('  hi  ')` | hi |
| `SUBSTRING()` | 부분 문자열 | `SUBSTRING('Hello', 1, 3)` | Hel |
| `REPLACE()` | 문자열 치환 | `REPLACE('Hello', 'l', 'L')` | HeLLo |
| `CONCAT()` | 문자열 연결 | `CONCAT('A', 'B', 'C')` | ABC |
| `SPLIT_PART()` | 구분자로 분리 | `SPLIT_PART('a,b,c', ',', 2)` | b |

```sql
SELECT
    LENGTH('PostgreSQL') AS len,
    UPPER('hello') AS upper,
    LOWER('WORLD') AS lower,
    TRIM('  text  ') AS trimmed,
    SUBSTRING('PostgreSQL', 1, 8) AS sub,
    REPLACE('Hello', 'l', 'L') AS replaced,
    CONCAT('Post', 'gre', 'SQL') AS concat;
```

### 숫자 함수

| 함수 | 설명 | 예시 | 결과 |
|------|------|------|------|
| `ROUND()` | 반올림 | `ROUND(3.567, 2)` | 3.57 |
| `FLOOR()` | 내림 | `FLOOR(3.9)` | 3 |
| `CEIL()` | 올림 | `CEIL(3.1)` | 4 |
| `ABS()` | 절대값 | `ABS(-5)` | 5 |
| `MOD()` | 나머지 | `MOD(10, 3)` | 1 |
| `POWER()` | 거듭제곱 | `POWER(2, 3)` | 8 |
| `SQRT()` | 제곱근 | `SQRT(16)` | 4 |
| `RANDOM()` | 0~1 난수 | `RANDOM()` | 0.xxx |

```sql
SELECT
    ROUND(123.456, 2),
    FLOOR(9.9),
    CEIL(1.1),
    ABS(-100),
    MOD(17, 5),
    POWER(2, 10),
    ROUND(RANDOM() * 100);
```

### 날짜/시간 함수

| 함수 | 설명 |
|------|------|
| `NOW()` | 현재 타임스탬프 |
| `CURRENT_DATE` | 현재 날짜 |
| `CURRENT_TIME` | 현재 시간 |
| `DATE_TRUNC()` | 날짜 자르기 |
| `EXTRACT()` | 날짜 요소 추출 |
| `AGE()` | 날짜 차이 |
| `TO_CHAR()` | 날짜 포맷팅 |

```sql
SELECT
    NOW(),
    CURRENT_DATE,
    DATE_TRUNC('month', NOW()),
    EXTRACT(YEAR FROM NOW()),
    EXTRACT(DOW FROM NOW()),  -- 0=일요일
    AGE('2024-12-31', '2024-01-01'),
    TO_CHAR(NOW(), 'YYYY-MM-DD HH24:MI:SS');
```

---

## 2. 사용자 정의 함수 기본

### SQL 함수

```sql
-- 간단한 함수
CREATE FUNCTION add_numbers(a INTEGER, b INTEGER)
RETURNS INTEGER
AS $$
    SELECT a + b;
$$ LANGUAGE SQL;

-- 사용
SELECT add_numbers(5, 3);  -- 8
```

### 함수 삭제

```sql
DROP FUNCTION add_numbers(INTEGER, INTEGER);
DROP FUNCTION IF EXISTS add_numbers(INTEGER, INTEGER);
```

---

## 3. PL/pgSQL 함수

PL/pgSQL은 PostgreSQL의 절차적 언어입니다.

### 기본 구조

```sql
CREATE FUNCTION function_name(parameters)
RETURNS return_type
AS $$
DECLARE
    -- 변수 선언
BEGIN
    -- 함수 본문
    RETURN value;
END;
$$ LANGUAGE plpgsql;
```

### 변수와 할당

```sql
CREATE FUNCTION calculate_tax(price NUMERIC)
RETURNS NUMERIC
AS $$
DECLARE
    tax_rate NUMERIC := 0.1;  -- 10%
    tax_amount NUMERIC;
BEGIN
    tax_amount := price * tax_rate;
    RETURN tax_amount;
END;
$$ LANGUAGE plpgsql;

SELECT calculate_tax(10000);  -- 1000
```

### IF-ELSE

```sql
CREATE FUNCTION get_grade(score INTEGER)
RETURNS VARCHAR
AS $$
BEGIN
    IF score >= 90 THEN
        RETURN 'A';
    ELSIF score >= 80 THEN
        RETURN 'B';
    ELSIF score >= 70 THEN
        RETURN 'C';
    ELSIF score >= 60 THEN
        RETURN 'D';
    ELSE
        RETURN 'F';
    END IF;
END;
$$ LANGUAGE plpgsql;

SELECT get_grade(85);  -- B
```

### CASE 문

```sql
CREATE FUNCTION day_name(day_num INTEGER)
RETURNS VARCHAR
AS $$
BEGIN
    RETURN CASE day_num
        WHEN 0 THEN '일요일'
        WHEN 1 THEN '월요일'
        WHEN 2 THEN '화요일'
        WHEN 3 THEN '수요일'
        WHEN 4 THEN '목요일'
        WHEN 5 THEN '금요일'
        WHEN 6 THEN '토요일'
        ELSE '잘못된 입력'
    END;
END;
$$ LANGUAGE plpgsql;
```

### 반복문

```sql
-- LOOP
CREATE FUNCTION factorial(n INTEGER)
RETURNS BIGINT
AS $$
DECLARE
    result BIGINT := 1;
    i INTEGER := 1;
BEGIN
    LOOP
        EXIT WHEN i > n;
        result := result * i;
        i := i + 1;
    END LOOP;
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- FOR 루프
CREATE FUNCTION sum_to_n(n INTEGER)
RETURNS INTEGER
AS $$
DECLARE
    total INTEGER := 0;
BEGIN
    FOR i IN 1..n LOOP
        total := total + i;
    END LOOP;
    RETURN total;
END;
$$ LANGUAGE plpgsql;

-- WHILE
CREATE FUNCTION count_digits(num INTEGER)
RETURNS INTEGER
AS $$
DECLARE
    n INTEGER := ABS(num);
    count INTEGER := 0;
BEGIN
    WHILE n > 0 LOOP
        n := n / 10;
        count := count + 1;
    END LOOP;
    RETURN CASE WHEN count = 0 THEN 1 ELSE count END;
END;
$$ LANGUAGE plpgsql;
```

---

## 4. 테이블 데이터 반환

### RETURNS TABLE

```sql
CREATE FUNCTION get_users_by_city(p_city VARCHAR)
RETURNS TABLE (
    user_id INTEGER,
    user_name VARCHAR,
    user_email VARCHAR
)
AS $$
BEGIN
    RETURN QUERY
    SELECT id, name, email
    FROM users
    WHERE city = p_city;
END;
$$ LANGUAGE plpgsql;

-- 사용
SELECT * FROM get_users_by_city('서울');
```

### RETURNS SETOF

```sql
CREATE FUNCTION get_expensive_products(min_price NUMERIC)
RETURNS SETOF products
AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM products WHERE price >= min_price;
END;
$$ LANGUAGE plpgsql;

-- 사용
SELECT * FROM get_expensive_products(100000);
```

### OUT 파라미터

```sql
CREATE FUNCTION get_user_stats(
    IN p_user_id INTEGER,
    OUT order_count INTEGER,
    OUT total_amount NUMERIC
)
AS $$
BEGIN
    SELECT COUNT(*), COALESCE(SUM(amount), 0)
    INTO order_count, total_amount
    FROM orders
    WHERE user_id = p_user_id;
END;
$$ LANGUAGE plpgsql;

-- 사용
SELECT * FROM get_user_stats(1);
```

---

## 5. 예외 처리

```sql
CREATE FUNCTION safe_divide(a NUMERIC, b NUMERIC)
RETURNS NUMERIC
AS $$
BEGIN
    IF b = 0 THEN
        RAISE EXCEPTION '0으로 나눌 수 없습니다.';
    END IF;
    RETURN a / b;
EXCEPTION
    WHEN division_by_zero THEN
        RAISE NOTICE '0으로 나누기 시도됨';
        RETURN NULL;
    WHEN OTHERS THEN
        RAISE NOTICE '예외 발생: %', SQLERRM;
        RETURN NULL;
END;
$$ LANGUAGE plpgsql;
```

### RAISE 레벨

```sql
RAISE DEBUG 'Debug message';
RAISE LOG 'Log message';
RAISE INFO 'Info message';
RAISE NOTICE 'Notice message';     -- 기본 출력
RAISE WARNING 'Warning message';
RAISE EXCEPTION 'Error message';   -- 실행 중단
```

---

## 6. 프로시저 (PROCEDURE)

함수와 달리 값을 반환하지 않고 작업을 수행합니다 (PostgreSQL 11+).

### 프로시저 생성

```sql
CREATE PROCEDURE update_user_status(p_user_id INTEGER, p_status VARCHAR)
AS $$
BEGIN
    UPDATE users SET status = p_status WHERE id = p_user_id;
    RAISE NOTICE '사용자 % 상태가 %로 변경됨', p_user_id, p_status;
END;
$$ LANGUAGE plpgsql;

-- 호출
CALL update_user_status(1, 'active');
```

### 트랜잭션 제어

```sql
CREATE PROCEDURE transfer_money(
    from_account INTEGER,
    to_account INTEGER,
    amount NUMERIC
)
AS $$
BEGIN
    UPDATE accounts SET balance = balance - amount WHERE id = from_account;
    UPDATE accounts SET balance = balance + amount WHERE id = to_account;
    COMMIT;
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        RAISE;
END;
$$ LANGUAGE plpgsql;
```

---

## 7. 함수 vs 프로시저

| 특성 | 함수 (FUNCTION) | 프로시저 (PROCEDURE) |
|------|-----------------|----------------------|
| 반환값 | 반드시 반환 | 반환 없음 |
| SELECT에서 | 사용 가능 | 사용 불가 |
| 호출 방법 | SELECT func() | CALL proc() |
| 트랜잭션 | 외부 트랜잭션 | 자체 트랜잭션 가능 |
| COMMIT/ROLLBACK | 불가능 | 가능 |

---

## 8. 실습 예제

### 실습 1: 유틸리티 함수

```sql
-- 1. 이메일 도메인 추출
CREATE FUNCTION get_email_domain(email VARCHAR)
RETURNS VARCHAR
AS $$
BEGIN
    RETURN SPLIT_PART(email, '@', 2);
END;
$$ LANGUAGE plpgsql;

SELECT get_email_domain('user@gmail.com');  -- gmail.com

-- 2. 나이 계산
CREATE FUNCTION calculate_age(birth_date DATE)
RETURNS INTEGER
AS $$
BEGIN
    RETURN EXTRACT(YEAR FROM AGE(CURRENT_DATE, birth_date));
END;
$$ LANGUAGE plpgsql;

SELECT calculate_age('1990-05-15');  -- 34 (2024년 기준)

-- 3. 가격 포맷팅
CREATE FUNCTION format_price(price NUMERIC)
RETURNS VARCHAR
AS $$
BEGIN
    RETURN TO_CHAR(price, 'FM999,999,999') || '원';
END;
$$ LANGUAGE plpgsql;

SELECT format_price(1500000);  -- 1,500,000원
```

### 실습 2: 비즈니스 로직 함수

```sql
-- 1. 주문 총액 계산
CREATE FUNCTION calculate_order_total(p_order_id INTEGER)
RETURNS NUMERIC
AS $$
DECLARE
    total NUMERIC;
BEGIN
    SELECT SUM(p.price * oi.quantity)
    INTO total
    FROM order_items oi
    JOIN products p ON oi.product_id = p.id
    WHERE oi.order_id = p_order_id;

    RETURN COALESCE(total, 0);
END;
$$ LANGUAGE plpgsql;

-- 2. 사용자 등급 결정
CREATE FUNCTION get_user_tier(p_user_id INTEGER)
RETURNS VARCHAR
AS $$
DECLARE
    total_spent NUMERIC;
BEGIN
    SELECT COALESCE(SUM(amount), 0)
    INTO total_spent
    FROM orders
    WHERE user_id = p_user_id;

    RETURN CASE
        WHEN total_spent >= 1000000 THEN 'VIP'
        WHEN total_spent >= 500000 THEN 'Gold'
        WHEN total_spent >= 100000 THEN 'Silver'
        ELSE 'Bronze'
    END;
END;
$$ LANGUAGE plpgsql;
```

### 실습 3: 데이터 검증 함수

```sql
-- 1. 이메일 유효성 검사
CREATE FUNCTION is_valid_email(email VARCHAR)
RETURNS BOOLEAN
AS $$
BEGIN
    RETURN email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$';
END;
$$ LANGUAGE plpgsql;

SELECT is_valid_email('test@email.com');  -- true
SELECT is_valid_email('invalid-email');   -- false

-- 2. 전화번호 포맷팅
CREATE FUNCTION format_phone(phone VARCHAR)
RETURNS VARCHAR
AS $$
DECLARE
    cleaned VARCHAR;
BEGIN
    cleaned := REGEXP_REPLACE(phone, '[^0-9]', '', 'g');
    IF LENGTH(cleaned) = 11 THEN
        RETURN SUBSTRING(cleaned, 1, 3) || '-' ||
               SUBSTRING(cleaned, 4, 4) || '-' ||
               SUBSTRING(cleaned, 8, 4);
    ELSE
        RETURN phone;
    END IF;
END;
$$ LANGUAGE plpgsql;

SELECT format_phone('01012345678');  -- 010-1234-5678
```

---

## 9. 함수 관리

### 함수 목록 확인

```sql
-- psql 명령
\df

-- SQL 쿼리
SELECT routine_name, routine_type
FROM information_schema.routines
WHERE routine_schema = 'public';
```

### 함수 정의 확인

```sql
-- 함수 소스 코드 보기
\sf function_name

-- 또는
SELECT prosrc FROM pg_proc WHERE proname = 'function_name';
```

### 함수 수정

```sql
CREATE OR REPLACE FUNCTION function_name(...)
RETURNS ...
AS $$
    -- 수정된 내용
$$ LANGUAGE plpgsql;
```

---

**이전**: [뷰와 인덱스](./09_Views_and_Indexes.md) | **다음**: [트랜잭션](./11_Transactions.md)

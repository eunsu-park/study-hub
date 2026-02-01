# 함수와 프로시저

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

## 다음 단계

[11_Transactions.md](./11_Transactions.md)에서 트랜잭션과 동시성 제어를 배워봅시다!

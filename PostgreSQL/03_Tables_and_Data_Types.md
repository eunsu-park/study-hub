# 테이블과 데이터타입

## 1. 테이블 기본 개념

테이블은 데이터를 행(row)과 열(column)로 구성하여 저장하는 구조입니다.

```
┌──────────────────────────────────────────────────────┐
│                    users 테이블                       │
├────────┬──────────┬─────────────────┬───────────────┤
│   id   │   name   │      email      │  created_at   │
├────────┼──────────┼─────────────────┼───────────────┤
│   1    │  김철수  │ kim@email.com   │ 2024-01-15    │
│   2    │  이영희  │ lee@email.com   │ 2024-01-16    │
│   3    │  박민수  │ park@email.com  │ 2024-01-17    │
└────────┴──────────┴─────────────────┴───────────────┘
  컬럼(Column)           ↑ 각 행은 하나의 레코드
```

---

## 2. 테이블 생성

### 기본 문법

```sql
CREATE TABLE 테이블명 (
    컬럼명1 데이터타입 [제약조건],
    컬럼명2 데이터타입 [제약조건],
    ...
);
```

### 기본 예제

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    age INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 존재하지 않는 경우에만 생성

```sql
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);
```

---

## 3. 숫자 데이터타입

### 정수형

| 타입 | 크기 | 범위 |
|------|------|------|
| `SMALLINT` | 2 bytes | -32,768 ~ 32,767 |
| `INTEGER` (INT) | 4 bytes | -2,147,483,648 ~ 2,147,483,647 |
| `BIGINT` | 8 bytes | -9경 ~ 9경 |

```sql
CREATE TABLE products (
    id INTEGER,
    quantity SMALLINT,
    total_sold BIGINT
);
```

### 자동 증가 (Serial)

| 타입 | 범위 |
|------|------|
| `SMALLSERIAL` | 1 ~ 32,767 |
| `SERIAL` | 1 ~ 2,147,483,647 |
| `BIGSERIAL` | 1 ~ 9경 |

```sql
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,  -- 자동으로 1, 2, 3, ... 증가
    order_date DATE
);

-- PostgreSQL 10+ 에서는 IDENTITY 권장
CREATE TABLE orders (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_date DATE
);
```

### 실수형

| 타입 | 설명 |
|------|------|
| `REAL` | 4 bytes, 6자리 정밀도 |
| `DOUBLE PRECISION` | 8 bytes, 15자리 정밀도 |
| `NUMERIC(p, s)` | 정확한 숫자 (p: 전체 자릿수, s: 소수점 자릿수) |
| `DECIMAL(p, s)` | NUMERIC과 동일 |

```sql
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    price NUMERIC(10, 2),      -- 최대 10자리, 소수점 2자리 (예: 99999999.99)
    weight REAL,               -- 부동 소수점
    rating DOUBLE PRECISION    -- 더 정밀한 부동 소수점
);

INSERT INTO products (price, weight, rating) VALUES
(19900.00, 1.5, 4.7);
```

---

## 4. 문자 데이터타입

| 타입 | 설명 |
|------|------|
| `CHAR(n)` | 고정 길이 문자열 (남는 공간은 공백으로 채움) |
| `VARCHAR(n)` | 가변 길이 문자열 (최대 n자) |
| `TEXT` | 길이 제한 없는 문자열 |

```sql
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    code CHAR(10),           -- 항상 10자 (코드 등에 사용)
    title VARCHAR(200),      -- 최대 200자
    content TEXT             -- 길이 제한 없음
);
```

### VARCHAR vs TEXT

```sql
-- 실질적으로 큰 차이 없음. PostgreSQL에서는 TEXT 선호하는 경우도 많음
CREATE TABLE posts (
    title VARCHAR(255),  -- 길이 제한이 필요한 경우
    body TEXT            -- 길이 제한이 필요 없는 경우
);
```

---

## 5. 날짜/시간 데이터타입

| 타입 | 설명 | 예시 |
|------|------|------|
| `DATE` | 날짜만 | 2024-01-15 |
| `TIME` | 시간만 | 14:30:00 |
| `TIMESTAMP` | 날짜 + 시간 | 2024-01-15 14:30:00 |
| `TIMESTAMPTZ` | 날짜 + 시간 + 타임존 | 2024-01-15 14:30:00+09 |
| `INTERVAL` | 시간 간격 | 2 days 3 hours |

```sql
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    event_name VARCHAR(100),
    event_date DATE,
    start_time TIME,
    created_at TIMESTAMP DEFAULT NOW(),
    scheduled_at TIMESTAMPTZ,
    duration INTERVAL
);

INSERT INTO events (event_name, event_date, start_time, duration) VALUES
('회의', '2024-01-20', '14:00:00', '2 hours'),
('워크샵', '2024-01-25', '09:00:00', '1 day');
```

### 날짜/시간 함수

```sql
-- 현재 시간
SELECT NOW();                    -- 2024-01-15 14:30:00.123456+09
SELECT CURRENT_DATE;             -- 2024-01-15
SELECT CURRENT_TIME;             -- 14:30:00.123456+09
SELECT CURRENT_TIMESTAMP;        -- NOW()와 동일

-- 날짜 연산
SELECT NOW() + INTERVAL '1 day';
SELECT NOW() - INTERVAL '2 hours';
SELECT '2024-01-20'::DATE - '2024-01-15'::DATE;  -- 5 (일수)

-- 날짜 추출
SELECT EXTRACT(YEAR FROM NOW());
SELECT EXTRACT(MONTH FROM NOW());
SELECT EXTRACT(DOW FROM NOW());  -- 요일 (0=일요일)
```

---

## 6. 불리언 데이터타입

| 값 | TRUE | FALSE | NULL |
|------|------|-------|------|
| 입력 | true, 't', 'yes', 'y', '1' | false, 'f', 'no', 'n', '0' | null |

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    is_admin BOOLEAN DEFAULT false
);

INSERT INTO users (name, is_active, is_admin) VALUES
('김철수', true, false),
('관리자', true, true);

SELECT * FROM users WHERE is_active = true;
SELECT * FROM users WHERE NOT is_admin;
```

---

## 7. JSON 데이터타입

| 타입 | 설명 |
|------|------|
| `JSON` | JSON 텍스트 저장 (매번 파싱) |
| `JSONB` | JSON 바이너리 저장 (인덱싱 가능, 권장) |

```sql
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    attributes JSONB
);

INSERT INTO products (name, attributes) VALUES
('노트북', '{"brand": "Samsung", "ram": 16, "storage": "512GB"}'),
('마우스', '{"brand": "Logitech", "wireless": true, "color": "black"}');

-- JSON 데이터 조회
SELECT name, attributes->>'brand' AS brand FROM products;
SELECT name, attributes->'ram' AS ram FROM products;

-- JSON 조건 검색
SELECT * FROM products WHERE attributes->>'brand' = 'Samsung';
SELECT * FROM products WHERE (attributes->>'ram')::int >= 16;

-- JSON 배열
INSERT INTO products (name, attributes) VALUES
('키보드', '{"brand": "Keychron", "colors": ["white", "black", "gray"]}');

SELECT attributes->'colors'->0 FROM products WHERE name = '키보드';  -- "white"
```

---

## 8. 기타 데이터타입

### UUID

```sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE sessions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO sessions (user_id) VALUES (1);
-- id: 550e8400-e29b-41d4-a716-446655440000
```

### 배열 (Array)

```sql
CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200),
    tags TEXT[]
);

INSERT INTO posts (title, tags) VALUES
('PostgreSQL 입문', ARRAY['database', 'postgresql', 'sql']),
('Docker 시작하기', '{"docker", "container", "devops"}');

-- 배열 조회
SELECT title, tags[1] FROM posts;  -- 첫 번째 요소

-- 배열 포함 여부
SELECT * FROM posts WHERE 'docker' = ANY(tags);
SELECT * FROM posts WHERE tags @> ARRAY['sql'];
```

### ENUM

```sql
CREATE TYPE mood AS ENUM ('happy', 'sad', 'neutral');

CREATE TABLE user_moods (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    current_mood mood
);

INSERT INTO user_moods (user_id, current_mood) VALUES (1, 'happy');
```

---

## 9. 제약조건 (Constraints)

### PRIMARY KEY

```sql
-- 단일 컬럼 기본키
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100)
);

-- 복합 기본키
CREATE TABLE order_items (
    order_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    PRIMARY KEY (order_id, product_id)
);
```

### NOT NULL

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,  -- NULL 허용 안함
    email VARCHAR(255) NOT NULL
);
```

### UNIQUE

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,  -- 중복 불가
    phone VARCHAR(20) UNIQUE             -- 중복 불가 (NULL은 여러 개 가능)
);

-- 복합 유니크
CREATE TABLE memberships (
    user_id INTEGER,
    group_id INTEGER,
    UNIQUE (user_id, group_id)
);
```

### DEFAULT

```sql
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    status VARCHAR(20) DEFAULT 'pending',
    quantity INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO orders DEFAULT VALUES;  -- 모든 컬럼 기본값 사용
```

### CHECK

```sql
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    price NUMERIC(10, 2) CHECK (price > 0),
    quantity INTEGER CHECK (quantity >= 0),
    discount NUMERIC(3, 2) CHECK (discount >= 0 AND discount <= 1)
);

-- 이름 있는 제약조건
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    age INTEGER,
    salary NUMERIC(10, 2),
    CONSTRAINT valid_age CHECK (age >= 18 AND age <= 100),
    CONSTRAINT positive_salary CHECK (salary > 0)
);
```

### FOREIGN KEY

```sql
-- 부모 테이블
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

-- 자식 테이블
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    category_id INTEGER REFERENCES categories(id)
);

-- 상세 옵션
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    category_id INTEGER,
    FOREIGN KEY (category_id) REFERENCES categories(id)
        ON DELETE CASCADE      -- 부모 삭제 시 자식도 삭제
        ON UPDATE CASCADE      -- 부모 수정 시 자식도 수정
);
```

### ON DELETE / ON UPDATE 옵션

| 옵션 | 설명 |
|------|------|
| `CASCADE` | 부모와 함께 삭제/수정 |
| `SET NULL` | NULL로 설정 |
| `SET DEFAULT` | 기본값으로 설정 |
| `RESTRICT` | 삭제/수정 불가 (기본값) |
| `NO ACTION` | RESTRICT와 유사 |

---

## 10. 테이블 수정

### 컬럼 추가

```sql
ALTER TABLE users ADD COLUMN phone VARCHAR(20);
ALTER TABLE users ADD COLUMN is_verified BOOLEAN DEFAULT false;
```

### 컬럼 삭제

```sql
ALTER TABLE users DROP COLUMN phone;
ALTER TABLE users DROP COLUMN IF EXISTS phone;
```

### 컬럼 타입 변경

```sql
ALTER TABLE users ALTER COLUMN name TYPE VARCHAR(200);
ALTER TABLE users ALTER COLUMN age TYPE SMALLINT;

-- 데이터 변환이 필요한 경우
ALTER TABLE users ALTER COLUMN price TYPE INTEGER USING price::INTEGER;
```

### 컬럼 이름 변경

```sql
ALTER TABLE users RENAME COLUMN name TO full_name;
```

### 제약조건 추가/삭제

```sql
-- NOT NULL 추가
ALTER TABLE users ALTER COLUMN email SET NOT NULL;

-- NOT NULL 제거
ALTER TABLE users ALTER COLUMN email DROP NOT NULL;

-- DEFAULT 설정
ALTER TABLE users ALTER COLUMN status SET DEFAULT 'active';

-- DEFAULT 제거
ALTER TABLE users ALTER COLUMN status DROP DEFAULT;

-- 제약조건 추가
ALTER TABLE users ADD CONSTRAINT users_email_unique UNIQUE (email);
ALTER TABLE users ADD CONSTRAINT valid_age CHECK (age >= 0);

-- 제약조건 삭제
ALTER TABLE users DROP CONSTRAINT users_email_unique;
```

### 테이블 이름 변경

```sql
ALTER TABLE users RENAME TO members;
```

---

## 11. 테이블 삭제

```sql
-- 기본 삭제
DROP TABLE users;

-- 존재하는 경우에만 삭제
DROP TABLE IF EXISTS users;

-- 의존 객체와 함께 삭제
DROP TABLE users CASCADE;
```

---

## 12. 테이블 정보 확인

```sql
-- 테이블 목록
\dt

-- 테이블 구조
\d users

-- 상세 정보
\d+ users

-- SQL 쿼리로 확인
SELECT
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_name = 'users';
```

---

## 13. 실습 예제

### 실습: 온라인 쇼핑몰 테이블 설계

```sql
-- 1. 사용자 테이블
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(100) NOT NULL,
    phone VARCHAR(20),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 2. 카테고리 테이블
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES categories(id),
    created_at TIMESTAMP DEFAULT NOW()
);

-- 3. 상품 테이블
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    category_id INTEGER REFERENCES categories(id),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    price NUMERIC(12, 2) NOT NULL CHECK (price >= 0),
    stock INTEGER DEFAULT 0 CHECK (stock >= 0),
    attributes JSONB,
    is_available BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 4. 주문 테이블
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    status VARCHAR(20) DEFAULT 'pending' CHECK (
        status IN ('pending', 'paid', 'shipped', 'delivered', 'cancelled')
    ),
    total_amount NUMERIC(12, 2) NOT NULL,
    shipping_address TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 5. 주문 상세 테이블
CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id) ON DELETE CASCADE,
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_price NUMERIC(12, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 테이블 구조 확인
\dt
\d products
```

---

## 다음 단계

[04_CRUD_기본.md](./04_CRUD_기본.md)에서 데이터 삽입, 조회, 수정, 삭제를 배워봅시다!

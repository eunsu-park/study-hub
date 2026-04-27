# 테이블과 데이터타입

**이전**: [데이터베이스 관리](./02_Database_Management.md) | **다음**: [CRUD 기본](./04_CRUD_Basics.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 적절한 컬럼 정의와 함께 `CREATE TABLE`을 사용하여 테이블을 생성할 수 있습니다
2. PostgreSQL의 숫자형 타입(INTEGER, NUMERIC, SERIAL)을 구별하고 적합한 타입을 선택할 수 있습니다
3. 문자형 타입(CHAR, VARCHAR, TEXT)과 날짜/시간 타입(DATE, TIMESTAMP, TIMESTAMPTZ)을 비교할 수 있습니다
4. BOOLEAN, JSONB, UUID, 배열(array), ENUM을 포함한 특수 데이터 타입을 적용할 수 있습니다
5. 데이터 무결성을 강제하기 위한 제약조건(PRIMARY KEY, NOT NULL, UNIQUE, CHECK, FOREIGN KEY)을 구현할 수 있습니다
6. ALTER TABLE을 사용하여 기존 테이블을 수정할 수 있습니다 (컬럼 추가/삭제, 타입 변경, 제약조건 관리)
7. 적절한 외래 키(Foreign Key) 관계를 갖춘 다중 테이블 스키마(schema)를 설계할 수 있습니다

---

테이블은 모든 관계형 데이터베이스의 근본적인 구성 요소입니다. 애플리케이션이 저장하는 모든 데이터 — 사용자 프로필, 상품 카탈로그, 금융 거래 — 는 결국 신중하게 선택된 컬럼, 데이터 타입(Data Type), 제약조건(Constraint)을 갖춘 테이블 안에 존재합니다. 설계 단계에서 스키마를 올바르게 정의하면, 미묘한 데이터 손상부터 느린 쿼리까지 나중에 발생할 수 있는 수많은 문제를 예방할 수 있습니다.

---

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

### 이론: 8 KB Page

PostgreSQL은 heap을 고정 크기 **page**(block이라고도 함) 단위로 읽고 씁니다. 기본 크기는 **8 KB**이며 컴파일 타임에 고정됩니다 — 모든 테이블 파일은 page 정수배이고, `shared_buffers`의 모든 buffer는 page 하나, WAL 갱신은 page를 추적합니다.

```
┌────────────────────────────────────────────────┐  byte 0
│ PageHeader (24 bytes)                          │
│  pd_lsn, pd_checksum, pd_lower, pd_upper, ...  │
├────────────────────────────────────────────────┤  pd_lower
│ ItemIdData[]  (각 4바이트, "line pointer")     │
│  ↓ 아래로 자람                                 │
├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤  free space
│                                                │
│ ↑ 위로 자람                                    │
│ Tuple (행)                                      │
├────────────────────────────────────────────────┤  pd_upper
│ Special space (인덱스에서 사용, heap에선 미사용)│
└────────────────────────────────────────────────┘  byte 8191
```

두 포인터(`pd_lower`와 `pd_upper`)가 free space 경계를 표시합니다. 행을 삽입하면 tuple은 `pd_upper - tuple_size`에 쓰이고 line pointer는 `pd_lower`에 쓰입니다. 두 포인터가 서로를 향해 자라며, 만나면 page가 가득 차서 PostgreSQL이 다음 page를 할당합니다.

#### A.1 line pointer가 존재하는 이유

인덱스는 행에 대한 안정적인 참조가 필요하지만, tuple은 삭제되거나, 갱신(새 버전 생성)되거나, VACUUM에 의해 이동될 수 있습니다. line pointer(`ItemId`)는 4바이트 간접 참조입니다 — 인덱스 항목은 `(page_number, line_pointer_index)`를 가리키고, line pointer는 page 내 실제 offset을 가리킵니다. tuple이 옮겨지면 line pointer만 갱신되고, 모든 인덱스는 계속 동작합니다.

#### A.2 `ctid` 시스템 컬럼

모든 행은 `SELECT`할 수 있는 `ctid`를 가집니다 — 정확히 `(page_number, line_pointer_index)`입니다. `ctid`는 update에 *대해 안정적이지 않습니다* — 행을 변경하는 UPDATE는 행을 재배치할 수 있고, 그러면 새 ctid를 갖게 됩니다(이전 line pointer는 새 line pointer로의 "redirect"가 됩니다).

## 2. 테이블 생성

### 이론: Tuple Layout

각 행은 다음과 같이 구성된 **tuple**입니다.

```
┌────────────────────────────────────────┐
│ HeapTupleHeader  (최소 23 bytes)        │
│  xmin, xmax, cmin, cmax, ctid,         │
│  t_infomask, t_hoff                    │
├────────────────────────────────────────┤
│ Null bitmap (옵션, 컬럼당 1 bit)        │
├────────────────────────────────────────┤
│ 8-byte 경계까지 alignment padding        │
├────────────────────────────────────────┤
│ Column 1 데이터                         │
│ Alignment padding                      │
│ Column 2 데이터                         │
│   ...                                  │
└────────────────────────────────────────┘
```

#### B.1 Header — MVCC와 visibility를 가능케 하는 부분

23바이트 header는 MVCC가 필요로 하는 visibility 메타데이터를 담습니다.

- **`xmin`** — 이 row version을 만든 트랜잭션 ID.
- **`xmax`** — 이 row version을 삭제/대체한 트랜잭션 ID(아직 살아 있으면 0).
- **`cmin`/`cmax`** — 트랜잭션 내부 command-id (자기 자신에 대한 visibility).
- **`ctid`** — 이 tuple의 물리 위치 (또는 update된 경우 다음 버전의 위치).
- **`t_infomask`** — 플래그 비트들 — 행에 null이 있나? VACUUM에 의해 frozen되었나? `xmax`가 실제로는 multixact인가?

header는 1-컬럼 테이블이든 100-컬럼 테이블이든 동일한 크기입니다. 좁은 행이 많을수록 넓은 행 몇 개보다 비례적으로 overhead가 큽니다.

#### B.2 Null bitmap — 적어도 한 컬럼이 null일 때만 할당

bitmap은 컬럼당 1 bit이며 8바이트로 패딩됩니다. PostgreSQL은 행의 모든 컬럼이 not-null이면 *할당하지 않습니다*(`t_infomask`의 `HEAP_HASNULL` 비트가 플래그). nullable 컬럼이 많고 대부분의 행에 null이 있는 테이블에서는 눈에 띄는 공간 절약입니다.

#### B.3 Alignment

PostgreSQL의 모든 데이터 타입은 **alignment requirement**를 가집니다 — `int2`는 2바이트, `int4`는 4바이트, `int8`과 `timestamp`는 8바이트, `text`는 4바이트(length word) 정렬. tuple builder는 alignment를 만족시키기 위해 컬럼 사이에 padding 바이트를 삽입합니다. 이것이 가장 흔한 저장 함정 중 하나의 원천입니다 — §D 참조.

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

### 이론: 타입별 저장 비용과 alignment trap

PostgreSQL 타입의 대표적 일부와 그 저장 크기:

| 타입 | 바이트 | Alignment | 비고 |
|------|--------|-----------|------|
| `boolean` | 1 | 1 | |
| `smallint` | 2 | 2 | |
| `integer` | 4 | 4 | |
| `bigint` | 8 | 8 | |
| `numeric(p, s)` | 가변 (~5–8 + 자릿수당 2) | 4 | 임의 정밀도, int보다 느림 |
| `real` (float4) | 4 | 4 | |
| `double precision` (float8) | 8 | 8 | |
| `date` | 4 | 4 | |
| `time` | 8 | 8 | |
| `timestamp`/`timestamptz` | 8 | 8 | 둘 다 8바이트, tz 정보는 세션 단위로 적용되며 저장되지 않음 |
| `interval` | 16 | 8 | |
| `uuid` | 16 | 4 | |
| `text`/`varchar(n)`/`bytea` | 1바이트 length header + payload (TOAST 가능) | 4 | `varchar(n)`은 길이 강제, 저장은 `text`와 동일 |
| `char(n)` | n 바이트 (공백 패딩) | 4 | 거의 항상 잘못된 선택, `text` 사용 권장 |
| `json` | 텍스트 + length header | 4 | 원시 텍스트로 저장 |
| `jsonb` | 바이너리 parse tree + length header | 4 | TOAST 가능, GIN 지원 |

#### D.1 컬럼 순서 함정

alignment padding 때문에, `CREATE TABLE`에서의 컬럼 *순서*가 행 크기에 영향을 줍니다. 예:

```sql
CREATE TABLE bad  (a int2, b int8, c int2);  -- 2 + 6 pad + 8 + 2 + 6 pad = 24 bytes
CREATE TABLE good (b int8, a int2, c int2);  -- 8 + 2 + 2 + 4 pad         = 16 bytes
```

"bad" 버전은 2바이트 필드 뒤에 오는 `int8`의 8바이트 alignment를 만족시키느라 행당 8바이트를 낭비합니다. **alignment가 큰 것부터 작은 것 순으로 컬럼을 배치**하면 padding을 최소화할 수 있습니다. 넓은 테이블에서는 컬럼 순서 변경만으로 heap을 10-20% 줄일 수 있습니다.

#### D.2 가변 길이 타입과 1바이트 short header

`text`, `bytea`, `varchar`는 length prefix를 사용합니다. PostgreSQL에는 영리한 최적화가 있습니다 — 126바이트까지의 값에는 표준 4바이트 대신 **1바이트 short header**를 사용합니다. 따라서 짧은 문자열이 많은 컬럼은 4바이트 overhead가 시사하는 것보다 훨씬 저렴합니다.

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

-- IDENTITY(SQL 표준)가 SERIAL보다 PG 10+에서 선호됨 — SERIAL은 느슨하게 결합된
-- 별도 시퀀스를 생성하지만, IDENTITY는 시퀀스를 컬럼 생명주기에 연결하여
-- 시퀀스를 깨뜨리는 수동 삽입을 방지
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
-- 금융/화폐 데이터에는 NUMERIC 사용 — 정확한 계산 (반올림 오류 없음).
-- REAL/DOUBLE PRECISION은 빠르지만 근사값; float에서 0.1 + 0.2 ≠ 0.3
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    price NUMERIC(10, 2),      -- 최대 10자리, 소수점 2자리 (예: 99999999.99)
    weight REAL,               -- 부동 소수점 (반올림이 허용되는 측정값에 사용)
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

### 이론: TOAST — Oversized-Attribute Storage Technique

행은 page 하나(8 KB)를 넘을 수 없습니다. 그러나 `text`나 `bytea` 값은 쉽게 메가바이트 단위가 될 수 있습니다. PostgreSQL은 이를 **TOAST**(The Oversized-Attribute Storage Technique)로 해결합니다.

#### C.1 TOAST 결정 흐름

tuple이 **TOAST threshold**(`TOAST_TUPLE_THRESHOLD`, 기본 ~2 KB, 즉 page의 ~1/4)를 초과할 때, planner는 가장 큰 TOAST 가능 컬럼에 대해 다음 루프를 돕니다.

1. 값을 **압축**(최신 버전에서는 PGLZ 또는 LZ4). 이제 들어맞으면 inline으로 저장.
2. 그래도 너무 크면, **~2 KB 청크로 슬라이스**해서 청크들을 테이블의 TOAST 테이블(`CREATE TABLE` 시 자동 생성되며 이름은 `pg_toast.pg_toast_<oid>`)에 저장.
3. 본 행에는 청크들을 OID와 총 길이로 참조하는 작은 **TOAST pointer**(18바이트)만 저장.

TOAST 가능 컬럼마다 `ALTER TABLE ... SET STORAGE`로 변경할 수 있는 **storage strategy**가 있습니다.

| Strategy | 압축? | Out-of-line? |
|----------|------|--------------|
| `PLAIN`  | 아니오 | 아니오 (TOAST 불가능 타입에만) |
| `EXTENDED` (TEXT/BYTEA의 기본값) | 예 | 예 |
| `EXTERNAL` | 아니오 | 예 (`substring` 호출이 빠름) |
| `MAIN`   | 예 | 압축 후에도 너무 크면만 |

#### C.2 실전에서 TOAST가 중요한 이유

`SELECT id FROM big_log_table`은 모든 행에 1 MB body가 있어도 빠릅니다. body는 TOAST 테이블에 살고, 명시적으로 projection되지 않으면 읽히지 않기 때문입니다. 반대로 `SELECT body`는 TOAST 테이블로의 join을 트리거합니다 — 보이지 않게, 그러나 I/O를 추가합니다. "필요한 것만 select하라"가 단순한 코드 스타일이 아닌 데이터베이스 차원의 이유입니다.

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

-- 비즈니스 규칙에 따라 ON DELETE 동작 선택:
-- CASCADE: 부모 없이 자식 데이터가 무의미할 때 (예: 주문 없는 주문항목)
-- SET NULL: 자식이 독립적으로 존재 가능할 때 (예: 카테고리 삭제 시 상품)
-- RESTRICT: 자식이 있으면 삭제 차단 (가장 안전한 기본값)
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

**이전**: [데이터베이스 관리](./02_Database_Management.md) | **다음**: [CRUD 기본](./04_CRUD_Basics.md)

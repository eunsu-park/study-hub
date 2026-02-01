# 뷰와 인덱스

## 1. 뷰 (VIEW) 개념

뷰는 저장된 쿼리로, 가상의 테이블처럼 사용할 수 있습니다.

```
┌─────────────────────────────────────────────────────────┐
│                       VIEW                              │
│  ┌───────────────────────────────────────────────────┐ │
│  │  SELECT u.name, SUM(o.amount) AS total           │ │
│  │  FROM users u JOIN orders o ON u.id = o.user_id  │ │
│  │  GROUP BY u.id, u.name                           │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                          ↓
              SELECT * FROM user_sales;
                    (간단하게 사용)
```

---

## 2. 뷰 생성

### 기본 뷰 생성

```sql
-- 활성 사용자만 보는 뷰
CREATE VIEW active_users AS
SELECT id, name, email
FROM users
WHERE is_active = true;

-- 뷰 사용
SELECT * FROM active_users;
SELECT * FROM active_users WHERE name LIKE '김%';
```

### 복잡한 쿼리를 뷰로

```sql
-- 사용자별 주문 통계 뷰
CREATE VIEW user_order_stats AS
SELECT
    u.id AS user_id,
    u.name,
    u.email,
    COUNT(o.id) AS order_count,
    COALESCE(SUM(o.amount), 0) AS total_amount,
    MAX(o.order_date) AS last_order_date
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name, u.email;

-- 간단하게 조회
SELECT * FROM user_order_stats WHERE order_count > 0;
```

### OR REPLACE

```sql
-- 뷰가 있으면 교체, 없으면 생성
CREATE OR REPLACE VIEW active_users AS
SELECT id, name, email, created_at
FROM users
WHERE is_active = true;
```

---

## 3. 뷰 수정 및 삭제

### 뷰 삭제

```sql
DROP VIEW active_users;
DROP VIEW IF EXISTS active_users;

-- 의존 객체와 함께 삭제
DROP VIEW active_users CASCADE;
```

### 뷰 이름 변경

```sql
ALTER VIEW active_users RENAME TO enabled_users;
```

---

## 4. 뷰의 장점

```sql
-- 1. 쿼리 단순화
-- 복잡한 조인을 뷰로 만들어 놓으면
SELECT * FROM user_order_stats WHERE total_amount > 1000000;

-- 2. 보안 (특정 컬럼만 노출)
CREATE VIEW public_users AS
SELECT id, name FROM users;  -- 이메일, 비밀번호 제외

-- 3. 논리적 데이터 독립성
-- 테이블 구조가 바뀌어도 뷰만 수정하면 됨
```

---

## 5. 업데이트 가능한 뷰

단순한 뷰는 INSERT, UPDATE, DELETE가 가능합니다.

```sql
-- 단순 뷰 (업데이트 가능)
CREATE VIEW seoul_users AS
SELECT * FROM users WHERE city = '서울';

-- 뷰를 통한 업데이트
UPDATE seoul_users SET name = '김서울' WHERE id = 1;

-- 뷰를 통한 삽입
INSERT INTO seoul_users (name, email, city)
VALUES ('새사용자', 'new@email.com', '서울');
```

### WITH CHECK OPTION

```sql
-- 뷰 조건을 벗어나는 데이터 삽입/수정 방지
CREATE VIEW seoul_users AS
SELECT * FROM users WHERE city = '서울'
WITH CHECK OPTION;

-- 오류 발생 (city가 '부산'이므로)
INSERT INTO seoul_users (name, email, city)
VALUES ('부산사람', 'busan@email.com', '부산');
```

---

## 6. Materialized View (구체화된 뷰)

결과를 물리적으로 저장하는 뷰입니다.

### 생성

```sql
CREATE MATERIALIZED VIEW monthly_sales AS
SELECT
    DATE_TRUNC('month', order_date) AS month,
    COUNT(*) AS order_count,
    SUM(amount) AS total_amount
FROM orders
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY month;
```

### 조회

```sql
SELECT * FROM monthly_sales;
```

### 새로고침 (데이터 갱신)

```sql
-- 전체 새로고침 (테이블 잠금)
REFRESH MATERIALIZED VIEW monthly_sales;

-- 동시 접근 허용 새로고침 (UNIQUE 인덱스 필요)
REFRESH MATERIALIZED VIEW CONCURRENTLY monthly_sales;
```

### 삭제

```sql
DROP MATERIALIZED VIEW monthly_sales;
```

### 일반 뷰 vs Materialized View

| 특성 | VIEW | MATERIALIZED VIEW |
|------|------|-------------------|
| 데이터 저장 | X | O |
| 실시간 반영 | O | X (REFRESH 필요) |
| 조회 속도 | 느림 (매번 실행) | 빠름 (저장된 결과) |
| 저장 공간 | 없음 | 필요 |

---

## 7. 인덱스 (INDEX) 개념

인덱스는 데이터 검색 속도를 높이는 자료구조입니다.

```
테이블 (순차 검색):
┌─────────────────────────────────────────────┐
│ 1, 2, 3, 4, 5, 6, ... 999998, 999999, 1000000
└─────────────────────────────────────────────┘
  → 최악의 경우 1,000,000번 비교

인덱스 (B-tree):
           ┌─── [500000] ───┐
           │                │
    ┌─[250000]─┐      ┌─[750000]─┐
    │          │      │          │
  [125K]    [375K]  [625K]    [875K]
  → 최대 약 20번 비교로 찾음
```

---

## 8. 인덱스 생성

### 기본 인덱스

```sql
-- 단일 컬럼 인덱스
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_orders_user_id ON orders(user_id);

-- 복합 인덱스 (다중 컬럼)
CREATE INDEX idx_orders_user_date ON orders(user_id, order_date);
```

### 유니크 인덱스

```sql
CREATE UNIQUE INDEX idx_users_email_unique ON users(email);
```

### 부분 인덱스 (조건부)

```sql
-- 활성 사용자만 인덱싱
CREATE INDEX idx_active_users ON users(email) WHERE is_active = true;

-- NULL이 아닌 값만
CREATE INDEX idx_orders_shipped ON orders(shipped_date) WHERE shipped_date IS NOT NULL;
```

### 표현식 인덱스

```sql
-- 소문자 변환 결과에 인덱스
CREATE INDEX idx_users_lower_email ON users(LOWER(email));

-- 사용
SELECT * FROM users WHERE LOWER(email) = 'kim@email.com';
```

---

## 9. 인덱스 종류

### B-tree (기본)

```sql
-- 기본 인덱스 (B-tree)
CREATE INDEX idx_products_price ON products(price);

-- 범위 검색, 정렬, 동등 비교에 효과적
SELECT * FROM products WHERE price BETWEEN 1000 AND 5000;
SELECT * FROM products ORDER BY price;
```

### Hash

```sql
-- 동등 비교에만 효과적
CREATE INDEX idx_users_email_hash ON users USING hash(email);

-- 효과적
SELECT * FROM users WHERE email = 'kim@email.com';

-- Hash 인덱스 사용 불가
SELECT * FROM users WHERE email LIKE 'kim%';
```

### GIN (Generalized Inverted Index)

```sql
-- 배열, JSON, 전문 검색에 사용
CREATE INDEX idx_products_tags ON products USING gin(tags);
CREATE INDEX idx_products_attrs ON products USING gin(attributes);

-- 배열 검색
SELECT * FROM products WHERE tags @> ARRAY['sale'];

-- JSON 검색
SELECT * FROM products WHERE attributes @> '{"color": "red"}';
```

### GiST (Generalized Search Tree)

```sql
-- 기하학 데이터, 전문 검색에 사용
CREATE INDEX idx_locations_coords ON locations USING gist(coordinates);
```

---

## 10. 인덱스 관리

### 인덱스 목록 확인

```sql
-- psql 명령
\di

-- SQL 쿼리
SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'users';
```

### 인덱스 삭제

```sql
DROP INDEX idx_users_email;
DROP INDEX IF EXISTS idx_users_email;
```

### 인덱스 재구성

```sql
-- 인덱스 재빌드
REINDEX INDEX idx_users_email;

-- 테이블의 모든 인덱스 재빌드
REINDEX TABLE users;
```

---

## 11. EXPLAIN - 실행 계획 분석

### 기본 EXPLAIN

```sql
EXPLAIN SELECT * FROM users WHERE email = 'kim@email.com';
```

출력:
```
                        QUERY PLAN
----------------------------------------------------------
 Index Scan using idx_users_email on users  (cost=0.29..8.30 rows=1 width=100)
   Index Cond: (email = 'kim@email.com'::text)
```

### EXPLAIN ANALYZE (실제 실행)

```sql
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'kim@email.com';
```

출력:
```
                        QUERY PLAN
----------------------------------------------------------
 Index Scan using idx_users_email on users  (cost=0.29..8.30 rows=1 width=100)
                                             (actual time=0.025..0.027 rows=1 loops=1)
   Index Cond: (email = 'kim@email.com'::text)
 Planning Time: 0.085 ms
 Execution Time: 0.045 ms
```

### 주요 스캔 방식

| 스캔 방식 | 설명 | 성능 |
|-----------|------|------|
| Seq Scan | 전체 테이블 순차 스캔 | 느림 |
| Index Scan | 인덱스 사용 | 빠름 |
| Index Only Scan | 인덱스만으로 결과 반환 | 매우 빠름 |
| Bitmap Index Scan | 여러 인덱스 결합 | 중간 |

### EXPLAIN 예제

```sql
-- 인덱스 없이
EXPLAIN SELECT * FROM orders WHERE user_id = 1;
-- Seq Scan on orders  (비효율적)

-- 인덱스 생성 후
CREATE INDEX idx_orders_user_id ON orders(user_id);
EXPLAIN SELECT * FROM orders WHERE user_id = 1;
-- Index Scan using idx_orders_user_id  (효율적)
```

---

## 12. 인덱스 설계 가이드

### 인덱스를 만들어야 하는 경우

```sql
-- 1. WHERE 절에 자주 사용되는 컬럼
CREATE INDEX idx_users_city ON users(city);

-- 2. JOIN 조건에 사용되는 컬럼 (외래키)
CREATE INDEX idx_orders_user_id ON orders(user_id);

-- 3. ORDER BY에 사용되는 컬럼
CREATE INDEX idx_products_price ON products(price);

-- 4. 유니크 제약이 필요한 컬럼
CREATE UNIQUE INDEX idx_users_email ON users(email);
```

### 인덱스를 피해야 하는 경우

```sql
-- 1. 자주 변경되는 컬럼 (INSERT/UPDATE 성능 저하)
-- 2. 카디널리티가 낮은 컬럼 (예: 성별, boolean)
-- 3. 작은 테이블 (전체 스캔이 더 빠름)
-- 4. 거의 사용되지 않는 컬럼
```

### 복합 인덱스 컬럼 순서

```sql
-- 왼쪽 컬럼부터 사용됨
CREATE INDEX idx_orders_user_date ON orders(user_id, order_date);

-- 효과적
SELECT * FROM orders WHERE user_id = 1;
SELECT * FROM orders WHERE user_id = 1 AND order_date > '2024-01-01';

-- 비효과적 (첫 번째 컬럼 없음)
SELECT * FROM orders WHERE order_date > '2024-01-01';
```

---

## 13. 실습 예제

### 실습 1: 뷰 생성

```sql
-- 1. 상품 상세 뷰
CREATE VIEW product_details AS
SELECT
    p.id,
    p.name,
    c.name AS category,
    p.price,
    p.stock,
    CASE
        WHEN p.stock = 0 THEN '품절'
        WHEN p.stock < 10 THEN '재고 부족'
        ELSE '판매중'
    END AS status
FROM products p
JOIN categories c ON p.category_id = c.id;

-- 사용
SELECT * FROM product_details WHERE status = '품절';

-- 2. 월별 매출 뷰
CREATE VIEW monthly_revenue AS
SELECT
    DATE_TRUNC('month', order_date) AS month,
    COUNT(*) AS orders,
    SUM(amount) AS revenue
FROM orders
WHERE status = 'completed'
GROUP BY DATE_TRUNC('month', order_date);
```

### 실습 2: Materialized View

```sql
-- 카테고리별 통계 (무거운 쿼리)
CREATE MATERIALIZED VIEW category_stats AS
SELECT
    c.name AS category,
    COUNT(p.id) AS product_count,
    AVG(p.price) AS avg_price,
    SUM(oi.quantity) AS total_sold
FROM categories c
LEFT JOIN products p ON c.id = p.category_id
LEFT JOIN order_items oi ON p.id = oi.product_id
GROUP BY c.id, c.name;

-- 유니크 인덱스 생성 (CONCURRENTLY 새로고침용)
CREATE UNIQUE INDEX idx_category_stats ON category_stats(category);

-- 새로고침
REFRESH MATERIALIZED VIEW CONCURRENTLY category_stats;
```

### 실습 3: 인덱스와 성능 비교

```sql
-- 테스트 데이터 생성
CREATE TABLE test_orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    amount NUMERIC(10,2),
    order_date DATE
);

INSERT INTO test_orders (user_id, amount, order_date)
SELECT
    (random() * 1000)::INTEGER,
    (random() * 10000)::NUMERIC(10,2),
    '2024-01-01'::DATE + (random() * 365)::INTEGER
FROM generate_series(1, 100000);

-- 인덱스 없이 쿼리
EXPLAIN ANALYZE SELECT * FROM test_orders WHERE user_id = 500;

-- 인덱스 생성
CREATE INDEX idx_test_user_id ON test_orders(user_id);

-- 인덱스 있을 때 쿼리
EXPLAIN ANALYZE SELECT * FROM test_orders WHERE user_id = 500;
```

---

## 다음 단계

[10_Functions_and_Procedures.md](./10_Functions_and_Procedures.md)에서 사용자 정의 함수를 배워봅시다!

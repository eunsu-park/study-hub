# 뷰와 인덱스

**이전**: [서브쿼리와 CTE](./08_Subqueries_and_CTE.md) | **다음**: [함수와 프로시저](./10_Functions_and_Procedures.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 뷰(View)가 무엇인지 설명하고 그 장점(쿼리 단순화, 보안, 논리적 독립성)을 서술한다
2. DDL 구문을 사용해 뷰를 생성·교체·이름 변경·삭제한다
3. 일반 뷰와 구체화된 뷰(Materialized View)를 구분하고 각각 언제 사용할지 설명한다
4. 뷰가 업데이트 가능한지 판단하고 데이터 무결성을 위해 WITH CHECK OPTION을 적용한다
5. B-tree 인덱스(Index)가 조회를 어떻게 가속하는지 설명하고 쓰기 성능과의 트레이드오프를 서술한다
6. 단일 컬럼·복합·부분·표현식·유니크 인덱스를 생성한다
7. 인덱스 유형(B-tree, Hash, GIN, GiST)을 비교하고 주어진 사용 사례에 적합한 것을 선택한다
8. EXPLAIN / EXPLAIN ANALYZE 출력을 읽고 인덱스가 사용되고 있는지 확인한다

---

데이터베이스가 커질수록 두 가지 문제가 나타납니다. 복잡한 쿼리를 반복 작성하는 번거로움과, 전체 테이블을 순차 스캔하는 속도 저하가 그것입니다. 뷰(View)는 쿼리를 간단한 이름 뒤에 캡슐화하여 첫 번째 문제를 해결하고, 인덱스(Index)는 필요한 행에 대한 빠른 조회 경로를 PostgreSQL에 제공하여 두 번째 문제를 해결합니다. 이 두 도구를 함께 사용하면 유지보수하기 쉽고 성능이 뛰어난 데이터베이스 애플리케이션을 구축할 수 있습니다.

---

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

### 이론: View와 Materialized View

#### A.1 일반 view — 순수 rewriting

`CREATE VIEW v AS SELECT ...;`는 쿼리의 SQL 텍스트를 `pg_rewrite`에 저장합니다. 이후 `SELECT * FROM v WHERE x > 5;`를 작성하면, rewriter가 view 정의를 쿼리에 치환한 뒤 planning합니다. 결과는 planner가 전체로서 최적화하는 단일 결합 쿼리 — 술어 push-down이 view 경계를 통과합니다.

이로써 view는 runtime에 본질적으로 무료 — "view 평가" 단계가 없습니다. 비용은 view 없는 등가 쿼리의 비용과 같습니다.

#### A.2 Updatable view

단순한 view(base table 1개, aggregate 없음, DISTINCT 없음, GROUP BY 없음)는 자동으로 updatable입니다 — 그에 대한 `INSERT`, `UPDATE`, `DELETE`가 base table을 수정하는 것처럼 동작합니다. PostgreSQL은 통제된 쓰기 의미론이 필요한 view를 위해 `INSTEAD OF` 트리거와 `WITH CHECK OPTION`도 지원합니다.

#### A.3 Materialized view — 얼린 결과

`CREATE MATERIALIZED VIEW mv AS SELECT ...;`는 쿼리를 *1번* 실행하고, 결과 행을 실제 heap 파일에 저장하고, SQL을 기억합니다. 이후 `SELECT * FROM mv`는 저장된 행을 읽음 — 재계산 없음. 단점 — 행이 stale해짐.

`REFRESH MATERIALIZED VIEW mv;`가 쿼리를 다시 실행해서 저장된 행을 교체. 기본은 `ACCESS EXCLUSIVE` lock을 잡음, `REFRESH MATERIALIZED VIEW CONCURRENTLY mv;`는 새 행을 임시 테이블에 계산하고 atomic하게 교체해서 동시 읽기를 허용 (MV에 unique 인덱스 필요).

materialized view가 빛나는 경우 — 비교적 정적인 데이터에 대한 비싼 집계(분석 대시보드, 월별 요약). 다칠 때 — source 데이터가 refresh보다 빠르게 변하기 시작하는 순간.

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

### 이론: B-tree 내부 — 기본 그림 너머

5번 레슨에서 B-tree를 좌우로 연결된 leaf의 정렬된 page들로 소개했습니다. 전체 이야기:

#### B.1 Page split

leaf page가 가득 차고 새 키를 삽입해야 할 때, page가 split됩니다. PostgreSQL은 full leaf를 읽고, 새 page를 할당하고, 키를 대략 반반으로 분배하고, parent page를 양쪽을 가리키도록 갱신하고, 두 leaf를 다시 씁니다. parent도 가득 찼으면 split이 위로 cascade — root 자신이 split될 수 있고, 그러면 트리에 레벨이 추가됩니다.

Page split은 비쌉니다(3+ page 쓰기, 각각의 WAL 레코드). random key(예 — UUID)가 많이 삽입되는 테이블에서는 split이 도처에서 일어납니다. sequential key(예 — SERIAL)에서는 삽입이 항상 rightmost leaf로 가서 split이 거의 없음.

#### B.2 Fillfactor와 split 회피

`CREATE INDEX ... WITH (fillfactor = 70);`은 인덱스가 빌드될 때 각 page의 30%를 비워 두라고 PostgreSQL에 말합니다. 미래 삽입은 split 없이 그 빈 공간에 들어맞을 수 있습니다. B-tree의 기본 fillfactor는 90 — random key 인덱스에 이미 보수적.

monotonic하게 증가하는 키에서는 fillfactor를 100에 가깝게 — page 중간을 target하는 삽입이 없으므로 빈 공간은 낭비.

#### B.3 인덱스 bloat

행이 삭제되어도 인덱스 항목은 즉시 제거되지 않음 — "killed"로 표시되고 인덱스 스캔에서 건너뜀. 결국 leaf의 모든 항목이 killed될 수 있지만, page는 VACUUM(또는 autovacuum)이 실행될 때까지 해제되지 않음. update와 delete가 많은 워크로드는 인덱스를 살아 있는 항목이 정당화하는 것보다 훨씬 크게 만들 수 있음 — 그것이 **인덱스 bloat**.

해결책 — `REINDEX`가 인덱스를 처음부터 다시 빌드. PostgreSQL 12+는 긴 lock을 회피하는 `REINDEX CONCURRENTLY`를 지원.

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

### 이론: GIN, GiST, BRIN, Hash — B-tree를 사용하지 않을 때

| 인덱스 | 적합 | 비용 모양 |
|--------|------|----------|
| **B-tree** | 스칼라/orderable 타입의 equality + range | O(log N) lookup, O(log N) insert |
| **Hash** | 어떤 타입에든 equality만 | O(1) lookup, range 미지원 |
| **GIN** | multi-valued 타입(배열, JSONB, tsvector)의 containment | 삽입 느림, lookup 매우 빠름 |
| **GiST** | 기하/공간, 풀텍스트, 사용자 도메인 | generic framework — 정확한 비용은 operator class에 따라 다름 |
| **BRIN** | physical-order correlation이 있는 매우 큰 테이블 | 매우 작은 크기, 낮은 정밀도 |
| **SP-GiST** | non-balanced 트리 — trie, quadtree, k-d tree | 특수 자료구조 |

#### C.1 GIN — inverted index

GIN(Generalized Inverted Index)은 일반적인 구조를 뒤집습니다. "각 행에 대해 그 컬럼들을 나열" 대신, GIN은 "각 *값*에 대해 그것을 포함하는 행들을 나열"합니다. `{"tags": ["red", "fast"]}` 같은 문서가 있는 JSONB 컬럼에서 GIN은 `red → [row1, row5]`, `fast → [row1, row3, row7]` 같은 항목을 저장합니다.

이로써 containment 쿼리(`@>`, `?`, `?|`, `?&`)가 극도로 빠릅니다. 비용 — 삽입이 느리고(값의 element당 항목 1개), 고-cardinality element에서는 인덱스가 테이블 자체보다 클 수 있음.

`fastupdate` 옵션은 인덱스 갱신을 "pending list"로 미루어 VACUUM이 batch로 flush하게 함 — 삽입은 빨라지지만 lookup이 약간 느려짐(pending 항목이 linear 스캔되므로)이라는 tradeoff.

#### C.2 GiST — generalized search tree

GiST는 *framework*입니다 — 트리 모양, locking, concurrency를 제공하고, operator class가 타입별 술어를 공급합니다. PostGIS 공간 인덱스가 GiST. `LIKE '%substring%'`를 위한 `pg_trgm` 확장의 trigram 인덱스도 GiST. `btree_gist` 확장은 일반 타입을 GiST-index해서 다중 컬럼 인덱스에서 비-B-tree 가능 타입과 합성될 수 있게 함.

#### C.3 BRIN — block-range index

BRIN은 N개 page(기본 128)의 range당 작은 summary(min/max 또는 null bitmap)를 저장합니다. 1 GB 테이블에 대해 BRIN 인덱스는 16 KB일 수 있습니다. lookup은 summary를 참조해 candidate page range를 식별한 뒤 그 range를 풀로 스캔합니다.

BRIN은 **physical-order correlation**이 있을 때만 유리 — indexed 컬럼의 값이 디스크에 대략 정렬되어 있을 때(예 — append-only 로그 테이블의 timestamp). correlation이 없으면 min/max range가 심하게 겹쳐서 BRIN이 prune할 수 없음.

#### C.4 Hash — equality 전용

Hash 인덱스는 PG 10 이전에는 unlogged(crash-safe 아님)였고 사실상 사용 불가였습니다. 이제는 WAL-logged이고 긴 문자열 키의 순수 equality에서는 B-tree보다 빠릅니다. range, ORDER BY, 다중 컬럼은 미지원.

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

### 이론: 세 직교 인덱스 modifier

#### D.1 Partial 인덱스

`CREATE INDEX ... WHERE active = true;`는 술어에 매치되는 행만 인덱스. 대부분의 쿼리가 작은 부분집합(예 — 미처리 주문)을 신경 쓸 때 사용. 더 작고 빠르며, 쿼리의 술어가 partial 인덱스의 술어를 논리적으로 함의할 때만 사용됨.

#### D.2 Expression 인덱스

`CREATE INDEX ... ON t (LOWER(email));`은 raw 컬럼이 아니라 expression의 결과를 인덱스. `WHERE LOWER(email) = ?`을 sargable로 만들기 위해 필요(5번 레슨 §B.1). 비용 — 모든 insert와 update에서 expression이 재계산됨.

#### D.3 Covering 인덱스 — `INCLUDE`

`CREATE INDEX ... ON t (a, b) INCLUDE (c, d);`는 `c`와 `d`를 leaf 항목에 추가하지만 search key의 일부로 만들지는 않음. `c` 또는 `d`가 필요하지만 `a, b`에만 필터링하는 쿼리에 대해 index-only scan을 enable. include된 컬럼은 트리 균형에 영향을 주지 않으므로 4-컬럼 인덱스보다 저렴.

세 modifier는 합성 가능 — partial expression covering 인덱스도 완벽히 합법.

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

**이전**: [서브쿼리와 CTE](./08_Subqueries_and_CTE.md) | **다음**: [함수와 프로시저](./10_Functions_and_Procedures.md)

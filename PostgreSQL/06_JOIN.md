# JOIN

## 1. JOIN 개념

JOIN은 두 개 이상의 테이블을 연결하여 데이터를 조회하는 방법입니다.

```
┌─────────────────┐     ┌─────────────────┐
│     users       │     │     orders      │
├─────────────────┤     ├─────────────────┤
│ id │ name       │     │ id │ user_id    │
├────┼────────────┤     ├────┼────────────┤
│ 1  │ 김철수     │◄────│ 1  │ 1          │
│ 2  │ 이영희     │◄────│ 2  │ 1          │
│ 3  │ 박민수     │     │ 3  │ 2          │
└────┴────────────┘     └────┴────────────┘
         ↑ users.id = orders.user_id
```

---

## 2. 실습 테이블 준비

```sql
-- 사용자 테이블
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255)
);

-- 주문 테이블
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    product_name VARCHAR(200),
    amount NUMERIC(10, 2),
    order_date DATE DEFAULT CURRENT_DATE
);

-- 샘플 데이터
INSERT INTO users (name, email) VALUES
('김철수', 'kim@email.com'),
('이영희', 'lee@email.com'),
('박민수', 'park@email.com'),
('최지영', 'choi@email.com');  -- 주문 없는 사용자

INSERT INTO orders (user_id, product_name, amount) VALUES
(1, '노트북', 1500000),
(1, '마우스', 50000),
(2, '키보드', 100000),
(2, '모니터', 300000),
(3, '헤드셋', 150000),
(NULL, '선물세트', 80000);  -- 회원 아닌 주문
```

---

## 3. INNER JOIN

양쪽 테이블 모두에 일치하는 데이터만 반환합니다.

```sql
-- 기본 문법
SELECT columns
FROM table1
INNER JOIN table2 ON table1.column = table2.column;

-- 사용자와 주문 정보 조회
SELECT
    users.name,
    users.email,
    orders.product_name,
    orders.amount
FROM users
INNER JOIN orders ON users.id = orders.user_id;
```

결과:
```
  name  │      email       │ product_name │  amount
────────┼──────────────────┼──────────────┼──────────
 김철수 │ kim@email.com    │ 노트북       │ 1500000
 김철수 │ kim@email.com    │ 마우스       │   50000
 이영희 │ lee@email.com    │ 키보드       │  100000
 이영희 │ lee@email.com    │ 모니터       │  300000
 박민수 │ park@email.com   │ 헤드셋       │  150000
```

### 테이블 별칭 사용

```sql
SELECT u.name, o.product_name, o.amount
FROM users u
INNER JOIN orders o ON u.id = o.user_id;
```

### JOIN만 쓰면 INNER JOIN

```sql
-- INNER 생략 가능
SELECT u.name, o.product_name
FROM users u
JOIN orders o ON u.id = o.user_id;
```

---

## 4. LEFT (OUTER) JOIN

왼쪽 테이블의 모든 행 + 오른쪽에서 일치하는 행을 반환합니다.
일치하지 않으면 NULL로 채워집니다.

```sql
SELECT
    u.name,
    o.product_name,
    o.amount
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;
```

결과:
```
  name  │ product_name │  amount
────────┼──────────────┼──────────
 김철수 │ 노트북       │ 1500000
 김철수 │ 마우스       │   50000
 이영희 │ 키보드       │  100000
 이영희 │ 모니터       │  300000
 박민수 │ 헤드셋       │  150000
 최지영 │ NULL         │ NULL      ← 주문 없는 사용자도 포함
```

### 주문 없는 사용자만 찾기

```sql
SELECT u.name, u.email
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE o.id IS NULL;
```

---

## 5. RIGHT (OUTER) JOIN

오른쪽 테이블의 모든 행 + 왼쪽에서 일치하는 행을 반환합니다.

```sql
SELECT
    u.name,
    o.product_name,
    o.amount
FROM users u
RIGHT JOIN orders o ON u.id = o.user_id;
```

결과:
```
  name  │ product_name │  amount
────────┼──────────────┼──────────
 김철수 │ 노트북       │ 1500000
 김철수 │ 마우스       │   50000
 이영희 │ 키보드       │  100000
 이영희 │ 모니터       │  300000
 박민수 │ 헤드셋       │  150000
 NULL   │ 선물세트     │   80000   ← 회원 없는 주문도 포함
```

---

## 6. FULL (OUTER) JOIN

양쪽 테이블의 모든 행을 반환합니다. 일치하지 않으면 NULL로 채워집니다.

```sql
SELECT
    u.name,
    o.product_name,
    o.amount
FROM users u
FULL JOIN orders o ON u.id = o.user_id;
```

결과:
```
  name  │ product_name │  amount
────────┼──────────────┼──────────
 김철수 │ 노트북       │ 1500000
 김철수 │ 마우스       │   50000
 이영희 │ 키보드       │  100000
 이영희 │ 모니터       │  300000
 박민수 │ 헤드셋       │  150000
 최지영 │ NULL         │ NULL      ← 주문 없는 사용자
 NULL   │ 선물세트     │   80000   ← 회원 없는 주문
```

---

## 7. CROSS JOIN

모든 가능한 조합을 반환합니다 (카티션 곱).

```sql
-- 색상과 사이즈 테이블
CREATE TABLE colors (name VARCHAR(20));
CREATE TABLE sizes (name VARCHAR(10));

INSERT INTO colors VALUES ('빨강'), ('파랑'), ('검정');
INSERT INTO sizes VALUES ('S'), ('M'), ('L');

-- 모든 조합
SELECT c.name AS color, s.name AS size
FROM colors c
CROSS JOIN sizes s;
```

결과:
```
 color │ size
───────┼──────
 빨강  │ S
 빨강  │ M
 빨강  │ L
 파랑  │ S
 파랑  │ M
 파랑  │ L
 검정  │ S
 검정  │ M
 검정  │ L
```

---

## 8. SELF JOIN (자기 조인)

같은 테이블을 자기 자신과 조인합니다.

```sql
-- 직원-관리자 관계
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    manager_id INTEGER REFERENCES employees(id)
);

INSERT INTO employees (name, manager_id) VALUES
('대표이사', NULL),
('부장', 1),
('과장A', 2),
('과장B', 2),
('사원', 3);

-- 직원과 관리자 이름 조회
SELECT
    e.name AS 직원,
    m.name AS 관리자
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;
```

결과:
```
  직원   │ 관리자
─────────┼─────────
 대표이사 │ NULL
 부장     │ 대표이사
 과장A    │ 부장
 과장B    │ 부장
 사원     │ 과장A
```

---

## 9. 다중 테이블 JOIN

3개 이상의 테이블을 연결합니다.

```sql
-- 카테고리 테이블 추가
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100)
);

-- 상품 테이블
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    category_id INTEGER REFERENCES categories(id),
    name VARCHAR(200),
    price NUMERIC(10, 2)
);

-- 주문 상세 테이블
CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER
);

-- 3개 테이블 JOIN
SELECT
    u.name AS user_name,
    p.name AS product_name,
    c.name AS category_name,
    oi.quantity
FROM users u
JOIN orders o ON u.id = o.user_id
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id
JOIN categories c ON p.category_id = c.id;
```

---

## 10. JOIN 조건과 WHERE

### ON vs WHERE

```sql
-- ON: 테이블 연결 조건
-- WHERE: 결과 필터링

-- LEFT JOIN + WHERE
SELECT u.name, o.product_name, o.amount
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE o.amount > 100000;  -- NULL 행 제거됨

-- LEFT JOIN + ON에 추가 조건
SELECT u.name, o.product_name, o.amount
FROM users u
LEFT JOIN orders o ON u.id = o.user_id AND o.amount > 100000;
-- 모든 사용자 유지, 조건 맞는 주문만 연결
```

### 복합 JOIN 조건

```sql
SELECT *
FROM table1 t1
JOIN table2 t2 ON t1.col1 = t2.col1 AND t1.col2 = t2.col2;
```

---

## 11. USING 절

동일한 컬럼명으로 조인할 때 간단하게 표현합니다.

```sql
-- ON 사용
SELECT * FROM orders o
JOIN users u ON o.user_id = u.id;

-- USING 사용 (컬럼명이 같을 때)
-- orders.user_id와 users.user_id가 같다면:
SELECT * FROM orders
JOIN users USING (user_id);
```

---

## 12. NATURAL JOIN

동일한 이름의 모든 컬럼으로 자동 조인합니다. (권장하지 않음)

```sql
-- 같은 이름의 모든 컬럼으로 조인
SELECT * FROM orders
NATURAL JOIN users;

-- 의도치 않은 결과가 나올 수 있어 명시적 ON 권장
```

---

## 13. JOIN 시각화

```
INNER JOIN:         LEFT JOIN:          RIGHT JOIN:         FULL JOIN:
    ┌───┐              ┌───┐              ┌───┐              ┌───┐
   ┌┼───┼┐            ┌┼───┼┐            ┌┼───┼┐            ┌┼───┼┐
  ┌┼│███│┼┐          ┌┼│███│ │          │ │███│┼┐          ┌┼│███│┼┐
  │ │███│ │          ││████│ │          │ │████││          ││█████││
  └┼│███│┼┘          └┼│███│ │          │ │███│┼┘          └┼│███│┼┘
   └┼───┼┘            └┼───┘ │          │ └───┼┘            └─────┼┘
    └───┘              └─────┘          └─────┘              └─────┘
   A ∩ B               A 전체            B 전체            A ∪ B
```

---

## 14. 실습 예제

### 실습 1: 기본 JOIN

```sql
-- 1. 주문한 적 있는 사용자와 주문 정보
SELECT u.name, o.product_name, o.amount, o.order_date
FROM users u
INNER JOIN orders o ON u.id = o.user_id
ORDER BY o.order_date DESC;

-- 2. 각 사용자별 총 주문 금액
SELECT u.name, SUM(o.amount) AS total_amount
FROM users u
INNER JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name
ORDER BY total_amount DESC;
```

### 실습 2: OUTER JOIN

```sql
-- 1. 모든 사용자 (주문 여부 관계없이)
SELECT
    u.name,
    COALESCE(SUM(o.amount), 0) AS total_amount,
    COUNT(o.id) AS order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name
ORDER BY total_amount DESC;

-- 2. 주문하지 않은 사용자 찾기
SELECT u.name, u.email
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE o.id IS NULL;

-- 3. 회원이 아닌 주문 찾기
SELECT o.id, o.product_name, o.amount
FROM users u
RIGHT JOIN orders o ON u.id = o.user_id
WHERE u.id IS NULL;
```

### 실습 3: 복합 조건 JOIN

```sql
-- 1. 100만원 이상 주문한 사용자
SELECT DISTINCT u.name, u.email
FROM users u
INNER JOIN orders o ON u.id = o.user_id
WHERE o.amount >= 1000000;

-- 2. 최근 30일 이내 주문한 사용자
SELECT DISTINCT u.name
FROM users u
INNER JOIN orders o ON u.id = o.user_id
WHERE o.order_date >= CURRENT_DATE - INTERVAL '30 days';
```

### 실습 4: 여러 테이블 JOIN

```sql
-- 카테고리 → 상품 → 주문 연결
SELECT
    c.name AS category,
    p.name AS product,
    u.name AS customer,
    oi.quantity,
    p.price * oi.quantity AS subtotal
FROM categories c
JOIN products p ON c.id = p.category_id
JOIN order_items oi ON p.id = oi.product_id
JOIN orders o ON oi.order_id = o.id
JOIN users u ON o.user_id = u.id
ORDER BY c.name, p.name;
```

---

## 15. 성능 고려사항

### 인덱스 활용

```sql
-- 외래키 컬럼에 인덱스 생성
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);
```

### 필요한 컬럼만 SELECT

```sql
-- 나쁜 예
SELECT * FROM users u JOIN orders o ON u.id = o.user_id;

-- 좋은 예
SELECT u.name, o.product_name, o.amount
FROM users u JOIN orders o ON u.id = o.user_id;
```

### EXPLAIN으로 실행 계획 확인

```sql
EXPLAIN SELECT u.name, o.product_name
FROM users u
JOIN orders o ON u.id = o.user_id;
```

---

## 다음 단계

[07_Aggregation_and_Grouping.md](./07_Aggregation_and_Grouping.md)에서 집계 함수와 GROUP BY를 배워봅시다!

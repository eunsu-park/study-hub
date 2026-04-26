# 서브쿼리와 CTE

**이전**: [집계와 그룹](./07_Aggregation_and_Grouping.md) | **다음**: [뷰와 인덱스](./09_Views_and_Indexes.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 서브쿼리(subquery)가 무엇인지 설명하고 어디에 사용될 수 있는지(WHERE, FROM, SELECT) 파악할 수 있습니다
2. 스칼라(scalar), 다중 행(multi-row), 상관(correlated) 서브쿼리를 작성할 수 있습니다
3. EXISTS / NOT EXISTS를 사용하고 IN / NOT IN과의 동작 차이를 비교할 수 있습니다
4. FROM 절 서브쿼리(인라인 뷰)를 적용하여 중간 결과 집합을 구성할 수 있습니다
5. 복잡한 서브쿼리를 WITH 절을 사용한 공통 테이블 표현식(Common Table Expression, CTE)으로 재작성하여 가독성을 높일 수 있습니다
6. 단일 쿼리에서 여러 CTE를 연결(chain)하여 사용할 수 있습니다
7. 재귀 CTE(WITH RECURSIVE)를 구현하여 계층적 데이터를 순회(traversal)할 수 있습니다
8. 가독성과 성능 필요에 따라 서브쿼리, CTE, JOIN 중 적절한 방법을 선택할 수 있습니다

---

SQL 질문이 복잡해질수록, 다음 질문을 하기 전에 먼저 하나의 질문에 대한 답을 구해야 하는 경우가 자주 생깁니다. 서브쿼리는 한 쿼리 안에 다른 쿼리를 내장할 수 있게 해주고, 공통 테이블 표현식(CTE)은 중간 결과에 이름을 붙여 재사용할 수 있게 해줍니다. 이 두 가지를 함께 사용하면 단일 SQL 구문을 구조화된 다단계 추론 과정으로 변환할 수 있습니다.

문법으로 들어가기 전, [**이론과 원리**](#이론과-원리) 절을 먼저 읽으세요 — correlated와 non-correlated subquery의 차이, CTE가 한때 optimization fence였다가 PostgreSQL 12에서 무엇이 바뀌었는지, 그리고 recursive CTE가 그래프 순회를 어떻게 구현하는지를 다룹니다.

---

## 이론과 원리

서브쿼리는 단지 괄호 안에 쓴 또 다른 `SELECT`이지만, planner는 매우 다른 네 가지 것 — projection 안의 scalar subquery, `WHERE` 안의 subquery(correlated와 non-correlated), `FROM` 안의 subquery(derived table), `WITH` 안의 CTE — 를 완전히 다른 규칙으로 처리합니다. 같은 데이터와 같은 답이 다섯 가지 방식으로 표현될 수 있고, 그 다섯 가지의 runtime은 극단적으로 다를 수 있습니다. 어떤 형태가 "그저 sugar"(planner가 동등한 형태로 다시 쓰는)이고 어떤 형태가 optimization fence로 작동하는지 이해하는 것이, "읽기 좋아서 CTE를 썼다"와 "CTE를 썼더니 우연히 100배 느려졌다"를 가르는 차이입니다.

이 절에서 다루는 내용:

- **(A)** correlated vs non-correlated subquery — 하나는 join이 되고 다른 하나는 상수가 되는 이유.
- **(B)** `FROM` 안의 subquery — derived table, planning, inlining.
- **(C)** CTE(`WITH`) — PG12 이전의 optimization fence 규칙과 새로운 `MATERIALIZED` / `NOT MATERIALIZED` 키워드.
- **(D)** Recursive CTE(`WITH RECURSIVE`) — 반복 알고리즘과 그래프 walk 의미론.

### A. Correlated vs Non-Correlated Subquery

#### A.1 Non-correlated — 1번만 실행

서브쿼리가 enclosing 쿼리의 어떤 컬럼도 참조하지 않으면 **non-correlated**입니다.

```sql
SELECT * FROM orders
WHERE customer_id IN (SELECT id FROM customers WHERE country = 'KR');
```

내부 `SELECT id FROM customers WHERE country = 'KR'`은 필터링되는 행에 의존하지 않습니다. planner는 이를 상수 집합으로 처리 — 1번 실행해서 결과를 materialize한 뒤, 각 `orders` 행에 대해 membership 검사. `customers(country)`에 인덱스가 있으면 inner가 빠르게 실행되고, `orders(customer_id)`에 또 인덱스가 있으면 outer membership 검사는 indexed semi-join이 됩니다.

#### A.2 Correlated — outer 행마다 (잠재적으로) 실행

서브쿼리가 enclosing 쿼리의 컬럼을 참조하면 **correlated**입니다.

```sql
SELECT o.* FROM orders o
WHERE o.amount > (SELECT AVG(amount) FROM orders WHERE customer_id = o.customer_id);
--                                                                   ^^^^^^^^^^^^^
```

순진한 실행 plan은 — `orders`의 각 `o`에 대해 inner aggregate 실행. O(N²)이고 큰 `orders`에선 재앙.

planner는 *거의 항상* correlated subquery를 다시 씁니다 — 위 예를 `customer_id` group-by가 있는 join으로 변환 — 그러나 `LATERAL` 의미론, 이상한 타입, volatile 함수 등으로 인해 다시 쓰기가 막힐 때가 있습니다. 막히면 per-row 실행으로 돌아갑니다. correlation이 join으로 평탄화되었는지 항상 `EXPLAIN`으로 확인.

#### A.3 Scalar subquery

projection 안의 subquery(`SELECT (subquery), ...`)는 최대 1행 1컬럼을 반환해야 합니다. PostgreSQL은 correlated일 때 outer 행마다 1번 실행, 아닐 때 총 1번 실행. scalar subquery는 "각 Y에 대해 X를 lookup"에 편하지만, per-row 비용을 주의 — `LEFT JOIN`이 보통 더 빠릅니다.

### B. FROM 안의 Subquery — Derived Table

```sql
SELECT t.region, t.total
FROM (SELECT region, SUM(sales) AS total FROM orders GROUP BY region) t
WHERE t.total > 100000;
```

`FROM` 안의 subquery를 **derived table** 또는 **inline view**라고 부릅니다. planner는 이를 outer 쿼리에 **inline**할 자유가 있습니다 — 중간 결과를 materialize할 필요 없이, 전체를 단일 plan으로 다시 쓰며, `total > 100000` 술어를 가능한 한 깊이(보통 inner aggregate의 HAVING으로) 밀어 넣습니다.

이 inlining이 derived table을 성능 좋게 만드는 것 — outer `WHERE`이 실행 중이 아니라 planning 중에 inner 쿼리에 도달합니다.

### C. CTE(`WITH`) — Optimization Fence, 그때와 지금

```sql
WITH big_orders AS (
    SELECT * FROM orders WHERE amount > 1000
)
SELECT * FROM big_orders WHERE region = 'KR';
```

derived-table 버전과 동일하게 보입니다. *예전에는* 매우 다르게 동작했습니다.

#### C.1 PG12 이전의 optimization fence

PostgreSQL 12 이전, 모든 CTE는 **optimization fence**였습니다 — planner가 CTE를 먼저 materialize하고, 그 다음 outer 쿼리를 materialize된 결과에 대해 실행. outer `WHERE region = 'KR'`은 CTE *안으로 push down되지 않았습니다*. 그래서 위 예는 `amount > 1000`인 모든 `orders` 행을 스캔하고, 모두 materialize한 뒤 `region = 'KR'`로 필터. 등가 derived-table 형태보다 100× 느릴 때도 있었습니다.

fence가 의도적일 때도 있었지만 — 실행 순서를 통제할 수 있게 해 줌 — 더 흔히는 성능 함정이었습니다.

#### C.2 PG12+ — `MATERIALIZED` 키워드

PostgreSQL 12에서 기본값이 바뀌었습니다. CTE는 이제 derived table처럼 inline되는 것이 *기본*입니다. *단*, CTE가 한 번 이상 참조되거나 `RECURSIVE` 또는 volatile 호출을 포함하면 예외. 두 새 키워드가 명시적 통제 제공:

- `WITH big_orders AS MATERIALIZED (...)` — 옛 fence 동작을 강제. CTE가 부수 효과(예 — CTE 안의 `INSERT ... RETURNING`)를 가지거나 planner가 잘못된 선택을 한다고 알 때 정확히 1번 실행시키고 싶다면 유용.
- `WITH big_orders AS NOT MATERIALIZED (...)` — CTE가 여러 번 참조되어도 inlining을 명시적으로 요청.

#### C.3 Writable CTE

CTE는 `RETURNING`이 있는 `INSERT`, `UPDATE`, `DELETE`를 포함할 수 있습니다. 이로써 "A 한 뒤 그 결과를 B에 사용"을 한 statement로 할 수 있습니다.

```sql
WITH deleted AS (
    DELETE FROM orders WHERE created_at < '2025-01-01' RETURNING *
)
INSERT INTO orders_archive SELECT * FROM deleted;
```

Writable CTE는 항상 materialize됩니다 — 부수 효과가 있고 inline될 수 없습니다.

### D. Recursive CTE — SQL 안의 반복

```sql
WITH RECURSIVE descendants AS (
    -- Base case
    SELECT id, parent_id, name FROM employees WHERE id = 5
    UNION ALL
    -- Recursive case — 이전 결과를 employees와 join
    SELECT e.id, e.parent_id, e.name
    FROM employees e
    JOIN descendants d ON e.parent_id = d.id
)
SELECT * FROM descendants;
```

#### D.1 실행 알고리즘

`WITH RECURSIVE`는 실제로는 재귀가 아니라 반복입니다.

```
working_set = base_case 실행      # non-recursive term의 행
result      = working_set
while working_set이 비어 있지 않으면:
    working_set = recursive_term 실행, working_set을 CTE 참조로 사용
    result.append(working_set)
return result
```

각 iteration은 *직전 iteration의 출력*을 CTE 입력으로 받아 recursive term을 실행하고, 새 행을 append. iteration이 0행을 만들면 루프 종료. (cycle 검출이 필요하면 `UNION ALL` 대신 `UNION` 사용 — cycle로 인한 중복이 dedup됨.)

#### D.2 표현 가능한 것들

- **트리 순회** — 조직도, 카테고리 트리, 댓글 스레드의 descendant/ancestor.
- **그래프 walk** — 최단 경로(cycle 회피 주의), 한 노드로부터의 reachability.
- **숫자 생성** — `WITH RECURSIVE n AS (SELECT 1 UNION ALL SELECT n+1 FROM n WHERE n < 100) SELECT * FROM n;`

#### D.3 비용 모양

비용은 recursive term의 비용 × 재귀 깊이. 깊은 트리나 넓은 그래프에서는 비쌀 수 있고 — 대부분의 쿼리와 달리 planner는 실행 없이는 비용을 추정할 수 없습니다(iteration 수에 통계가 없음). 사용자 제공 그래프를 순회할 때는 항상 `WHERE level < N` 술어로 재귀 깊이를 제한.

### 이론에서 아래 SQL로

이어지는 각 절은 위 메커니즘이 구체화된 형태입니다:

- **`WHERE` 안의 subquery (`IN`, `EXISTS`, `=`)** — non-correlated는 1번 실행, correlated는 가능할 때 join으로 다시 쓰임 (§A).
- **`SELECT` 안의 scalar subquery** — 1행 1컬럼, correlated일 때 outer 행마다 실행 (§A.3).
- **`FROM` 안의 subquery** — derived table, planner가 inline하고 술어 push (§B).
- **`WITH name AS (...)`** — PG12+에선 기본 inline, fence 동작은 `MATERIALIZED`로 (§C).
- **`WITH RECURSIVE`** — base case + recursive case + 종료, 반복적으로 실행 (§D).
- **`LATERAL`** — derived table의 명시적 per-row 실행, correlated subquery를 보완.

---

## 1. 서브쿼리란?

서브쿼리(Subquery)는 쿼리 안에 포함된 또 다른 쿼리입니다.

```sql
SELECT * FROM users
WHERE id IN (SELECT user_id FROM orders);  -- 서브쿼리
          ↑
       괄호 안의 쿼리
```

---

## 2. WHERE 절 서브쿼리

### 스칼라 서브쿼리 (단일 값)

```sql
-- 평균 가격보다 비싼 상품
SELECT * FROM products
WHERE price > (SELECT AVG(price) FROM products);

-- 최신 주문 날짜의 주문들
SELECT * FROM orders
WHERE order_date = (SELECT MAX(order_date) FROM orders);
```

### 다중 행 서브쿼리

```sql
-- 주문한 적 있는 사용자
SELECT * FROM users
WHERE id IN (SELECT DISTINCT user_id FROM orders);

-- 전자기기를 구매한 사용자
SELECT * FROM users
WHERE id IN (
    SELECT o.user_id FROM orders o
    JOIN order_items oi ON o.id = oi.order_id
    JOIN products p ON oi.product_id = p.id
    WHERE p.category = '전자기기'
);
```

### NOT IN

```sql
-- 주문한 적 없는 사용자
SELECT * FROM users
WHERE id NOT IN (
    SELECT user_id FROM orders WHERE user_id IS NOT NULL
);
-- 주의: NOT IN에서 NULL이 있으면 결과가 비어버릴 수 있음
```

### ANY / SOME

```sql
-- 어떤 전자기기보다 비싼 가구
SELECT * FROM products
WHERE category = '가구'
  AND price > ANY (SELECT price FROM products WHERE category = '전자기기');
-- = ANY 는 IN과 동일
```

### ALL

```sql
-- 모든 전자기기보다 비싼 상품
SELECT * FROM products
WHERE price > ALL (SELECT price FROM products WHERE category = '전자기기');
```

---

## 3. EXISTS / NOT EXISTS

행의 존재 여부만 확인합니다.

```sql
-- 주문이 있는 사용자
SELECT * FROM users u
WHERE EXISTS (
    SELECT 1 FROM orders o WHERE o.user_id = u.id
);

-- 주문이 없는 사용자
SELECT * FROM users u
WHERE NOT EXISTS (
    SELECT 1 FROM orders o WHERE o.user_id = u.id
);
```

### IN vs EXISTS

```sql
-- IN: 서브쿼리 결과를 메모리에 로드
SELECT * FROM users
WHERE id IN (SELECT user_id FROM orders);

-- EXISTS: 매 행마다 존재 여부 확인
SELECT * FROM users u
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id);

-- 일반적으로:
-- - 서브쿼리 결과가 작으면 IN
-- - 서브쿼리 결과가 크면 EXISTS
-- - NOT IN 대신 NOT EXISTS 권장 (NULL 문제 방지)
```

---

## 4. FROM 절 서브쿼리 (인라인 뷰)

```sql
-- 카테고리별 평균 가격 계산 후 필터링
SELECT *
FROM (
    SELECT category, AVG(price) AS avg_price
    FROM products
    GROUP BY category
) AS category_avg
WHERE avg_price > 100000;

-- 서브쿼리에 별칭 필수 (AS category_avg)
```

### 복잡한 집계 후 JOIN

```sql
-- 사용자별 주문 통계와 사용자 정보 결합
SELECT
    u.name,
    u.email,
    stats.order_count,
    stats.total_amount
FROM users u
JOIN (
    SELECT
        user_id,
        COUNT(*) AS order_count,
        SUM(amount) AS total_amount
    FROM orders
    GROUP BY user_id
) AS stats ON u.id = stats.user_id;
```

---

## 5. SELECT 절 서브쿼리 (스칼라 서브쿼리)

```sql
-- 각 상품과 함께 카테고리 평균 가격 표시
SELECT
    name,
    price,
    (SELECT AVG(price) FROM products p2 WHERE p2.category = p.category) AS category_avg
FROM products p;

-- 각 사용자의 주문 수
SELECT
    u.name,
    (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id) AS order_count
FROM users u;
```

---

## 6. 상관 서브쿼리

외부 쿼리의 값을 참조하는 서브쿼리입니다.

```sql
-- 자신의 카테고리 평균보다 비싼 상품
SELECT * FROM products p
WHERE price > (
    SELECT AVG(price) FROM products WHERE category = p.category
);
--                                                    ↑ 외부 쿼리 참조

-- 각 카테고리에서 가장 비싼 상품
SELECT * FROM products p
WHERE price = (
    SELECT MAX(price) FROM products WHERE category = p.category
);
```

---

> **비유 -- SQL은 집합으로 사고한다(SQL Thinks in Sets)**: 서브쿼리는 질문 안의 질문과 같습니다: "직원이 10명 이상인 부서에 속한 직원들의 평균 급여는 얼마인가?" 먼저 내부 질문(어떤 부서인가?)에 답한 뒤, 그 답을 외부 질문에 활용합니다. 공통 테이블 표현식(CTE)은 그 내부 답에 이름을 붙여 여러 번 참조할 수 있게 해줍니다 -- 마치 최종 계산에 사용하기 전에 중간 결과를 화이트보드에 적어두는 것과 같습니다.

## 7. CTE (Common Table Expression)

WITH 절을 사용하여 임시 결과 집합에 이름을 붙입니다.

### 기본 CTE

```sql
-- 서브쿼리 방식
SELECT * FROM (
    SELECT category, AVG(price) AS avg_price
    FROM products
    GROUP BY category
) AS category_stats
WHERE avg_price > 100000;

-- CTE 방식 (더 읽기 쉬움)
WITH category_stats AS (
    SELECT category, AVG(price) AS avg_price
    FROM products
    GROUP BY category
)
SELECT * FROM category_stats
WHERE avg_price > 100000;
```

### 여러 CTE 사용

```sql
WITH
-- 카테고리별 통계
category_stats AS (
    SELECT
        category,
        COUNT(*) AS product_count,
        AVG(price) AS avg_price
    FROM products
    GROUP BY category
),
-- 고가 상품 (100만원 이상)
expensive_products AS (
    SELECT * FROM products WHERE price >= 1000000
)
SELECT
    cs.category,
    cs.product_count,
    cs.avg_price,
    COUNT(ep.id) AS expensive_count
FROM category_stats cs
LEFT JOIN expensive_products ep ON cs.category = ep.category
GROUP BY cs.category, cs.product_count, cs.avg_price;
```

### CTE와 메인 쿼리 결합

```sql
WITH monthly_sales AS (
    SELECT
        DATE_TRUNC('month', order_date) AS month,
        SUM(amount) AS total
    FROM orders
    GROUP BY DATE_TRUNC('month', order_date)
)
SELECT
    month,
    total,
    LAG(total) OVER (ORDER BY month) AS prev_month,
    total - LAG(total) OVER (ORDER BY month) AS diff
FROM monthly_sales
ORDER BY month;
```

---

## 8. 재귀 CTE (WITH RECURSIVE)

자기 자신을 참조하는 CTE입니다.

### 조직도 탐색

```sql
-- 직원 테이블
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    manager_id INTEGER REFERENCES employees(id)
);

INSERT INTO employees (name, manager_id) VALUES
('CEO', NULL),
('CTO', 1),
('개발팀장', 2),
('개발자A', 3),
('개발자B', 3),
('CFO', 1),
('재무팀장', 6);

-- CEO부터 모든 부하 직원 조회
WITH RECURSIVE org_tree AS (
    -- 기본 케이스: CEO
    SELECT id, name, manager_id, 1 AS level, name::TEXT AS path
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- 재귀 케이스: 부하 직원들
    SELECT
        e.id,
        e.name,
        e.manager_id,
        ot.level + 1,
        ot.path || ' > ' || e.name
    FROM employees e
    JOIN org_tree ot ON e.manager_id = ot.id
)
SELECT
    REPEAT('  ', level - 1) || name AS org_chart,
    level,
    path
FROM org_tree
ORDER BY path;
```

결과:
```
    org_chart    │ level │           path
─────────────────┼───────┼──────────────────────────
 CEO             │     1 │ CEO
   CFO           │     2 │ CEO > CFO
     재무팀장    │     3 │ CEO > CFO > 재무팀장
   CTO           │     2 │ CEO > CTO
     개발팀장    │     3 │ CEO > CTO > 개발팀장
       개발자A   │     4 │ CEO > CTO > 개발팀장 > 개발자A
       개발자B   │     4 │ CEO > CTO > 개발팀장 > 개발자B
```

### 숫자 시퀀스 생성

```sql
-- 1부터 10까지
WITH RECURSIVE numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < 10
)
SELECT * FROM numbers;
```

### 날짜 범위 생성

```sql
-- 최근 7일
WITH RECURSIVE date_range AS (
    SELECT CURRENT_DATE - INTERVAL '6 days' AS date
    UNION ALL
    SELECT date + INTERVAL '1 day'
    FROM date_range
    WHERE date < CURRENT_DATE
)
SELECT date::DATE FROM date_range;
```

---

## 9. 실습 예제

### 샘플 데이터

```sql
-- 테이블 생성
CREATE TABLE departments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100)
);

CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    department_id INTEGER REFERENCES departments(id),
    salary NUMERIC(10, 2),
    hire_date DATE
);

-- 데이터 삽입
INSERT INTO departments (name) VALUES
('개발'), ('마케팅'), ('인사'), ('재무');

INSERT INTO employees (name, department_id, salary, hire_date) VALUES
('김개발', 1, 5000000, '2020-03-15'),
('이개발', 1, 4500000, '2021-06-20'),
('박마케팅', 2, 4000000, '2019-11-10'),
('최마케팅', 2, 3800000, '2022-01-05'),
('정인사', 3, 3500000, '2020-08-25'),
('한재무', 4, 4200000, '2021-03-10'),
('오재무', 4, 3900000, '2022-07-15');
```

### 실습 1: WHERE 서브쿼리

```sql
-- 1. 전체 평균 급여보다 높은 직원
SELECT name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- 2. 가장 최근 입사한 직원
SELECT * FROM employees
WHERE hire_date = (SELECT MAX(hire_date) FROM employees);

-- 3. 개발 또는 마케팅 부서 직원
SELECT * FROM employees
WHERE department_id IN (
    SELECT id FROM departments WHERE name IN ('개발', '마케팅')
);
```

### 실습 2: 상관 서브쿼리

```sql
-- 1. 자기 부서 평균보다 급여가 높은 직원
SELECT
    e.name,
    e.salary,
    d.name AS department
FROM employees e
JOIN departments d ON e.department_id = d.id
WHERE e.salary > (
    SELECT AVG(salary)
    FROM employees
    WHERE department_id = e.department_id
);

-- 2. 각 부서에서 급여가 가장 높은 직원
SELECT * FROM employees e
WHERE salary = (
    SELECT MAX(salary)
    FROM employees
    WHERE department_id = e.department_id
);
```

### 실습 3: CTE 활용

```sql
-- 1. 부서별 통계와 함께 직원 정보 조회
WITH dept_stats AS (
    SELECT
        department_id,
        AVG(salary) AS avg_salary,
        COUNT(*) AS emp_count
    FROM employees
    GROUP BY department_id
)
SELECT
    e.name,
    e.salary,
    d.name AS department,
    ds.avg_salary AS dept_avg,
    ds.emp_count AS dept_count
FROM employees e
JOIN departments d ON e.department_id = d.id
JOIN dept_stats ds ON e.department_id = ds.department_id;

-- 2. 급여 순위와 함께 조회
WITH ranked_employees AS (
    SELECT
        *,
        RANK() OVER (ORDER BY salary DESC) AS salary_rank,
        RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS dept_rank
    FROM employees
)
SELECT
    name,
    salary,
    salary_rank AS 전체순위,
    dept_rank AS 부서내순위
FROM ranked_employees
ORDER BY salary_rank;
```

### 실습 4: 복합 활용

```sql
-- 각 부서에서 평균 이상 급여를 받는 직원과 그 차이
WITH
dept_avg AS (
    SELECT department_id, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department_id
),
above_avg AS (
    SELECT
        e.*,
        da.avg_salary,
        e.salary - da.avg_salary AS diff
    FROM employees e
    JOIN dept_avg da ON e.department_id = da.department_id
    WHERE e.salary >= da.avg_salary
)
SELECT
    aa.name,
    d.name AS department,
    aa.salary,
    ROUND(aa.avg_salary, 0) AS dept_avg,
    ROUND(aa.diff, 0) AS above_avg_by
FROM above_avg aa
JOIN departments d ON aa.department_id = d.id
ORDER BY aa.diff DESC;
```

---

## 10. 서브쿼리 vs CTE vs JOIN

| 상황 | 권장 |
|------|------|
| 단순 값 비교 | 서브쿼리 |
| 여러 번 참조 | CTE |
| 테이블 연결 | JOIN |
| 복잡한 로직 분리 | CTE |
| 재귀 탐색 | WITH RECURSIVE |

---

**이전**: [집계와 그룹](./07_Aggregation_and_Grouping.md) | **다음**: [뷰와 인덱스](./09_Views_and_Indexes.md)

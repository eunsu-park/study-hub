# 서브쿼리와 CTE

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

## 다음 단계

[09_Views_and_Indexes.md](./09_Views_and_Indexes.md)에서 VIEW와 INDEX를 배워봅시다!

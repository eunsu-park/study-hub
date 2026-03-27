# 집계와 그룹

**이전**: [JOIN](./06_JOIN.md) | **다음**: [서브쿼리와 CTE](./08_Subqueries_and_CTE.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. COUNT, SUM, AVG, MIN, MAX 다섯 가지 핵심 집계 함수(aggregate function)를 적용할 수 있습니다
2. GROUP BY를 사용하여 행을 그룹으로 분할하고 그룹별 통계를 계산할 수 있습니다
3. HAVING으로 그룹 결과를 필터링하고, WHERE와의 차이점을 설명할 수 있습니다
4. GROUP BY와 JOIN을 결합하여 관련 테이블 간 데이터를 집계할 수 있습니다
5. DATE_TRUNC와 EXTRACT를 사용하여 날짜 기반 집계를 수행할 수 있습니다
6. CASE 표현식과 FILTER 절을 사용하여 조건부 집계(conditional aggregation)를 작성할 수 있습니다
7. ROLLUP과 CUBE를 사용하여 소계와 총계를 생성할 수 있습니다
8. SQL 쿼리 실행 순서(FROM, WHERE, GROUP BY, HAVING, SELECT, ORDER BY, LIMIT)를 설명할 수 있습니다

---

데이터베이스는 방대한 양의 데이터를 간결하고 실용적인 수치로 요약하는 데 탁월합니다. "지역별 총 매출은 얼마인가?" 또는 "어떤 상품 카테고리의 평균 매출이 가장 높은가?"와 같은 질문에는 집계 함수와 그룹화가 필요합니다. 이러한 연산을 숙달하면 단순한 거래 테이블을 비즈니스 의사결정을 이끄는 대시보드, 보고서, KPI로 변환할 수 있습니다.

## 1. 집계 함수 (Aggregate Functions)

집계 함수는 여러 행의 값을 하나의 결과로 계산합니다.

| 함수 | 설명 |
|------|------|
| `COUNT()` | 행 개수 |
| `SUM()` | 합계 |
| `AVG()` | 평균 |
| `MIN()` | 최소값 |
| `MAX()` | 최대값 |

---

## 2. 실습 테이블 준비

```sql
CREATE TABLE sales (
    id SERIAL PRIMARY KEY,
    product VARCHAR(100),
    category VARCHAR(50),
    amount NUMERIC(10, 2),
    quantity INTEGER,
    sale_date DATE,
    region VARCHAR(50)
);

INSERT INTO sales (product, category, amount, quantity, sale_date, region) VALUES
('노트북', '전자기기', 1500000, 2, '2024-01-05', '서울'),
('마우스', '전자기기', 50000, 10, '2024-01-05', '서울'),
('키보드', '전자기기', 100000, 5, '2024-01-06', '부산'),
('모니터', '전자기기', 300000, 3, '2024-01-07', '서울'),
('책상', '가구', 250000, 2, '2024-01-08', '대전'),
('의자', '가구', 150000, 4, '2024-01-08', '서울'),
('노트북', '전자기기', 1800000, 1, '2024-01-10', '부산'),
('마우스', '전자기기', 45000, 20, '2024-01-12', '대전'),
('책상', '가구', 280000, 1, '2024-01-15', '서울'),
('의자', '가구', 180000, 3, '2024-01-15', '부산');
```

---

## 3. COUNT - 개수 세기

### 전체 행 수

```sql
SELECT COUNT(*) FROM sales;
-- 10
```

### 특정 컬럼 개수 (NULL 제외)

```sql
SELECT COUNT(region) FROM sales;
-- NULL이 아닌 region 개수
```

### 중복 제거 개수

```sql
SELECT COUNT(DISTINCT category) FROM sales;
-- 2 (전자기기, 가구)

SELECT COUNT(DISTINCT region) FROM sales;
-- 3 (서울, 부산, 대전)
```

---

## 4. SUM - 합계

```sql
-- 집계 함수는 수백만 행을 단일 결과로 압축 — 데이터베이스가 서버 측에서 계산하므로
-- 모든 행을 애플리케이션으로 전송하는 비용을 절감
-- 총 매출액
SELECT SUM(amount) FROM sales;
-- 4653000

-- 총 판매 수량
SELECT SUM(quantity) FROM sales;
-- 51

-- 조건부 합계
SELECT SUM(amount) FROM sales WHERE category = '전자기기';
```

---

## 5. AVG - 평균

```sql
-- 평균 매출액
SELECT AVG(amount) FROM sales;
-- 465300

-- 소수점 처리
SELECT ROUND(AVG(amount), 2) AS avg_amount FROM sales;

-- 조건부 평균
SELECT ROUND(AVG(amount), 2)
FROM sales
WHERE region = '서울';
```

---

## 6. MIN / MAX - 최소/최대

```sql
-- 최소 매출액
SELECT MIN(amount) FROM sales;
-- 45000

-- 최대 매출액
SELECT MAX(amount) FROM sales;
-- 1800000

-- 가장 최근 판매일
SELECT MAX(sale_date) FROM sales;

-- 가장 오래된 판매일
SELECT MIN(sale_date) FROM sales;
```

---

## 7. 여러 집계 함수 함께 사용

```sql
SELECT
    COUNT(*) AS total_count,
    SUM(amount) AS total_sales,
    ROUND(AVG(amount), 2) AS avg_sales,
    MIN(amount) AS min_sales,
    MAX(amount) AS max_sales,
    SUM(quantity) AS total_quantity
FROM sales;
```

---

## 8. GROUP BY - 그룹화

데이터를 특정 컬럼 기준으로 그룹화하여 집계합니다.

### 기본 GROUP BY

```sql
-- GROUP BY는 행을 그룹으로 분할하여 각 집계(COUNT, SUM)가 그룹별로 독립 실행됨
-- 하나의 쿼리로 "카테고리별 합계"를 구하는 방법
SELECT
    category,
    COUNT(*) AS count,
    SUM(amount) AS total_amount
FROM sales
GROUP BY category;
```

결과:
```
 category │ count │ total_amount
──────────┼───────┼──────────────
 전자기기 │     6 │      3795000
 가구     │     4 │       858000
```

### 지역별 매출

```sql
SELECT
    region,
    COUNT(*) AS sales_count,
    SUM(amount) AS total_amount,
    ROUND(AVG(amount), 2) AS avg_amount
FROM sales
GROUP BY region
ORDER BY total_amount DESC;
```

### 상품별 매출

```sql
SELECT
    product,
    SUM(quantity) AS total_qty,
    SUM(amount) AS total_sales
FROM sales
GROUP BY product
ORDER BY total_sales DESC;
```

---

## 9. 다중 컬럼 GROUP BY

```sql
-- 카테고리 + 지역별 매출
SELECT
    category,
    region,
    COUNT(*) AS count,
    SUM(amount) AS total
FROM sales
GROUP BY category, region
ORDER BY category, region;
```

결과:
```
 category │ region │ count │  total
──────────┼────────┼───────┼─────────
 가구     │ 대전   │     1 │  250000
 가구     │ 부산   │     1 │  180000
 가구     │ 서울   │     2 │  430000
 전자기기 │ 대전   │     1 │   45000
 전자기기 │ 부산   │     2 │ 1900000
 전자기기 │ 서울   │     3 │ 1850000
```

---

## 10. HAVING - 그룹 필터링

WHERE는 그룹화 전 개별 행을 필터링하고, HAVING은 집계 후 그룹을 필터링합니다.
HAVING은 SUM, COUNT 같은 집계 함수 조건에 사용합니다 — WHERE는 그룹이 형성되기 전에
실행되므로 집계 함수를 참조할 수 없습니다.

```sql
-- 총 매출 50만원 이상인 카테고리만
SELECT
    category,
    SUM(amount) AS total_amount
FROM sales
GROUP BY category
HAVING SUM(amount) >= 500000;
```

### WHERE + HAVING

```sql
-- WHERE + HAVING 협력: WHERE가 먼저 데이터셋을 축소(더 저렴)한 후
-- HAVING이 집계된 그룹을 필터링 — 성능을 위해 가능한 조건은 WHERE로 이동
SELECT
    product,
    SUM(amount) AS total_amount
FROM sales
WHERE region IN ('서울', '부산')  -- 그룹화 전 필터 (행 수준)
GROUP BY product
HAVING SUM(amount) >= 1000000     -- 그룹화 후 필터 (그룹 수준)
ORDER BY total_amount DESC;
```

### HAVING에서 별칭 사용 (PostgreSQL)

```sql
-- PostgreSQL은 HAVING에서 별칭 사용 가능
SELECT
    product,
    SUM(amount) AS total
FROM sales
GROUP BY product
HAVING SUM(amount) > 500000;  -- 표준 방식

-- 또는 (PostgreSQL 확장)
-- HAVING total > 500000;  -- 일부 버전에서만 동작
```

---

## 11. GROUP BY + JOIN

```sql
-- 준비: 카테고리 테이블
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50),
    description TEXT
);

INSERT INTO categories (name, description) VALUES
('전자기기', '전자 제품'),
('가구', '가구 제품');

-- 카테고리 정보와 함께 집계
SELECT
    c.name AS category,
    c.description,
    COUNT(s.id) AS sales_count,
    SUM(s.amount) AS total_sales
FROM categories c
LEFT JOIN sales s ON c.name = s.category
GROUP BY c.id, c.name, c.description;
```

---

## 12. 날짜별 집계

### 일별 매출

```sql
SELECT
    sale_date,
    COUNT(*) AS count,
    SUM(amount) AS daily_total
FROM sales
GROUP BY sale_date
ORDER BY sale_date;
```

### 월별 매출

```sql
SELECT
    DATE_TRUNC('month', sale_date) AS month,
    COUNT(*) AS count,
    SUM(amount) AS monthly_total
FROM sales
GROUP BY DATE_TRUNC('month', sale_date)
ORDER BY month;
```

### 연도별 매출

```sql
SELECT
    EXTRACT(YEAR FROM sale_date) AS year,
    SUM(amount) AS yearly_total
FROM sales
GROUP BY EXTRACT(YEAR FROM sale_date);
```

---

## 13. 조건부 집계

### CASE + SUM

```sql
SELECT
    SUM(CASE WHEN category = '전자기기' THEN amount ELSE 0 END) AS electronics,
    SUM(CASE WHEN category = '가구' THEN amount ELSE 0 END) AS furniture
FROM sales;
```

### FILTER (PostgreSQL 9.4+)

```sql
-- FILTER는 CASE+SUM보다 가독성이 좋고, "~인 경우만 집계"로 읽히며
-- 플래너가 동등한 CASE 표현식보다 더 잘 최적화할 수 있음
SELECT
    COUNT(*) FILTER (WHERE category = '전자기기') AS electronics_count,
    COUNT(*) FILTER (WHERE category = '가구') AS furniture_count,
    SUM(amount) FILTER (WHERE region = '서울') AS seoul_sales
FROM sales;
```

---

## 14. ROLLUP과 CUBE

### ROLLUP - 소계 추가

```sql
SELECT
    category,
    region,
    SUM(amount) AS total
FROM sales
GROUP BY ROLLUP (category, region)
ORDER BY category NULLS LAST, region NULLS LAST;
```

결과:
```
 category │ region │   total
──────────┼────────┼──────────
 가구     │ 대전   │   250000
 가구     │ 부산   │   180000
 가구     │ 서울   │   430000
 가구     │ NULL   │   860000  ← 가구 소계
 전자기기 │ 대전   │    45000
 전자기기 │ 부산   │  1900000
 전자기기 │ 서울   │  1850000
 전자기기 │ NULL   │  3795000  ← 전자기기 소계
 NULL     │ NULL   │  4655000  ← 총계
```

### CUBE - 모든 조합의 소계

```sql
SELECT
    category,
    region,
    SUM(amount) AS total
FROM sales
GROUP BY CUBE (category, region)
ORDER BY category NULLS LAST, region NULLS LAST;
```

### GROUPING - NULL 구분

```sql
SELECT
    CASE WHEN GROUPING(category) = 1 THEN '전체' ELSE category END AS category,
    CASE WHEN GROUPING(region) = 1 THEN '전체' ELSE region END AS region,
    SUM(amount) AS total
FROM sales
GROUP BY ROLLUP (category, region);
```

---

## 15. 실습 예제

### 실습 1: 기본 집계

```sql
-- 1. 전체 매출 통계
SELECT
    COUNT(*) AS 총_판매건수,
    SUM(amount) AS 총_매출,
    ROUND(AVG(amount), 0) AS 평균_매출,
    MIN(amount) AS 최소_매출,
    MAX(amount) AS 최대_매출
FROM sales;

-- 2. 카테고리별 판매 통계
SELECT
    category AS 카테고리,
    COUNT(*) AS 판매건수,
    SUM(quantity) AS 총_수량,
    SUM(amount) AS 총_매출,
    ROUND(AVG(amount), 0) AS 평균_매출
FROM sales
GROUP BY category
ORDER BY 총_매출 DESC;
```

### 실습 2: 복합 조건

```sql
-- 1. 지역별 매출 (50만원 이상만)
SELECT
    region,
    SUM(amount) AS total
FROM sales
GROUP BY region
HAVING SUM(amount) >= 500000
ORDER BY total DESC;

-- 2. 상품별 판매 수량 랭킹
SELECT
    product,
    SUM(quantity) AS total_qty
FROM sales
GROUP BY product
ORDER BY total_qty DESC
LIMIT 5;
```

### 실습 3: 날짜 집계

```sql
-- 1. 일별 매출 추이
SELECT
    sale_date,
    SUM(amount) AS daily_sales,
    SUM(SUM(amount)) OVER (ORDER BY sale_date) AS cumulative_sales
FROM sales
GROUP BY sale_date
ORDER BY sale_date;

-- 2. 최근 7일 일평균 매출
SELECT
    ROUND(AVG(daily_total), 2) AS avg_daily_sales
FROM (
    SELECT sale_date, SUM(amount) AS daily_total
    FROM sales
    WHERE sale_date >= CURRENT_DATE - INTERVAL '7 days'
    GROUP BY sale_date
) daily;
```

### 실습 4: 크로스탭 (피벗)

```sql
-- 카테고리 × 지역 매출 크로스탭
SELECT
    category,
    SUM(amount) FILTER (WHERE region = '서울') AS 서울,
    SUM(amount) FILTER (WHERE region = '부산') AS 부산,
    SUM(amount) FILTER (WHERE region = '대전') AS 대전,
    SUM(amount) AS 총계
FROM sales
GROUP BY category;
```

결과:
```
 category │  서울   │  부산   │ 대전  │   총계
──────────┼─────────┼─────────┼───────┼──────────
 가구     │  430000 │  180000 │ 250000│   860000
 전자기기 │ 1850000 │ 1900000 │  45000│  3795000
```

---

## 16. 쿼리 실행 순서

```
FROM / JOIN    ← 테이블 지정
    ↓
WHERE          ← 행 필터링
    ↓
GROUP BY       ← 그룹화
    ↓
HAVING         ← 그룹 필터링
    ↓
SELECT         ← 컬럼 선택
    ↓
DISTINCT       ← 중복 제거
    ↓
ORDER BY       ← 정렬
    ↓
LIMIT/OFFSET   ← 결과 제한
```

---

**이전**: [JOIN](./06_JOIN.md) | **다음**: [서브쿼리와 CTE](./08_Subqueries_and_CTE.md)

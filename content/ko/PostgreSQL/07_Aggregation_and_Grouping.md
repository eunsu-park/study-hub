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

---

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

### 이론: 집계 state 머신

집계 함수는 함수 하나가 아니라, **세 함수**와 하나의 **state**입니다.

| 구성 요소 | 역할 |
|----------|------|
| `initcond` | state의 초기값 (예 — `SUM`은 0, `MAX`은 NULL) |
| `sfunc` (state transition) | 입력 행마다 호출 — `state := sfunc(state, new_value)` |
| `finalfunc` (옵션) | 끝에 1번 호출 — `result := finalfunc(state)` |
| `stype` | running state의 데이터 타입 |

`SUM(x)`의 경우:

- `stype` = `numeric`, `initcond` = 0
- `sfunc(s, x)` = `s + x`
- `finalfunc` = identity (생략)

`AVG(x)`의 경우:

- `stype` = `(sum, count)` 쌍, `initcond` = `(0, 0)`
- `sfunc((sum, count), x)` = `(sum + x, count + 1)`
- `finalfunc((sum, count))` = `sum / count`

`array_agg(x)`의 경우:

- `stype` = 내부 accumulator
- `sfunc`은 append, `finalfunc`은 배열 materialize

이 구조가 중요한 이유 — 집계는 병렬 실행과 합성됩니다. PostgreSQL parallel-aggregate는 입력을 worker들에 분산하고, 각 worker가 행별 `sfunc`을 실행하고, coordinator가 worker별 state를 `combinefunc`으로 결합합니다(`SUM`의 combine은 `+`, `AVG`는 element-wise 쌍 합). `combinefunc`을 제공하는 사용자 정의 집계는 병렬화 가능, 아니면 불가능.

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

### 이론: HashAggregate vs GroupAggregate

`GROUP BY region`이 주어졌을 때, PostgreSQL은 그룹별 state를 조직하는 두 가지 방법을 가집니다.

#### B.1 HashAggregate

`region`을 키로 한 in-memory hash table 유지. 입력 행마다 `region`을 hash, bucket에서 state lookup, `sfunc` 실행. 마지막에 bucket당 행 1개 emit.

```
H = {}
input의 각 행 r에 대해:
    state = H.get(r.region, initcond)
    H[r.region] = sfunc(state, r.sales)
H.items()의 region, state에 대해:
    (region, finalfunc(state)) emit
```

- **비용** — O(N) + 그룹 수에 비례하는 hash table 메모리.
- **유리한 경우** — distinct group 수가 hash table을 `work_mem`에 넣을 만큼 작을 때. 순서 보존 안 됨 — 출력은 hash 순서.
- **Spill 동작** — PostgreSQL 13+는 `work_mem`을 초과하면 hash partition을 디스크로 spill. 이전 버전은 그 경우 HashAggregate를 거부.

#### B.2 GroupAggregate

GROUP BY 컬럼으로 입력 정렬, 그 다음 정렬된 stream을 walk. 키가 바뀔 때마다 이전 그룹을 emit하고 state reset.

```
input을 region 기준 정렬
state, prev = initcond, None
sorted_input의 각 행 r에 대해:
    if r.region != prev:
        if prev is not None: (prev, finalfunc(state)) emit
        state, prev = initcond, r.region
    state = sfunc(state, r.sales)
(prev, finalfunc(state)) emit
```

- **비용** — sort에 O(N log N), walk에 O(N). state 메모리는 *상수* — 한 번에 한 그룹의 state만.
- **유리한 경우** — 입력이 이미 정렬됨(예 — `region`에 대한 index scan), 또는 그룹 수가 HashAggregate에 너무 많을 때.
- **Bonus** — 출력이 GROUP BY 컬럼으로 정렬되어 있어, 하류 `ORDER BY`가 매치되면 유용.

planner는 추정 그룹 수, 사용 가능한 인덱스, `work_mem`을 바탕으로 고릅니다. `EXPLAIN`은 어느 것이 선택되었는지 `HashAggregate` 또는 `GroupAggregate`로 표시.

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

### 이론: WHERE는 GROUP BY 전, HAVING은 후

`SELECT` 절 실행의 논리 순서:

```
1. FROM     → join된 행 집결
2. WHERE    → 그룹핑 BEFORE 행 필터
3. GROUP BY → 그룹으로 분할
4. (그룹별로 집계 함수 평가)
5. HAVING   → 집계 AFTER 그룹 필터
6. SELECT   → 컬럼 projection
7. ORDER BY → 정렬
8. LIMIT    → truncate
```

두 필터링 단계는 집계의 양쪽에 살고 있습니다.

```sql
-- 세기 전에 행 필터
SELECT region, COUNT(*) FROM orders
WHERE order_date >= '2026-01-01'   -- per-row 술어
GROUP BY region;

-- 센 후에 그룹 필터
SELECT region, COUNT(*) FROM orders
GROUP BY region
HAVING COUNT(*) > 100;             -- per-group 술어
```

#### C.1 술어를 HAVING에서 WHERE로 옮기는 것이 속도 향상인 이유

술어가 base 컬럼만 참조한다면(집계 아님), `WHERE`에 두세요. WHERE는 hash나 sort *전에* 입력을 줄이므로, 이후 모든 단계의 작업이 줄어듭니다. HAVING은 전체 집계 pass *이후*에 동작하므로, 어차피 버려질 행에 대해 풀 비용을 치른 뒤입니다.

```sql
-- 느림 — non-aggregate에 HAVING
SELECT region, SUM(sales) FROM orders
GROUP BY region
HAVING region <> 'EU';            -- ← WHERE여야 함

-- 빠름
SELECT region, SUM(sales) FROM orders
WHERE region <> 'EU'
GROUP BY region;
```

planner가 가끔 잡아내서 다시 쓰지만, 의존하지는 마세요.

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

### 이론: ROLLUP, CUBE, GROUPING SETS — 다차원 집계

이들은 OLAP에서 온 것이며, 한 쿼리에서 여러 `GROUP BY` granularity를 계산하게 해 줍니다.

#### D.1 ROLLUP — 계층적 합계

`GROUP BY ROLLUP (a, b, c)`는 다음을 생성:

- `(a, b, c)`별 그룹
- `(a, b)`별 소계 (모든 `c`에 걸쳐 합한 `(a, b)`당 행 1개)
- `(a)`별 소계 (모든 `(b, c)`에 걸쳐 합)
- 전체 합계 (`a, b, c` 모두 rolled up)

계층 보고서에 유용 — region → country → city, country와 region 소계가 같은 결과 집합에 보임.

#### D.2 CUBE — 모든 차원 조합

`GROUP BY CUBE (a, b, c)`는 2³ = 8 그룹핑 레벨 — `{a, b, c}`의 모든 부분집합 — 을 생성. 가능한 모든 조합으로 slice하고 싶은 cross-tabulation에 유용.

#### D.3 GROUPING SETS — 어떤 조합인지를 정확히 지정

`GROUP BY GROUPING SETS ((a, b), (a, c), ())`은 나열된 조합만 생성. ROLLUP과 CUBE는 GROUPING SETS의 syntactic sugar.

#### D.4 `GROUPING()` indicator 함수

ROLLUP/CUBE/GROUPING SETS 쿼리에서 NULL은 두 가지 다른 의미로 등장합니다 — 데이터의 *실제* NULL 값, 또는 "이 컬럼의 모든 값에 걸쳐 rolled up"의 *placeholder*. `GROUPING(col)` 함수가 이를 구분 — `col`이 rollup 때문에 NULL이면 1, 아니면 0 반환. 여러 컬럼을 bitmap으로 결합 — `GROUPING(a, b)`는 2-bit 정수 반환.

#### D.5 구현

PostgreSQL은 GROUPING SETS를 grouping set당 집계 알고리즘을 1번 실행해서 구현(또는 가능할 때 겹치는 set 사이에 sort/hash 작업을 공유). 비용은 대략 GROUP BY 1개의 비용 × grouping set 수 — 5개 컬럼에 대한 CUBE는 32× 비용.

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

# 17. 윈도우 함수와 분석 쿼리 (Window Functions & Analytics)

## 학습 목표
- 윈도우 함수의 개념과 일반 집계 함수와의 차이 이해
- OVER 절과 파티션, 프레임 개념 마스터
- 순위 함수 (ROW_NUMBER, RANK, DENSE_RANK) 활용
- 분석 함수 (LEAD, LAG, FIRST_VALUE) 활용
- 실무 분석 쿼리 작성 능력 향상

## 목차
1. [윈도우 함수 기초](#1-윈도우-함수-기초)
2. [순위 함수](#2-순위-함수)
3. [분석 함수](#3-분석-함수)
4. [집계 윈도우 함수](#4-집계-윈도우-함수)
5. [프레임 상세](#5-프레임-상세)
6. [실전 활용 패턴](#6-실전-활용-패턴)
7. [연습 문제](#7-연습-문제)

---

## 1. 윈도우 함수 기초

### 1.1 윈도우 함수란?

```
┌─────────────────────────────────────────────────────────────────┐
│                 일반 집계 vs 윈도우 함수                          │
│                                                                 │
│   일반 집계 (GROUP BY)         윈도우 함수 (OVER)               │
│   ┌───────────────┐            ┌───────────────┐               │
│   │ A | B | SUM   │            │ A | B | val | SUM             │
│   ├───────────────┤            ├───────────────────┤           │
│   │ X |   | 150   │            │ X | 1 | 50  | 150 │           │
│   │ Y |   | 120   │            │ X | 2 | 100 | 150 │           │
│   └───────────────┘            │ Y | 3 | 70  | 120 │           │
│   (행이 그룹으로 축소)         │ Y | 4 | 50  | 120 │           │
│                                └───────────────────┘           │
│                                (모든 행 유지 + 집계값)           │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 기본 문법

```sql
-- 윈도우 함수 기본 구조
함수명() OVER (
    [PARTITION BY 컬럼]    -- 그룹 나누기 (선택)
    [ORDER BY 컬럼]        -- 정렬 (선택)
    [프레임 절]            -- 범위 지정 (선택)
)

-- 예시
SELECT
    department,
    employee_name,
    salary,
    SUM(salary) OVER (PARTITION BY department) AS dept_total,
    AVG(salary) OVER (PARTITION BY department) AS dept_avg
FROM employees;
```

### 1.3 샘플 데이터

```sql
-- 테스트 테이블 생성
CREATE TABLE sales (
    id SERIAL PRIMARY KEY,
    salesperson VARCHAR(50),
    region VARCHAR(20),
    sale_date DATE,
    amount NUMERIC(10,2)
);

INSERT INTO sales (salesperson, region, sale_date, amount) VALUES
    ('Alice', 'East', '2024-01-15', 1000),
    ('Alice', 'East', '2024-01-20', 1500),
    ('Alice', 'East', '2024-02-10', 2000),
    ('Bob', 'East', '2024-01-18', 800),
    ('Bob', 'East', '2024-02-15', 1200),
    ('Charlie', 'West', '2024-01-10', 900),
    ('Charlie', 'West', '2024-01-25', 1100),
    ('Charlie', 'West', '2024-02-20', 1300),
    ('Diana', 'West', '2024-01-30', 700),
    ('Diana', 'West', '2024-02-05', 1600);
```

---

## 2. 순위 함수

### 2.1 ROW_NUMBER, RANK, DENSE_RANK 비교

```sql
-- 순위 함수 비교
SELECT
    salesperson,
    amount,
    ROW_NUMBER() OVER (ORDER BY amount DESC) AS row_num,
    RANK() OVER (ORDER BY amount DESC) AS rank,
    DENSE_RANK() OVER (ORDER BY amount DESC) AS dense_rank
FROM sales;
```

```
결과 예시 (동점 처리 차이):
┌────────────┬────────┬─────────┬──────┬────────────┐
│ salesperson│ amount │ row_num │ rank │ dense_rank │
├────────────┼────────┼─────────┼──────┼────────────┤
│ Alice      │ 2000   │ 1       │ 1    │ 1          │
│ Diana      │ 1600   │ 2       │ 2    │ 2          │
│ Alice      │ 1500   │ 3       │ 3    │ 3          │
│ Charlie    │ 1300   │ 4       │ 4    │ 4          │
│ Bob        │ 1200   │ 5       │ 5    │ 5          │
│ Charlie    │ 1100   │ 6       │ 6    │ 6          │
│ Alice      │ 1000   │ 7       │ 7    │ 7          │  -- 동점 없음
│ Charlie    │  900   │ 8       │ 8    │ 8          │
│ Bob        │  800   │ 9       │ 9    │ 9          │
│ Diana      │  700   │ 10      │ 10   │ 10         │
└────────────┴────────┴─────────┴──────┴────────────┘

동점이 있는 경우:
│ A          │ 1000   │ 1       │ 1    │ 1          │
│ B          │ 1000   │ 2       │ 1    │ 1          │  -- 동점!
│ C          │  900   │ 3       │ 3    │ 2          │
                      (연속)    (건너뜀) (연속)
```

### 2.2 PARTITION BY와 함께 순위

```sql
-- 지역별 판매 순위
SELECT
    region,
    salesperson,
    amount,
    RANK() OVER (
        PARTITION BY region
        ORDER BY amount DESC
    ) AS region_rank
FROM sales;
```

```
결과:
┌────────┬────────────┬────────┬─────────────┐
│ region │ salesperson│ amount │ region_rank │
├────────┼────────────┼────────┼─────────────┤
│ East   │ Alice      │ 2000   │ 1           │
│ East   │ Alice      │ 1500   │ 2           │
│ East   │ Bob        │ 1200   │ 3           │
│ East   │ Alice      │ 1000   │ 4           │
│ East   │ Bob        │  800   │ 5           │
│ West   │ Diana      │ 1600   │ 1           │  ← 파티션 리셋
│ West   │ Charlie    │ 1300   │ 2           │
│ West   │ Charlie    │ 1100   │ 3           │
│ West   │ Charlie    │  900   │ 4           │
│ West   │ Diana      │  700   │ 5           │
└────────┴────────────┴────────┴─────────────┘
```

### 2.3 NTILE - 분위수 할당

```sql
-- 4분위로 나누기
SELECT
    salesperson,
    amount,
    NTILE(4) OVER (ORDER BY amount DESC) AS quartile
FROM sales;

-- 사용 예: 상위 25% 고객 식별
SELECT *
FROM (
    SELECT
        customer_id,
        total_purchase,
        NTILE(4) OVER (ORDER BY total_purchase DESC) AS quartile
    FROM customer_summary
) sub
WHERE quartile = 1;  -- 상위 25%
```

### 2.4 Top-N 쿼리

```sql
-- 각 지역별 상위 2개 판매 추출
SELECT *
FROM (
    SELECT
        region,
        salesperson,
        sale_date,
        amount,
        ROW_NUMBER() OVER (
            PARTITION BY region
            ORDER BY amount DESC
        ) AS rn
    FROM sales
) ranked
WHERE rn <= 2;
```

---

## 3. 분석 함수

### 3.1 LAG와 LEAD

```sql
-- LAG: 이전 행 값 참조
-- LEAD: 다음 행 값 참조
SELECT
    salesperson,
    sale_date,
    amount,
    LAG(amount) OVER (
        PARTITION BY salesperson
        ORDER BY sale_date
    ) AS prev_amount,
    LEAD(amount) OVER (
        PARTITION BY salesperson
        ORDER BY sale_date
    ) AS next_amount
FROM sales
ORDER BY salesperson, sale_date;
```

```
결과:
┌────────────┬────────────┬────────┬─────────────┬─────────────┐
│ salesperson│ sale_date  │ amount │ prev_amount │ next_amount │
├────────────┼────────────┼────────┼─────────────┼─────────────┤
│ Alice      │ 2024-01-15 │ 1000   │ NULL        │ 1500        │
│ Alice      │ 2024-01-20 │ 1500   │ 1000        │ 2000        │
│ Alice      │ 2024-02-10 │ 2000   │ 1500        │ NULL        │
│ Bob        │ 2024-01-18 │  800   │ NULL        │ 1200        │
│ Bob        │ 2024-02-15 │ 1200   │  800        │ NULL        │
└────────────┴────────────┴────────┴─────────────┴─────────────┘
```

### 3.2 증감률 계산

```sql
-- 전월 대비 증감률
SELECT
    salesperson,
    sale_date,
    amount,
    LAG(amount) OVER (
        PARTITION BY salesperson
        ORDER BY sale_date
    ) AS prev_amount,
    ROUND(
        (amount - LAG(amount) OVER (
            PARTITION BY salesperson ORDER BY sale_date
        )) * 100.0 /
        NULLIF(LAG(amount) OVER (
            PARTITION BY salesperson ORDER BY sale_date
        ), 0),
        2
    ) AS growth_pct
FROM sales
ORDER BY salesperson, sale_date;
```

### 3.3 FIRST_VALUE, LAST_VALUE, NTH_VALUE

```sql
-- 파티션 내 첫 번째/마지막 값
SELECT
    salesperson,
    sale_date,
    amount,
    FIRST_VALUE(amount) OVER (
        PARTITION BY salesperson
        ORDER BY sale_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS first_sale,
    LAST_VALUE(amount) OVER (
        PARTITION BY salesperson
        ORDER BY sale_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS last_sale,
    NTH_VALUE(amount, 2) OVER (
        PARTITION BY salesperson
        ORDER BY sale_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS second_sale
FROM sales;
```

---

## 4. 집계 윈도우 함수

### 4.1 SUM, AVG, COUNT

```sql
-- 윈도우 집계 함수
SELECT
    salesperson,
    sale_date,
    amount,
    -- 영업사원별 총계
    SUM(amount) OVER (PARTITION BY salesperson) AS person_total,
    -- 영업사원별 평균
    AVG(amount) OVER (PARTITION BY salesperson) AS person_avg,
    -- 전체 대비 비율
    ROUND(amount * 100.0 / SUM(amount) OVER (), 2) AS pct_of_total
FROM sales
ORDER BY salesperson, sale_date;
```

### 4.2 누적 합계 (Running Total)

```sql
-- 누적 합계
SELECT
    sale_date,
    amount,
    SUM(amount) OVER (
        ORDER BY sale_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total
FROM sales
ORDER BY sale_date;

-- 영업사원별 누적 합계
SELECT
    salesperson,
    sale_date,
    amount,
    SUM(amount) OVER (
        PARTITION BY salesperson
        ORDER BY sale_date
    ) AS cumulative_sales
FROM sales
ORDER BY salesperson, sale_date;
```

### 4.3 이동 평균 (Moving Average)

```sql
-- 최근 3건의 이동 평균
SELECT
    sale_date,
    amount,
    AVG(amount) OVER (
        ORDER BY sale_date
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS moving_avg_3
FROM sales
ORDER BY sale_date;

-- 전후 1건 포함 이동 평균 (중심 이동 평균)
SELECT
    sale_date,
    amount,
    AVG(amount) OVER (
        ORDER BY sale_date
        ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
    ) AS centered_avg
FROM sales
ORDER BY sale_date;
```

---

## 5. 프레임 상세

### 5.1 프레임 구문

```
ROWS | RANGE | GROUPS BETWEEN 시작점 AND 끝점

시작점/끝점:
- UNBOUNDED PRECEDING  -- 파티션 처음
- n PRECEDING          -- n개 이전
- CURRENT ROW          -- 현재 행
- n FOLLOWING          -- n개 이후
- UNBOUNDED FOLLOWING  -- 파티션 끝
```

### 5.2 ROWS vs RANGE

```sql
-- 동점 데이터로 차이 확인
CREATE TABLE test_frame (
    id INT,
    val INT
);
INSERT INTO test_frame VALUES (1, 100), (2, 100), (3, 200), (4, 200), (5, 300);

-- ROWS: 물리적 행 단위
SELECT
    id, val,
    SUM(val) OVER (ORDER BY val ROWS UNBOUNDED PRECEDING) AS rows_sum
FROM test_frame;

-- RANGE: 논리적 값 단위 (동일 값은 같은 그룹)
SELECT
    id, val,
    SUM(val) OVER (ORDER BY val RANGE UNBOUNDED PRECEDING) AS range_sum
FROM test_frame;
```

```
결과 비교:
ROWS:                          RANGE:
┌────┬─────┬──────────┐       ┌────┬─────┬───────────┐
│ id │ val │ rows_sum │       │ id │ val │ range_sum │
├────┼─────┼──────────┤       ├────┼─────┼───────────┤
│ 1  │ 100 │ 100      │       │ 1  │ 100 │ 200       │ ← 100이 2개
│ 2  │ 100 │ 200      │       │ 2  │ 100 │ 200       │ ← 동일
│ 3  │ 200 │ 400      │       │ 3  │ 200 │ 600       │ ← 200이 2개
│ 4  │ 200 │ 600      │       │ 4  │ 200 │ 600       │ ← 동일
│ 5  │ 300 │ 900      │       │ 5  │ 300 │ 900       │
└────┴─────┴──────────┘       └────┴─────┴───────────┘
```

### 5.3 GROUPS (PostgreSQL 11+)

```sql
-- GROUPS: 동일 ORDER BY 값을 하나의 그룹으로
SELECT
    id, val,
    SUM(val) OVER (
        ORDER BY val
        GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW
    ) AS groups_sum
FROM test_frame;
```

### 5.4 EXCLUDE 절 (PostgreSQL 11+)

```sql
-- 프레임에서 특정 행 제외
SELECT
    id, val,
    SUM(val) OVER (
        ORDER BY val
        ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
        EXCLUDE CURRENT ROW  -- 현재 행 제외
    ) AS sum_excluding_current
FROM test_frame;

-- EXCLUDE 옵션:
-- EXCLUDE NO OTHERS (기본값)
-- EXCLUDE CURRENT ROW
-- EXCLUDE GROUP (현재 행과 동일 값)
-- EXCLUDE TIES (동일 값 중 현재 행 제외)
```

---

## 6. 실전 활용 패턴

### 6.1 날짜별 누적 매출과 목표 달성률

```sql
SELECT
    sale_date,
    amount,
    SUM(amount) OVER (
        ORDER BY sale_date
    ) AS cumulative_sales,
    ROUND(
        SUM(amount) OVER (ORDER BY sale_date) * 100.0 / 10000,
        2
    ) AS target_pct  -- 목표: 10,000
FROM sales
ORDER BY sale_date;
```

### 6.2 이상치 탐지

```sql
-- 평균 ± 2 표준편차 이상인 데이터
WITH stats AS (
    SELECT
        salesperson,
        amount,
        AVG(amount) OVER (PARTITION BY salesperson) AS avg_amount,
        STDDEV(amount) OVER (PARTITION BY salesperson) AS stddev_amount
    FROM sales
)
SELECT *
FROM stats
WHERE amount > avg_amount + 2 * stddev_amount
   OR amount < avg_amount - 2 * stddev_amount;
```

### 6.3 연속 기록 분석

```sql
-- 연속 판매 일수 계산
WITH daily_sales AS (
    SELECT
        salesperson,
        sale_date,
        sale_date - (ROW_NUMBER() OVER (
            PARTITION BY salesperson
            ORDER BY sale_date
        ))::int AS grp
    FROM sales
)
SELECT
    salesperson,
    MIN(sale_date) AS streak_start,
    MAX(sale_date) AS streak_end,
    COUNT(*) AS streak_length
FROM daily_sales
GROUP BY salesperson, grp
ORDER BY salesperson, streak_start;
```

### 6.4 피벗 없는 행/열 비교

```sql
-- 현재 vs 전월 vs 전년 동월
SELECT
    salesperson,
    DATE_TRUNC('month', sale_date) AS month,
    SUM(amount) AS monthly_total,
    LAG(SUM(amount)) OVER (
        PARTITION BY salesperson
        ORDER BY DATE_TRUNC('month', sale_date)
    ) AS prev_month,
    LAG(SUM(amount), 12) OVER (
        PARTITION BY salesperson
        ORDER BY DATE_TRUNC('month', sale_date)
    ) AS same_month_last_year
FROM sales
GROUP BY salesperson, DATE_TRUNC('month', sale_date)
ORDER BY salesperson, month;
```

### 6.5 세션화 (Sessionization)

```sql
-- 30분 이상 간격이면 새 세션
WITH events AS (
    SELECT
        user_id,
        event_time,
        LAG(event_time) OVER (
            PARTITION BY user_id ORDER BY event_time
        ) AS prev_event_time
    FROM user_events
),
session_flags AS (
    SELECT
        user_id,
        event_time,
        CASE
            WHEN prev_event_time IS NULL THEN 1
            WHEN event_time - prev_event_time > INTERVAL '30 minutes' THEN 1
            ELSE 0
        END AS is_new_session
    FROM events
)
SELECT
    user_id,
    event_time,
    SUM(is_new_session) OVER (
        PARTITION BY user_id
        ORDER BY event_time
    ) AS session_id
FROM session_flags;
```

### 6.6 간격 채우기 (Gap Filling)

```sql
-- 날짜 시퀀스 생성 후 LEFT JOIN
WITH date_series AS (
    SELECT generate_series(
        '2024-01-01'::date,
        '2024-01-31'::date,
        '1 day'::interval
    )::date AS date
),
daily_totals AS (
    SELECT sale_date, SUM(amount) AS total
    FROM sales
    GROUP BY sale_date
)
SELECT
    ds.date,
    COALESCE(dt.total, 0) AS daily_total,
    SUM(COALESCE(dt.total, 0)) OVER (ORDER BY ds.date) AS running_total
FROM date_series ds
LEFT JOIN daily_totals dt ON ds.date = dt.sale_date
ORDER BY ds.date;
```

### 6.7 퍼센타일 계산

```sql
-- 백분위 계산
SELECT
    salesperson,
    amount,
    PERCENT_RANK() OVER (ORDER BY amount) AS pct_rank,
    CUME_DIST() OVER (ORDER BY amount) AS cume_dist,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) OVER () AS median
FROM sales;

-- 그룹별 중앙값
SELECT DISTINCT
    region,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount)
        OVER (PARTITION BY region) AS median_by_region
FROM sales;
```

---

## 7. 연습 문제

### 연습 1: 영업 성과 분석
각 영업사원의 판매 금액을 분석하여 다음을 계산하세요:
- 영업사원별 순위
- 전체 대비 비율
- 이전 판매 대비 증감

```sql
-- 예시 답안
SELECT
    salesperson,
    sale_date,
    amount,
    RANK() OVER (ORDER BY amount DESC) AS overall_rank,
    RANK() OVER (
        PARTITION BY salesperson
        ORDER BY amount DESC
    ) AS personal_rank,
    ROUND(amount * 100.0 / SUM(amount) OVER (), 2) AS pct_of_total,
    amount - LAG(amount) OVER (
        PARTITION BY salesperson ORDER BY sale_date
    ) AS change_from_prev
FROM sales
ORDER BY salesperson, sale_date;
```

### 연습 2: 이동 합계
최근 7일간의 이동 합계를 계산하세요.

```sql
-- 예시 답안
SELECT
    sale_date,
    amount,
    SUM(amount) OVER (
        ORDER BY sale_date
        RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW
    ) AS rolling_7day_sum
FROM sales
ORDER BY sale_date;
```

### 연습 3: 매출 목표 달성일 찾기
누적 매출이 5000을 처음 달성한 날짜를 찾으세요.

```sql
-- 예시 답안
SELECT sale_date, cumulative
FROM (
    SELECT
        sale_date,
        SUM(amount) OVER (ORDER BY sale_date) AS cumulative,
        LAG(SUM(amount) OVER (ORDER BY sale_date)) OVER (ORDER BY sale_date) AS prev_cumulative
    FROM sales
) sub
WHERE cumulative >= 5000
  AND (prev_cumulative IS NULL OR prev_cumulative < 5000)
LIMIT 1;
```

---

## 다음 단계
- [18. 테이블 파티셔닝](./18_Table_Partitioning.md)
- [14. JSON/JSONB 기능](./14_JSON_JSONB.md)

## 참고 자료
- [PostgreSQL Window Functions](https://www.postgresql.org/docs/current/functions-window.html)
- [Window Function Tutorial](https://www.postgresql.org/docs/current/tutorial-window.html)
- [SQL Window Functions](https://mode.com/sql-tutorial/sql-window-functions/)

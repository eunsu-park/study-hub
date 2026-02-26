# 조건과 정렬

**이전**: [CRUD 기본](./04_CRUD_Basics.md) | **다음**: [JOIN](./06_JOIN.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 비교 연산자(`=`, `<>`, `<`, `>`, `<=`, `>=`)를 사용하여 WHERE 절을 작성할 수 있습니다
2. 논리 연산자(AND, OR, NOT)를 조합하고 올바른 연산자 우선순위(precedence)를 적용할 수 있습니다
3. BETWEEN, IN, LIKE/ILIKE를 사용하여 범위 검사, 집합 멤버십, 패턴 매칭(pattern matching)을 수행할 수 있습니다
4. IS NULL, IS NOT NULL, COALESCE, NULLIF로 NULL 값을 올바르게 처리할 수 있습니다
5. 다중 컬럼 및 표현식 기반 정렬을 포함하여 ORDER BY로 쿼리 결과를 정렬할 수 있습니다
6. LIMIT, OFFSET, SQL 표준 FETCH 구문을 사용하여 페이지네이션(pagination)을 구현할 수 있습니다
7. DISTINCT와 DISTINCT ON을 적용하여 결과 집합에서 중복 행을 제거할 수 있습니다

---

테이블의 원시 데이터는 효율적으로 필터링하고 정렬하고 페이지를 넘길 수 있을 때 비로소 유용해집니다. 실제로 작성하는 거의 모든 쿼리에는 결과를 좁히는 WHERE 절과 의미 있는 순서로 제시하는 ORDER BY가 포함됩니다. 이러한 필터링과 정렬 기술은 데이터를 저장하는 것과 그로부터 실행 가능한 정보를 추출하는 것 사이의 다리 역할을 합니다.

## 1. WHERE 절 기본

WHERE 절은 조건에 맞는 행만 선택합니다.

```sql
SELECT * FROM users WHERE 조건;
UPDATE users SET ... WHERE 조건;
DELETE FROM users WHERE 조건;
```

---

## 2. 비교 연산자

| 연산자 | 설명 | 예시 |
|--------|------|------|
| `=` | 같음 | `age = 30` |
| `<>` 또는 `!=` | 다름 | `city <> '서울'` |
| `<` | 작음 | `age < 30` |
| `>` | 큼 | `age > 30` |
| `<=` | 작거나 같음 | `age <= 30` |
| `>=` | 크거나 같음 | `age >= 30` |

```sql
-- 나이가 30인 사용자
SELECT * FROM users WHERE age = 30;

-- 나이가 30이 아닌 사용자
SELECT * FROM users WHERE age <> 30;
SELECT * FROM users WHERE age != 30;

-- 나이가 25 이상 35 이하
SELECT * FROM users WHERE age >= 25 AND age <= 35;
```

---

## 3. 논리 연산자

### AND

모든 조건이 참이어야 합니다.

```sql
-- 서울에 사는 30대
SELECT * FROM users
WHERE city = '서울' AND age >= 30 AND age < 40;
```

### OR

하나 이상의 조건이 참이면 됩니다.

```sql
-- 서울 또는 부산에 사는 사용자
SELECT * FROM users
WHERE city = '서울' OR city = '부산';
```

### NOT

조건을 부정합니다.

```sql
-- 서울에 살지 않는 사용자
SELECT * FROM users WHERE NOT city = '서울';
SELECT * FROM users WHERE city <> '서울';  -- 동일

-- 30세 이상이 아닌 사용자
SELECT * FROM users WHERE NOT age >= 30;
SELECT * FROM users WHERE age < 30;  -- 동일
```

### 연산자 우선순위

`NOT` > `AND` > `OR` 순서로 처리됩니다. 괄호로 명확하게 표현하는 것이 좋습니다.

```sql
-- 의도와 다를 수 있음
SELECT * FROM users WHERE city = '서울' OR city = '부산' AND age >= 30;
-- 실제: 서울 전체 OR (부산 AND 30세 이상)

-- 괄호로 명확하게
SELECT * FROM users WHERE (city = '서울' OR city = '부산') AND age >= 30;
```

---

## 4. BETWEEN

범위 조건을 간단하게 표현합니다.

```sql
-- 나이가 25 이상 35 이하
SELECT * FROM users WHERE age BETWEEN 25 AND 35;
-- 동일: WHERE age >= 25 AND age <= 35

-- NOT BETWEEN
SELECT * FROM users WHERE age NOT BETWEEN 25 AND 35;

-- 날짜 범위
SELECT * FROM orders
WHERE created_at BETWEEN '2024-01-01' AND '2024-01-31';
```

---

## 5. IN

여러 값 중 하나와 일치하는지 확인합니다.

```sql
-- 서울, 부산, 대전 중 하나
SELECT * FROM users WHERE city IN ('서울', '부산', '대전');
-- 동일: WHERE city = '서울' OR city = '부산' OR city = '대전'

-- NOT IN
SELECT * FROM users WHERE city NOT IN ('서울', '부산');

-- 숫자에도 사용 가능
SELECT * FROM users WHERE age IN (25, 30, 35);

-- 서브쿼리와 함께
SELECT * FROM users WHERE id IN (SELECT user_id FROM orders);
```

---

## 6. LIKE - 패턴 매칭

### 와일드카드

| 기호 | 의미 |
|------|------|
| `%` | 0개 이상의 모든 문자 |
| `_` | 정확히 1개의 문자 |

```sql
-- '김'으로 시작하는 이름
SELECT * FROM users WHERE name LIKE '김%';

-- '수'로 끝나는 이름
SELECT * FROM users WHERE name LIKE '%수';

-- '영'이 포함된 이름
SELECT * FROM users WHERE name LIKE '%영%';

-- 정확히 3글자 이름
SELECT * FROM users WHERE name LIKE '___';

-- '김'으로 시작하는 2글자 이름
SELECT * FROM users WHERE name LIKE '김_';
```

### ILIKE - 대소문자 구분 없음

```sql
-- 대소문자 구분 없이 검색 (PostgreSQL 전용)
SELECT * FROM users WHERE email ILIKE '%KIM%';
SELECT * FROM users WHERE email ILIKE 'kim@%';
```

### NOT LIKE

```sql
SELECT * FROM users WHERE name NOT LIKE '김%';
```

### 이스케이프

```sql
-- 실제 %나 _를 검색할 때
SELECT * FROM products WHERE name LIKE '%50\%%' ESCAPE '\';  -- 50%가 포함된
```

---

## 7. NULL 처리

NULL은 "알 수 없는 값"으로, 일반 비교 연산자로는 비교할 수 없습니다.

### IS NULL / IS NOT NULL

```sql
-- 도시가 NULL인 사용자
SELECT * FROM users WHERE city IS NULL;

-- 도시가 NULL이 아닌 사용자
SELECT * FROM users WHERE city IS NOT NULL;

-- 잘못된 예 (항상 거짓)
SELECT * FROM users WHERE city = NULL;  -- 작동 안 함!
```

### COALESCE - NULL 대체값

```sql
-- COALESCE는 첫 번째 non-NULL 인자 반환 — 사용자 화면에서 NULL이
-- 빈칸으로 표시되거나 애플리케이션 코드에서 하류 오류를 일으키는 것을 방지
SELECT name, COALESCE(city, '미지정') AS city FROM users;

-- 여러 폴백을 체이닝: 전화번호 시도 후 이메일, 그 후 리터럴 기본값
SELECT COALESCE(phone, email, '연락처 없음') AS contact FROM users;
```

### NULLIF

```sql
-- 두 값이 같으면 NULL 반환
SELECT NULLIF(age, 0) FROM users;  -- age가 0이면 NULL

-- 0으로 나누기 방지
SELECT total / NULLIF(count, 0) FROM stats;
```

---

## 8. ORDER BY - 정렬

### 기본 정렬

```sql
-- 오름차순 (기본값)
SELECT * FROM users ORDER BY age;
SELECT * FROM users ORDER BY age ASC;

-- 내림차순
SELECT * FROM users ORDER BY age DESC;

-- 문자열 정렬
SELECT * FROM users ORDER BY name;  -- 가나다순
SELECT * FROM users ORDER BY name DESC;
```

### 다중 컬럼 정렬

```sql
-- 도시로 먼저 정렬, 같으면 나이로 정렬
SELECT * FROM users ORDER BY city, age;

-- 도시 오름차순, 나이 내림차순
SELECT * FROM users ORDER BY city ASC, age DESC;
```

### NULL 정렬 순서

```sql
-- NULL을 마지막으로 (기본값: ASC에서 NULL이 마지막)
SELECT * FROM users ORDER BY city NULLS LAST;

-- NULL을 처음으로
SELECT * FROM users ORDER BY city NULLS FIRST;

-- DESC에서 NULL 처리
SELECT * FROM users ORDER BY city DESC NULLS LAST;
```

### 표현식으로 정렬

```sql
-- 이름 길이로 정렬
SELECT * FROM users ORDER BY LENGTH(name);

-- 계산 결과로 정렬
SELECT name, age, age * 12 AS months FROM users ORDER BY months DESC;

-- 컬럼 위치로 정렬 (1-based)
SELECT name, email, age FROM users ORDER BY 3 DESC;  -- age로 정렬
```

---

## 9. LIMIT / OFFSET - 결과 제한

### LIMIT

```sql
-- 상위 5개만
SELECT * FROM users LIMIT 5;

-- 나이가 많은 순서로 상위 3명
SELECT * FROM users ORDER BY age DESC LIMIT 3;
```

### OFFSET

```sql
-- 처음 5개 건너뛰고 그 다음부터
SELECT * FROM users ORDER BY id OFFSET 5;

-- 페이지네이션: 6번째부터 5개
SELECT * FROM users ORDER BY id LIMIT 5 OFFSET 5;
```

### 페이지네이션 계산

```sql
-- 페이지 1 (1~10번)
SELECT * FROM users ORDER BY id LIMIT 10 OFFSET 0;

-- 페이지 2 (11~20번)
SELECT * FROM users ORDER BY id LIMIT 10 OFFSET 10;

-- 페이지 N (계산: OFFSET = (N-1) * 페이지크기)
SELECT * FROM users ORDER BY id LIMIT 10 OFFSET 20;  -- 페이지 3
```

### FETCH (SQL 표준)

```sql
-- LIMIT과 동일
SELECT * FROM users
ORDER BY age DESC
FETCH FIRST 5 ROWS ONLY;

-- OFFSET과 함께
SELECT * FROM users
ORDER BY id
OFFSET 10 ROWS
FETCH NEXT 5 ROWS ONLY;
```

---

## 10. DISTINCT - 중복 제거

```sql
-- 중복 도시 제거
SELECT DISTINCT city FROM users;

-- 여러 컬럼 조합의 중복 제거
SELECT DISTINCT city, age FROM users;

-- COUNT와 함께
SELECT COUNT(DISTINCT city) FROM users;
```

### DISTINCT ON (PostgreSQL 전용)

```sql
-- 각 도시별로 첫 번째 사용자만
SELECT DISTINCT ON (city) * FROM users ORDER BY city, created_at;

-- 각 도시별로 가장 나이 많은 사용자
SELECT DISTINCT ON (city) * FROM users ORDER BY city, age DESC;
```

---

## 11. 실습 예제

### 샘플 데이터

```sql
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    category VARCHAR(100),
    price NUMERIC(10, 2),
    stock INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO products (name, category, price, stock) VALUES
('맥북 프로 14', '노트북', 2490000, 50),
('맥북 에어 M2', '노트북', 1590000, 100),
('갤럭시북 프로', '노트북', 1790000, 30),
('아이패드 프로', '태블릿', 1290000, 80),
('갤럭시탭 S9', '태블릿', 1190000, 60),
('에어팟 프로', '이어폰', 329000, 200),
('갤럭시버즈2', '이어폰', 179000, 150),
('애플워치 9', '스마트워치', 599000, 70),
('갤럭시워치6', '스마트워치', 399000, 90),
('아이폰 15', '스마트폰', 1250000, 120),
('갤럭시 S24', '스마트폰', 1150000, NULL);
```

### 실습 1: 기본 조건 검색

```sql
-- 1. 노트북 카테고리 상품
SELECT * FROM products WHERE category = '노트북';

-- 2. 가격이 100만원 이상인 상품
SELECT * FROM products WHERE price >= 1000000;

-- 3. 재고가 100개 이상인 상품
SELECT * FROM products WHERE stock >= 100;

-- 4. 노트북이면서 가격이 200만원 이하인 상품
SELECT * FROM products
WHERE category = '노트북' AND price <= 2000000;
```

### 실습 2: 복합 조건

```sql
-- 1. 노트북 또는 태블릿
SELECT * FROM products
WHERE category IN ('노트북', '태블릿')
ORDER BY price DESC;

-- 2. 가격이 50만원~150만원 사이
SELECT * FROM products
WHERE price BETWEEN 500000 AND 1500000
ORDER BY price;

-- 3. 이름에 '프로'가 포함된 상품
SELECT * FROM products WHERE name LIKE '%프로%';

-- 4. 재고가 NULL이거나 0인 상품
SELECT * FROM products
WHERE stock IS NULL OR stock = 0;
```

### 실습 3: 정렬과 페이지네이션

```sql
-- 1. 가격 높은 순서로 상위 5개
SELECT * FROM products ORDER BY price DESC LIMIT 5;

-- 2. 카테고리별, 가격 낮은 순서
SELECT * FROM products ORDER BY category, price;

-- 3. 페이지 2 (6~10번째 상품)
SELECT * FROM products ORDER BY id LIMIT 5 OFFSET 5;

-- 4. 각 카테고리별 가장 비싼 상품
SELECT DISTINCT ON (category) *
FROM products
ORDER BY category, price DESC;
```

### 실습 4: NULL 처리

```sql
-- 1. 재고가 없거나 NULL인 상품
SELECT name, COALESCE(stock, 0) AS stock FROM products
WHERE stock IS NULL OR stock = 0;

-- 2. NULL을 '재고 확인 중'으로 표시
SELECT name, COALESCE(stock::TEXT, '재고 확인 중') AS stock_status
FROM products;

-- 3. NULL을 마지막으로 정렬
SELECT * FROM products ORDER BY stock NULLS LAST;
```

---

## 12. 성능 팁

### 인덱스 활용

```sql
-- 인덱스는 O(n) 순차 스캔을 O(log n) B-tree 검색으로 변환.
-- WHERE, JOIN, ORDER BY에 나오는 컬럼에만 생성 — 각 인덱스는
-- INSERT/UPDATE 시 인덱스 유지 비용(쓰기 오버헤드)을 추가
CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_products_price ON products(price);

-- 복합 인덱스
CREATE INDEX idx_products_cat_price ON products(category, price);
```

### LIKE 패턴 최적화

```sql
-- 접두사 패턴은 시작점 고정 — B-tree가 정렬된 값을 이진 검색 가능
WHERE name LIKE '맥북%'

-- 선행 와일드카드는 모든 행 스캔 필요 — 이 패턴이 빈번하면
-- pg_trgm GIN 인덱스나 전문 검색(Full-Text Search) 고려
WHERE name LIKE '%맥북%'
```

### LIMIT 먼저 적용

```sql
-- 정렬 후 LIMIT (비효율적일 수 있음)
SELECT * FROM products ORDER BY price DESC LIMIT 10;

-- 인덱스가 있으면 효율적
CREATE INDEX idx_products_price_desc ON products(price DESC);
```

---

**이전**: [CRUD 기본](./04_CRUD_Basics.md) | **다음**: [JOIN](./06_JOIN.md)

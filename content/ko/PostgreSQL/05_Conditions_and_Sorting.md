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

WHERE/ORDER BY 문법으로 들어가기 전, [**이론과 원리**](#이론과-원리) 절을 먼저 읽으세요 — 인덱스가 `WHERE` 술어(predicate)를 어떻게 B-tree range scan으로 변환하는지, `NULL`이 모든 것을 3-valued로 만드는 이유, 그리고 planner가 sequential scan과 index scan 중 어떤 것을 고르는지를 다룹니다.

---

## 이론과 원리

`WHERE` 절은 단순한 필터가 아닙니다. planner가 구조화된 access path — sequential scan, index scan, index-only scan, bitmap heap scan, 또는 이들의 조합 — 로 변환하려고 시도하는 *술어(predicate)*입니다. 쿼리가 3 ms에 끝나느냐 3 s에 끝나느냐는 거의 전적으로 planner가 어떤 path를 고르는지에 달려 있고, 그것은 술어가 **sargable**(Search ARGument-able — 인덱스가 사용할 수 있는)인지, 그리고 얼마나 selective한지에 달려 있습니다. `ORDER BY`에도 동일한 논리가 적용됩니다 — B-tree 인덱스가 이미 요청된 순서로 행을 저장하고 있다면 정렬 단계가 통째로 사라질 수 있습니다.

이 절에서 다루는 내용:

- **(A)** B-tree 내부 — 정렬된 page, range scan, `=`, `<`, `>`, `BETWEEN`이 동일한 자료구조에서 작동하는 이유.
- **(B)** Sargability — 어떤 `WHERE` 술어가 인덱스를 사용할 수 있고 어떤 것이 못 하는지, 그 이유.
- **(C)** 3-valued logic — `NULL`이 비교, `AND`, `OR`, `NOT`, `IN`과 어떻게 상호작용하는지.
- **(D)** sequential vs index vs bitmap scan — planner의 세 가지 옵션과 하나를 고르는 cost model.

### A. B-tree — 기본 인덱스 타입

`CREATE INDEX idx ON t(col);`(`USING` 절 없이)은 **B-tree** — 정확히는 Lehman-Yao concurrent B-tree — 를 만듭니다. 디스크 모양은:

```
            ┌────────────────┐
            │  Root page     │   (1 page, internal page들을 가리킴)
            └───┬─────┬──────┘
                │     │
        ┌───────┘     └────────┐
   ┌────┴──────┐         ┌─────┴──────┐
   │ Internal  │   ...   │  Internal  │     (depth = log₂(N) / log₂(branch_factor))
   └─┬───┬─────┘         └─────┬──────┘
     │   │                     │
   ┌─┴┐ ┌┴───┐              ┌──┴───┐
   │L │ │ L  │   ...        │  L   │              (Leaf page — 정렬됨)
   │  │ │    │              │      │
   └──┘ └────┘              └──────┘
   ←────── leaf들의 linked list ──────→
```

leaf는 **key 정렬 순서**로 있고, 좌→우로 연결되어 있습니다. 따라서 key(또는 key 범위)가 주어지면 PostgreSQL은 root에서 적합한 leaf까지 navigate하고, leaf list를 앞으로(또는 뒤로 — 양방향 연결) 스캔하며 따라갑니다.

#### A.1 B-tree가 `=`, `<`, `>`, `BETWEEN`, `ORDER BY`를 공짜로 처리하는 이유

네 가지 연산 모두 "leaf로 navigate 후 sequential 읽기"로 환원됩니다.

- `WHERE x = 5` — key `5`를 포함하는 첫 leaf로 내려가서, key가 바뀔 때까지 읽음.
- `WHERE x > 5` — `x > 5`인 첫 leaf로 내려가서, 끝까지 앞으로 읽음.
- `WHERE x BETWEEN 5 AND 10` — `x ≥ 5`인 첫 leaf로 내려가서, `x > 10`이면 멈춤.
- `ORDER BY x` — leftmost leaf로 내려가서 앞으로 walk — 별도 정렬 단계 없음.
- `ORDER BY x DESC` — rightmost leaf로 내려가서 뒤로 walk.

이 때문에 B-tree 하나가 다양한 쿼리를 처리합니다. Hash, GIN, GiST, BRIN — 이후 다룸 — 은 B-tree가 처리할 수 없는 경우를 위해 존재합니다.

#### A.2 다중 컬럼 B-tree와 leftmost-prefix 규칙

`CREATE INDEX idx ON t(a, b, c);`은 행을 `a` 순으로, `a`가 같으면 `b` 순으로, `(a, b)`가 같으면 `c` 순으로 정렬합니다. 따라서 인덱스는 다음에 사용 가능합니다.

- `WHERE a = ?`
- `WHERE a = ? AND b = ?`
- `WHERE a = ? AND b = ? AND c = ?`
- `WHERE a = ? AND b > ?`
- `WHERE a > ?` (첫 컬럼 range)

그러나 *불가능*(또는 비효율적):

- `WHERE b = ?` 단독 — 인덱스가 `a` 순이라 모든 `a` 값을 스캔해야 함.
- `WHERE c = ?` 단독 — 같은 이유, 더 깊음.

이것이 **leftmost-prefix 규칙**입니다. 인덱스 컬럼 순서는 어떤 쿼리가 빠른지를 못 박는 설계 선택입니다.

### B. Sargability — 인덱스가 사용할 수 있는 술어

술어가 **sargable**이라는 것은, indexed expression에 대한 key range로 다시 쓰일 수 있다는 뜻입니다. sargable 형태:

```sql
WHERE x = 10                  -- indexed 컬럼에 =
WHERE x > 10                  -- range
WHERE x BETWEEN 10 AND 20     -- 복합 range
WHERE x IN (1, 2, 3)          -- 여러 equality probe
WHERE name LIKE 'abc%'        -- prefix LIKE — sargable!
```

non-sargable 형태 — 인덱스 사용을 막음:

```sql
WHERE LOWER(name) = 'alice'   -- indexed 컬럼에 함수
WHERE x + 1 = 10              -- indexed 컬럼에 산술
WHERE name LIKE '%abc'        -- 선두 wildcard — navigate할 prefix 없음
WHERE x::text = '10'          -- indexed 컬럼에 cast
WHERE date_part('year', d) = 2026  -- 함수 호출이 indexable 형태를 가림
```

#### B.1 함수가 인덱스 사용을 죽이는 이유

`name`에 대한 B-tree는 `name` 값을 저장하지, `LOWER(name)`을 저장하지 않습니다. planner는 `LOWER(name) = 'alice'`인 행을 찾기 위해 `name`의 B-tree를 내려갈 방법이 없습니다 — `LOWER`를 평가하려면 모든 행을 읽어야 합니다. 해결책은 쿼리를 다시 쓰거나, **functional index**를 만드는 것 — `CREATE INDEX ON t (LOWER(name));`. 이제 인덱스는 `LOWER` 값을 저장하고, 술어는 *이 새 인덱스에 대해* sargable이 됩니다.

#### B.2 prefix LIKE는 sargable이지만 suffix LIKE는 아닌 이유

`WHERE name LIKE 'abc%'`은 `WHERE name >= 'abc' AND name < 'abd'`와 동치입니다 — B-tree가 native하게 처리하는 key range. `WHERE name LIKE '%abc'`는 어떤 range로도 환원될 수 없습니다 — matching key가 인덱스 전반에 흩어져 있습니다. suffix나 substring 매치에는 다른 인덱스 타입(`pg_trgm` GIN/GiST — 19번 레슨)이 필요합니다.

### C. 3-valued logic와 NULL

SQL은 *3-valued*입니다 — 모든 boolean 표현식은 TRUE, FALSE, 또는 NULL("UNKNOWN")로 평가됩니다. NULL은 "값을 모름"이라는 뜻이며, 모르는 값이 관여한 어떤 연산도 그 자체로 모름입니다.

| 표현식 | 결과 |
|--------|------|
| `5 = NULL` | NULL (FALSE 아님!) |
| `5 <> NULL` | NULL |
| `NULL = NULL` | NULL |
| `NULL AND TRUE` | NULL |
| `NULL AND FALSE` | FALSE (false absorbs) |
| `NULL OR TRUE` | TRUE (true absorbs) |
| `NULL OR FALSE` | NULL |
| `NOT NULL` | NULL |

`WHERE`는 술어가 TRUE로 평가될 때만 행을 유지합니다 — 따라서 NULL 결과는 걸러집니다. 이로부터 놀라움이 생깁니다.

```sql
SELECT count(*) FROM users WHERE age <> 30;
-- age IS NULL인 user는 포함되지 않음!
```

NULL을 명시적으로 처리하려면 `IS NULL`, `IS NOT NULL`(항상 TRUE 또는 FALSE 반환 — NULL이 절대 아님) 또는 `IS DISTINCT FROM`(NULL-safe `<>`)을 사용합니다.

#### C.1 NULL과 IN

`WHERE x IN (1, 2, NULL)`은 정확히 `WHERE x = 1 OR x = 2 OR x = NULL`입니다. 마지막 항은 항상 NULL. 따라서 표현식은 `x = 1`이거나 `x = 2`이면 TRUE, 아니면 NULL — NULL인 행은 걸러집니다. 이것은 괜찮습니다.

그러나 `WHERE x NOT IN (1, 2, NULL)`은 `WHERE x <> 1 AND x <> 2 AND x <> NULL` — 마지막 AND가 절대 TRUE일 수 없으므로, *전체* 절이 NULL 또는 FALSE이고, *어떤 행도 매치하지 않습니다*. SQL의 가장 유명한 함정입니다 — `NOT IN` 목록에서는 항상 NULL을 제외하거나, 대신 `NOT EXISTS`를 사용합니다.

### D. Sequential, Index, Bitmap Scan

sargable한 술어가 주어지면, planner는 **selectivity**(술어가 매치하는 행의 비율)에 따라 세 가지 access method 중 하나를 고릅니다.

#### D.1 Sequential scan

heap의 모든 page를 처음부터 끝까지 읽고, 행마다 술어를 평가. 비용은 테이블 크기에 비례. selectivity가 높을 때(반환 행 > ~10%) 유리 — random I/O가 필요 없고 page를 큰 prefetch 청크로 읽을 수 있기 때문.

#### D.2 Index scan

B-tree를 walk해서 matching key를 찾고, 각 key마다 heap pointer를 따라 행을 읽음. 비용은 `(matching_rows × random_page_cost) + log B-tree depth`. selectivity가 낮을 때(반환 행 < ~1%) 유리. selectivity가 높으면 random heap 읽기가 sequential 읽기의 4배(`random_page_cost` 기본 4.0, `seq_page_cost` 기본 1.0)이므로 크게 손해.

#### D.3 Bitmap scan

중간 selectivity(1%–10%)에서는 planner가 hybrid를 사용:

1. **Index scan**으로 matching heap page number의 **bitmap**을 만듦.
2. bitmap을 page number로 **정렬**.
3. heap page를 순서대로 **sequential 읽기**, 행마다 술어 적용.

이는 random 읽기를 sequential 읽기로 변환합니다 — bitmap을 materialize하는 비용을 치르고. 큰 결과 집합에서 순수 index scan보다 극적으로 빠릅니다.

#### D.4 Index-only scan

쿼리가 필요로 하는 모든 컬럼이 인덱스 안에 있다면("covering index"), PostgreSQL은 heap을 만지지 않고 인덱스만으로 답할 수 있습니다. heap MVCC header를 확인하지 않고도 행이 visible임을 확인하기 위해 **visibility map**이 필요합니다. 이 목적만으로 non-key 컬럼을 추가하려면 `CREATE INDEX ... INCLUDE (col1, col2)`를 사용합니다.

### 이론에서 아래 SQL로

이어지는 각 절은 위 메커니즘이 구체화된 형태입니다:

- **`=`, `<`, `>`을 포함한 `WHERE`** — B-tree 혜택을 받는 sargable 술어 (§A, §B).
- **`WHERE LIKE 'abc%'`** — sargable, `WHERE LIKE '%abc'`는 아님 (§B.2).
- **`AND`, `OR`, `NOT`** — 3-valued logic, `NULL`이 유지하거나 제외하려던 행을 가릴 수 있음 (§C).
- **`IS NULL`, `IS NOT NULL`** — NULL-safe인 유일한 술어 (§C).
- **`ORDER BY col`, `ORDER BY col DESC`** — 매칭되는 B-tree가 있으면 무료, 아니면 명시적 정렬 단계가 트리거됨 (§A.1).
- **`LIMIT n`** — `ORDER BY`와 결합해 일찍 스캔을 멈추는 "top-N" 최적화를 enable.

---

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

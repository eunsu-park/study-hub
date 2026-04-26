# 14. PostgreSQL JSON/JSONB 기능

**이전**: [백업과 운영](./13_Backup_and_Operations.md) | **다음**: [쿼리 최적화](./15_Query_Optimization.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 수행할 수 있습니다:

1. JSON과 JSONB 데이터 타입을 구별하고, 주어진 사용 사례에 적합한 타입을 선택한다
2. PostgreSQL 내장 함수와 연산자를 사용하여 JSON 데이터를 저장, 수정, 삭제한다
3. 화살표 연산자(`->`, `->>`, `#>`, `#>>`)를 사용하여 중첩 JSON 구조를 조회한다
4. JSONB 데이터에 포함 연산자(`@>`), 존재 연산자(`?`), 병합 연산자(`||`)를 적용한다
5. JSONB 컬럼에 GIN 인덱스(GIN index)를 생성하고 EXPLAIN ANALYZE로 사용 여부를 검증한다
6. 스키마리스(schema-less) 이벤트 로깅, EAV(Entity-Attribute-Value) 대체, 문서 버전 관리 등 실전 패턴을 구현한다

---

현대 애플리케이션은 사용자 설정, API 응답, 이벤트 페이로드, 설정 데이터 등 엄격한 관계형 컬럼에 맞지 않는 반구조화 데이터(semi-structured data)를 자주 다룹니다. PostgreSQL의 네이티브 JSON/JSONB 지원을 사용하면 이러한 유연한 데이터를 관계형 테이블과 함께 저장하고 조회할 수 있으며, 완전한 인덱싱과 트랜잭션 보장이 제공됩니다. 이를 통해 많은 아키텍처에서 별도의 문서 데이터베이스가 불필요해지고, SQL의 강력함을 유지하면서 스택을 단순화할 수 있습니다.

## 목차

연산자와 함수로 들어가기 전, [**이론과 원리**](#이론과-원리) 절을 먼저 읽으세요 — JSONB의 binary parse-tree 표현, containment 쿼리를 위한 GIN의 inverted-index 구조, 그리고 JSON path 언어 의미론을 다룹니다.

1. [JSON vs JSONB](#1-json-vs-jsonb)
2. [JSON 데이터 저장](#2-json-데이터-저장)
3. [JSON 연산자](#3-json-연산자)
4. [JSON 함수](#4-json-함수)
5. [인덱싱과 성능](#5-인덱싱과-성능)
6. [실전 패턴](#6-실전-패턴)
7. [연습 문제](#7-연습-문제)

---

## 이론과 원리

PostgreSQL의 JSON 지원은 텍스트의 얇은 wrapper가 아닙니다. JSONB는 문서를 *parsing된 binary tree*로 저장하고, GIN 인덱스는 그 트리를 키별 posting list로 inverted시키며, JSON path 언어(PG 12+)는 트리를 재-parsing 없이 walk하는 executor로 컴파일됩니다. JSONB의 on-disk 모양과 GIN 인덱스의 구조를 이해하는 것이 "테이블 전체를 스캔하는 JSONB 쿼리"와 "마이크로초 안에 인덱스 probe로 끝나는 JSONB 쿼리"의 차이입니다.

이 절에서 다루는 내용:

- **(A)** JSONB의 on-disk 포맷 — header, key-value 쌍, 정렬, TOAST.
- **(B)** Inverted index로서의 GIN — posting list, `jsonb_ops` vs `jsonb_path_ops` operator class.
- **(C)** arrow/path 연산자(`->`, `->>`, `#>`, `#>>`)와 containment 연산자(`@>`).
- **(D)** JSON path 언어와 `jsonb_path_query` — 선언적 트리 traversal.

### A. JSONB On-Disk 포맷

JSON 값(`{"a": 1, "b": [2, 3]}`)은 1번 *parsing*되어 compact binary 포맷의 parse tree로 JSONB로 저장됩니다.

```
JSONB header (4 바이트)
├── Type tag — object | array | scalar
├── Children 수 (count)
└── Total size

각 child에 대해:
├── JEntry (4 바이트) — type + offset/length
└── Key/value 바이트 (object의 경우 key로 정렬)
```

#### A.1 "binary"가 중요한 이유

JSON-as-text와 비교해, JSONB는:

- 모든 연산자 호출에서 **재-parsing을 건너뜀**. `doc->'a'`는 키 `'a'`의 JEntry를 직접 읽음.
- parse 시점에 **object key를 정렬**(사전순), key 조회를 linear scan이 아니라 O(log K) binary search로.
- **중복 key 제거** — `{"a": 1, "a": 2}`는 `{"a": 2}`가 됨. JSON 텍스트는 둘 다 유지.
- **공백과 key 순서 손실**. 그것을 보존하려면 `jsonb` 대신 `json`(text) 사용.

#### A.2 TOAST와 JSONB

JSONB 컬럼은 TOAST 가능(03번 레슨 §C). ~2 KB보다 큰 문서는 압축되거나 TOAST 테이블로 옮겨짐. 결정적 결과 — 1 MB 문서에서 단일 key를 추출해도 여전히 문서 전체를 읽고 압축 해제해야 함, 부분-트리 접근 없음. 작은 문서를 많이 저장하는 설계가 보통 거대한 문서 하나를 저장하는 설계보다 빠릅니다.

### B. GIN — JSONB를 위한 Inverted Index

JSONB 컬럼에 대한 GIN(Generalized Inverted Index)은 구조를 뒤집음 — "각 행에 무엇이 들어 있나" 대신 "각 값에 대해 어느 행이 그것을 포함하나"를 저장.

#### B.1 두 operator class

PostgreSQL은 JSONB에 두 GIN operator class를 제공:

**`jsonb_ops`** (기본) — 모든 key, 모든 value, 모든 key:value 쌍을 인덱스.

```
row 5 — {"tags": ["red", "fast"], "color": "red"}에 대해:
GIN이 항목 삽입 — 'tags', 'red', 'fast', 'color', 'tags':'red', 'tags':'fast', 'color':'red'
```

`@>`, `?`, `?|`, `?&` 연산자 지원. 인덱스 큼, 쓰기 느림, 그러나 유연.

**`jsonb_path_ops`** — leaf value까지의 완전한 path만 hash해서 인덱스.

```
row 5 — {"tags": ["red", "fast"], "color": "red"}에 대해:
GIN이 항목 삽입 — hash('tags'->'red'), hash('tags'->'fast'), hash('color'->'red')
```

`@>`만 지원(containment). 대략 1/3 크기, `@>` 쿼리에 3× 빠른 lookup — 그러나 "이 행이 key X를 가지나"(`?`)는 답 불가.

#### B.2 Lookup 알고리즘

`WHERE jsonb_col @> '{"color": "red"}'`에 대해:

1. 쿼리 값에서 **검색 가능 항목 추출** — `'color', 'red', 'color':'red'`(또는 `path_ops`의 경우 `hash(color->red)`만).
2. 각 항목을 GIN B-tree에서 **lookup**해서 row TID의 posting list 획득.
3. posting list **교집합**.
4. 살아남은 각 TID에 대해 실제 행을 **recheck**해서 완전한 containment 확인(인덱스는 개별 element에 대한 것이고, AND-of-elements는 containment와 같지 않으므로 검증 필요).

recheck 단계는 GIN이 *후보 집합*을 준다는 뜻 — 보통 충분히 작아서 recheck 비용이 무시할 만함.

### C. Arrow 연산자와 Containment

이 레슨의 연산자는 두 카테고리로 나뉨.

#### C.1 Path 추출 — `->`, `->>`, `#>`, `#>>`

| 연산자 | 반환 타입 | 의미 |
|--------|----------|------|
| `->` | jsonb | 키(object) 또는 인덱스(array)로 child 가져오기 |
| `->>` | text | 같음, 결과를 text로 cast |
| `#>` | jsonb | path 배열로 descendant 가져오기 |
| `#>>` | text | `#>`와 같음, text로 cast |

```sql
'{"a":{"b":1}}'::jsonb -> 'a'           -- {"b":1}    (jsonb)
'{"a":{"b":1}}'::jsonb -> 'a' ->> 'b'   -- '1'        (text)
'{"a":{"b":1}}'::jsonb #> '{a,b}'       -- 1          (jsonb)
'{"a":{"b":1}}'::jsonb #>> '{a,b}'      -- '1'        (text)
```

이들은 `IMMUTABLE`이고 expression 인덱스로 indexable. `CREATE INDEX ON t ((data->>'email'));`은 `WHERE data->>'email' = ?`이 B-tree를 사용하게 함.

#### C.2 Containment와 existence — `@>`, `?`, `?|`, `?&`

| 연산자 | 의미 |
|--------|------|
| `@>` | 좌측이 우측을 포함(deep, 의미적) |
| `?` | top-level key 존재 |
| `?|` | 나열된 key 중 하나라도 존재 |
| `?&` | 나열된 key가 모두 존재 |

Containment `@>`가 일꾼. `'{"a":1, "b":2}'::jsonb @> '{"a":1}'::jsonb`는 좌측이 우측의 모든 것을 포함하므로 true. 속도를 위해 GIN 사용.

### D. JSON Path 언어 — `jsonb_path_query`

PostgreSQL 12가 SQL/JSON path 언어(Oracle과 SQL 표준이 사용하는 동일한 것)를 추가. path 표현식은 JSONB 트리를 walk하는 executor로 컴파일됨.

#### D.1 기본 문법

```sql
SELECT jsonb_path_query(
    '{"users":[{"name":"Alice","age":30},{"name":"Bob","age":25}]}',
    '$.users[*] ? (@.age > 28).name'
);
-- 반환 — "Alice"
```

- `$` — 문서의 root
- `.users` — key로 child
- `[*]` — 모든 배열 element
- `? (predicate)` — 필터
- `@` — 필터 안에서 현재 항목
- `.name` — 최종 추출

#### D.2 Arrow 연산자와 비교

Arrow 연산자는 단순하지만 제한적 — 한 번에 한 단계, 필터링 없음. JSON path는 전체 traversal을 하나의 선언적 표현식으로 표현, executor가 단위로 최적화. 깊이 nested되거나 필터링된 쿼리에서 JSON path가 더 짧고 빠름.

### 이론에서 아래 SQL로

이어지는 각 절은 위 메커니즘이 구체화된 형태입니다:

- **`json` vs `jsonb`** — text vs binary parse tree (§A).
- **`->`, `->>`, `#>`, `#>>`** — path 추출 (§C.1).
- **`@>`, `?`, `?|`, `?&`** — containment와 existence, GIN으로 가속 (§B, §C.2).
- **`CREATE INDEX ... USING gin (jsonb_col)`** — 기본 `jsonb_ops`, 유연하지만 큼.
- **`CREATE INDEX ... USING gin (jsonb_col jsonb_path_ops)`** — 더 작고, `@>`만 지원하지만 더 빠름.
- **`jsonb_path_query`, `jsonb_path_exists`, `@@`** — 완전한 SQL/JSON path 언어 (§D).

---

## 1. JSON vs JSONB

### 1.1 타입 비교

```
┌─────────────────────────────────────────────────────────────┐
│                    JSON vs JSONB                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  JSON                          JSONB                        │
│  ────────────────────         ────────────────────         │
│  • 텍스트로 저장               • 바이너리로 저장            │
│  • 입력 그대로 유지            • 파싱 후 저장               │
│  • 공백/순서 보존              • 공백 제거, 키 정렬         │
│  • 중복 키 허용                • 마지막 키 값만 유지        │
│  • 저장 빠름                   • 저장 약간 느림             │
│  • 처리 느림 (매번 파싱)       • 처리 빠름                  │
│  • 인덱싱 제한적               • GIN 인덱스 지원            │
│                                                             │
│  권장: 대부분의 경우 JSONB 사용                             │
│        JSON은 원본 형식 유지 필요할 때만 사용               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 기본 사용

```sql
-- JSONB는 사전 파싱된 바이너리 저장 — GIN 인덱싱 및 포함 연산자(@>, ?) 지원
-- JSON은 원시 텍스트 저장 — 매 접근마다 재파싱, 인덱스 미지원; 쓰기 전용/원시 읽기에만 사용
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    attributes JSONB,    -- JSONB 권장
    raw_data JSON        -- 원본 보존 필요 시
);

-- 데이터 삽입
INSERT INTO products (name, attributes) VALUES
('Laptop', '{"brand": "Dell", "specs": {"cpu": "i7", "ram": 16}}'),
('Phone', '{"brand": "Apple", "specs": {"model": "iPhone 15", "storage": 256}}');

-- JSON 형식 검증
SELECT '{"valid": true}'::jsonb;  -- 성공
SELECT '{invalid}'::jsonb;        -- 오류: 유효하지 않은 JSON
```

---

## 2. JSON 데이터 저장

### 2.1 JSON 생성 함수

```sql
-- json_build_object: 키-값 쌍으로 객체 생성
SELECT json_build_object(
    'name', 'John',
    'age', 30,
    'active', true
);
-- {"name": "John", "age": 30, "active": true}

-- jsonb_build_object (JSONB 버전)
SELECT jsonb_build_object(
    'product', 'Laptop',
    'price', 999.99
);

-- json_build_array: 배열 생성
SELECT json_build_array(1, 2, 'three', true, null);
-- [1, 2, "three", true, null]

-- row_to_json: 행을 JSON으로
SELECT row_to_json(t)
FROM (SELECT 1 AS id, 'test' AS name) t;
-- {"id": 1, "name": "test"}

-- to_jsonb: 값을 JSONB로 변환
SELECT to_jsonb(ARRAY[1, 2, 3]);
-- [1, 2, 3]

-- json_agg: 여러 행을 배열로
SELECT json_agg(name) FROM products;
-- ["Laptop", "Phone"]

-- jsonb_object_agg: 키-값 쌍을 객체로
SELECT jsonb_object_agg(name, id) FROM products;
-- {"Laptop": 1, "Phone": 2}
```

### 2.2 JSON 데이터 수정

```sql
-- jsonb_set: 값 설정/추가
UPDATE products
SET attributes = jsonb_set(attributes, '{specs,ram}', '32')
WHERE name = 'Laptop';

-- 중첩 경로 추가 (create_if_missing = true)
UPDATE products
SET attributes = jsonb_set(
    attributes,
    '{specs,gpu}',
    '"RTX 4090"',
    true  -- 경로가 없으면 생성
)
WHERE name = 'Laptop';

-- 여러 값 한 번에 수정
UPDATE products
SET attributes = attributes || '{"color": "silver", "weight": 2.1}'
WHERE name = 'Laptop';

-- 키 삭제
UPDATE products
SET attributes = attributes - 'color'
WHERE name = 'Laptop';

-- 중첩 키 삭제
UPDATE products
SET attributes = attributes #- '{specs,gpu}'
WHERE name = 'Laptop';

-- 배열 요소 추가
UPDATE products
SET attributes = jsonb_set(
    attributes,
    '{tags}',
    COALESCE(attributes->'tags', '[]'::jsonb) || '"new_tag"'
);
```

---

## 3. JSON 연산자

### 3.1 접근 연산자

```sql
-- -> : JSON 객체/배열 요소 (JSON 반환)
SELECT attributes->'brand' FROM products;
-- "Dell" (따옴표 포함 JSON)

-- ->> : 텍스트로 추출
SELECT attributes->>'brand' FROM products;
-- Dell (텍스트)

-- #> : 경로로 접근 (JSON 반환)
SELECT attributes#>'{specs,cpu}' FROM products;
-- "i7"

-- #>> : 경로로 접근 (텍스트 반환)
SELECT attributes#>>'{specs,cpu}' FROM products;
-- i7

-- 배열 접근
SELECT '[1, 2, 3]'::jsonb->0;   -- 1
SELECT '[1, 2, 3]'::jsonb->-1;  -- 3 (마지막)
SELECT '[1, 2, 3]'::jsonb->10;  -- NULL (범위 초과)
```

### 3.2 비교 연산자 (JSONB 전용)

```sql
-- = : 동등 비교
SELECT * FROM products
WHERE attributes->'brand' = '"Dell"'::jsonb;

-- @> : 포함 (왼쪽이 오른쪽 포함)
-- 필터링 시 ->> = 대신 @> 선호 — @>는 GIN 인덱스를 활용하여 O(1) 검색 가능
-- 반면 ->>는 모든 행의 텍스트 값을 추출하며 스캔해야 함
SELECT * FROM products
WHERE attributes @> '{"brand": "Dell"}'::jsonb;

-- <@ : 포함됨 (오른쪽이 왼쪽 포함)
SELECT * FROM products
WHERE '{"brand": "Dell", "specs": {}}'::jsonb <@ attributes;

-- ? : 키 존재
SELECT * FROM products
WHERE attributes ? 'brand';

-- ?| : 키 중 하나 존재 (OR)
SELECT * FROM products
WHERE attributes ?| ARRAY['brand', 'manufacturer'];

-- ?& : 모든 키 존재 (AND)
SELECT * FROM products
WHERE attributes ?& ARRAY['brand', 'specs'];

-- || : 병합
SELECT '{"a": 1}'::jsonb || '{"b": 2}'::jsonb;
-- {"a": 1, "b": 2}

-- - : 키 제거
SELECT '{"a": 1, "b": 2}'::jsonb - 'a';
-- {"b": 2}

-- - : 배열 요소 제거 (인덱스)
SELECT '[1, 2, 3]'::jsonb - 1;
-- [1, 3]

-- #- : 경로로 제거
SELECT '{"a": {"b": 2}}'::jsonb #- '{a,b}';
-- {"a": {}}
```

### 3.3 조건 검색

```sql
-- 특정 값 포함
SELECT * FROM products
WHERE attributes @> '{"brand": "Dell"}';

-- 중첩 값 검색
SELECT * FROM products
WHERE attributes @> '{"specs": {"cpu": "i7"}}';

-- 배열 내 값 검색
-- 가정: attributes = {"tags": ["laptop", "electronics"]}
SELECT * FROM products
WHERE attributes->'tags' ? 'laptop';

-- 숫자 비교
SELECT * FROM products
WHERE (attributes->>'price')::numeric > 500;

-- 존재하지 않는 키 확인
SELECT * FROM products
WHERE NOT (attributes ? 'discontinued');

-- NULL 값 확인
SELECT * FROM products
WHERE attributes->'stock' IS NULL;

-- JSON 값이 null인지 확인 (JSON null과 SQL NULL 다름)
SELECT * FROM products
WHERE attributes->'stock' = 'null'::jsonb;
```

---

## 4. JSON 함수

### 4.1 추출 함수

```sql
-- jsonb_extract_path: 경로로 값 추출
SELECT jsonb_extract_path(attributes, 'specs', 'cpu') FROM products;

-- jsonb_extract_path_text: 텍스트로 추출
SELECT jsonb_extract_path_text(attributes, 'specs', 'cpu') FROM products;

-- jsonb_array_elements: 배열을 행으로 확장
SELECT jsonb_array_elements('[1, 2, 3]'::jsonb);
-- 1
-- 2
-- 3

-- jsonb_array_elements_text: 텍스트로 확장
SELECT jsonb_array_elements_text('["a", "b", "c"]'::jsonb);

-- jsonb_each: 객체를 키-값 행으로
SELECT * FROM jsonb_each('{"a": 1, "b": 2}'::jsonb);
-- key | value
-- a   | 1
-- b   | 2

-- jsonb_each_text: 텍스트 값으로
SELECT * FROM jsonb_each_text('{"a": 1, "b": "text"}'::jsonb);

-- jsonb_object_keys: 키 목록
SELECT jsonb_object_keys('{"a": 1, "b": 2}'::jsonb);
-- a
-- b

-- jsonb_array_length: 배열 길이
SELECT jsonb_array_length('[1, 2, 3]'::jsonb);
-- 3
```

### 4.2 변환 함수

```sql
-- jsonb_typeof: JSON 타입 확인
SELECT jsonb_typeof('"string"'::jsonb);  -- string
SELECT jsonb_typeof('123'::jsonb);       -- number
SELECT jsonb_typeof('true'::jsonb);      -- boolean
SELECT jsonb_typeof('null'::jsonb);      -- null
SELECT jsonb_typeof('[]'::jsonb);        -- array
SELECT jsonb_typeof('{}'::jsonb);        -- object

-- jsonb_strip_nulls: null 값 제거
SELECT jsonb_strip_nulls('{"a": 1, "b": null}'::jsonb);
-- {"a": 1}

-- jsonb_pretty: 보기 좋게 출력
SELECT jsonb_pretty('{"a":1,"b":2}'::jsonb);
/*
{
    "a": 1,
    "b": 2
}
*/

-- 배열을 PostgreSQL 배열로
SELECT ARRAY(SELECT jsonb_array_elements_text('["a", "b"]'::jsonb));
-- {a,b}

-- PostgreSQL 배열을 JSON 배열로
SELECT to_jsonb(ARRAY['a', 'b']);
-- ["a", "b"]
```

### 4.3 집계 함수

```sql
-- 여러 행을 JSON 배열로
SELECT jsonb_agg(attributes) FROM products;

-- 필터링하여 집계
SELECT jsonb_agg(attributes) FILTER (WHERE name LIKE 'L%') FROM products;

-- 객체로 집계
SELECT jsonb_object_agg(id, attributes) FROM products;

-- 배열 합치기
SELECT jsonb_agg(elem)
FROM products, jsonb_array_elements(attributes->'tags') AS elem;
```

---

## 5. 인덱싱과 성능

### 5.1 GIN 인덱스

```sql
-- 기본 GIN: @>, ?, ?|, ?& 지원 — 다양한 연산자 타입이 필요한 쿼리에 사용
CREATE INDEX idx_products_attrs
ON products USING GIN (attributes);

-- jsonb_path_ops: 인덱스 크기 2~3배 작고 @> 검색 빠름 — 포함 쿼리만 필요할 때 선택
CREATE INDEX idx_products_attrs_path
ON products USING GIN (attributes jsonb_path_ops);

-- 특정 키에 대한 인덱스
CREATE INDEX idx_products_brand
ON products USING GIN ((attributes->'brand'));

-- B-tree 인덱스 (특정 값 비교용)
CREATE INDEX idx_products_brand_btree
ON products ((attributes->>'brand'));

-- 함수 기반 인덱스
CREATE INDEX idx_products_price
ON products (((attributes->>'price')::numeric));
```

### 5.2 인덱스 사용 확인

```sql
-- 실행 계획 확인
EXPLAIN ANALYZE
SELECT * FROM products
WHERE attributes @> '{"brand": "Dell"}';

-- GIN 인덱스가 사용되면:
-- Bitmap Index Scan on idx_products_attrs

-- 인덱스 크기 확인
SELECT pg_size_pretty(pg_indexes_size('products'));
```

### 5.3 성능 최적화

```sql
-- 자주 조회하는 키는 별도 컬럼으로 추출 — 스칼라 컬럼의 B-tree 인덱스가
-- 전체 JSONB 문서의 GIN 인덱스보다 빠르고 메모리 효율적
ALTER TABLE products ADD COLUMN brand VARCHAR(100);
UPDATE products SET brand = attributes->>'brand';
CREATE INDEX idx_products_brand_col ON products(brand);

-- Partial 인덱스
CREATE INDEX idx_active_products
ON products USING GIN (attributes)
WHERE (attributes->>'active')::boolean = true;

-- 복합 인덱스
CREATE INDEX idx_products_composite
ON products (name, (attributes->>'brand'));

-- 통계 업데이트
ANALYZE products;
```

---

## 6. 실전 패턴

### 6.1 스키마리스 테이블

```sql
-- JSONB는 이벤트 로그에 최적: 이벤트 타입마다 필드가 다르므로 고정 스키마는
-- 빈번한 ALTER TABLE이나 넓은 NULL 허용 컬럼이 필요. JSONB는 어떤 형태든 수용 가능
CREATE TABLE events (
    id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    occurred_at TIMESTAMPTZ DEFAULT NOW(),
    data JSONB NOT NULL
);

CREATE INDEX idx_events_type ON events(event_type);
CREATE INDEX idx_events_data ON events USING GIN (data);
CREATE INDEX idx_events_occurred ON events(occurred_at);

-- 이벤트 삽입
INSERT INTO events (event_type, data) VALUES
('user_signup', '{"user_id": 123, "email": "user@example.com"}'),
('purchase', '{"user_id": 123, "product_id": 456, "amount": 99.99}'),
('page_view', '{"user_id": 123, "page": "/products", "referrer": "google"}');

-- 이벤트 조회
SELECT * FROM events
WHERE event_type = 'purchase'
AND (data->>'amount')::numeric > 50
AND occurred_at > NOW() - INTERVAL '7 days';
```

### 6.2 EAV (Entity-Attribute-Value) 대체

```sql
-- EAV는 속성당 한 행 필요 — "제품 X의 모든 속성" 조회 시 다수의 행을 조인/피벗해야 하며,
-- 타입 안전성 상실 (모든 값이 VARCHAR)
CREATE TABLE product_attributes_eav (
    product_id INT,
    attribute_name VARCHAR(100),
    attribute_value VARCHAR(255)
);

-- JSONB는 모든 속성을 하나의 인덱싱된 컬럼에 통합 — 단일 행 읽기,
-- 네이티브 타입(number, boolean, array), GIN 가속 포함 쿼리 지원
CREATE TABLE products_jsonb (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    base_price DECIMAL(10,2),
    attributes JSONB DEFAULT '{}'
);

-- 다양한 속성 저장
INSERT INTO products_jsonb (name, base_price, attributes) VALUES
('T-Shirt', 29.99, '{"size": "M", "color": "blue", "material": "cotton"}'),
('Laptop', 999.99, '{"cpu": "i7", "ram": 16, "storage": "512GB SSD"}'),
('Book', 15.99, '{"author": "John Doe", "pages": 300, "isbn": "123-456"}');

-- 동적 필터링
SELECT * FROM products_jsonb
WHERE attributes @> '{"color": "blue"}'
OR attributes @> '{"ram": 16}';
```

### 6.3 버전 관리

```sql
-- 문서 버전 관리
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200),
    current_version INT DEFAULT 1,
    content JSONB
);

CREATE TABLE document_versions (
    id SERIAL PRIMARY KEY,
    document_id INT REFERENCES documents(id),
    version INT,
    content JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by INT
);

-- 트리거로 버전 자동 저장
CREATE OR REPLACE FUNCTION save_document_version()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO document_versions (document_id, version, content, created_by)
    VALUES (OLD.id, OLD.current_version, OLD.content, current_setting('app.user_id')::int);

    NEW.current_version := OLD.current_version + 1;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_document_version
BEFORE UPDATE ON documents
FOR EACH ROW
WHEN (OLD.content IS DISTINCT FROM NEW.content)
EXECUTE FUNCTION save_document_version();
```

### 6.4 JSON Schema 검증

```sql
-- CHECK 제약조건으로 간단한 검증
ALTER TABLE products ADD CONSTRAINT valid_attributes CHECK (
    attributes ? 'brand' AND
    jsonb_typeof(attributes->'brand') = 'string'
);

-- 함수로 복잡한 검증
CREATE OR REPLACE FUNCTION validate_product_attributes(attrs JSONB)
RETURNS BOOLEAN AS $$
BEGIN
    -- 필수 필드 확인
    IF NOT (attrs ? 'brand') THEN
        RETURN FALSE;
    END IF;

    -- 타입 확인
    IF jsonb_typeof(attrs->'brand') != 'string' THEN
        RETURN FALSE;
    END IF;

    -- specs가 있으면 객체여야 함
    IF attrs ? 'specs' AND jsonb_typeof(attrs->'specs') != 'object' THEN
        RETURN FALSE;
    END IF;

    RETURN TRUE;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

ALTER TABLE products ADD CONSTRAINT chk_attributes
CHECK (validate_product_attributes(attributes));
```

---

## 7. 연습 문제

### 연습 1: 사용자 설정 저장
```sql
-- 요구사항:
-- 1. 사용자별 설정을 JSONB로 저장하는 테이블 생성
-- 2. 기본 설정 병합 함수 작성
-- 3. 특정 설정 조회/업데이트 함수 작성

-- 스키마 및 함수 작성:
```

### 연습 2: JSON 집계 보고서
```sql
-- 요구사항:
-- 주문 테이블에서 다음 JSON 형식의 보고서 생성:
-- {
--   "total_orders": 100,
--   "total_revenue": 5000.00,
--   "by_status": {"pending": 20, "completed": 80},
--   "top_products": [{"id": 1, "count": 50}, ...]
-- }

-- 쿼리 작성:
```

### 연습 3: JSON 검색 최적화
```sql
-- 요구사항:
-- 1. 100만 행의 이벤트 데이터 생성
-- 2. 다양한 인덱스 비교
-- 3. 최적의 인덱스 전략 수립

-- 테스트 및 분석:
```

### 연습 4: 계층적 JSON 처리
```sql
-- 요구사항:
-- 조직 구조 JSON 데이터 처리:
-- {"name": "CEO", "children": [{"name": "CTO", "children": [...]}]}
-- 모든 노드 평면화, 경로 추출 등

-- 재귀 CTE 활용:
```

---

## 참고 자료

- [PostgreSQL JSON Functions](https://www.postgresql.org/docs/current/functions-json.html)
- [PostgreSQL JSON Types](https://www.postgresql.org/docs/current/datatype-json.html)
- [GIN Index](https://www.postgresql.org/docs/current/gin.html)

---

**이전**: [백업과 운영](./13_Backup_and_Operations.md) | **다음**: [쿼리 최적화](./15_Query_Optimization.md)

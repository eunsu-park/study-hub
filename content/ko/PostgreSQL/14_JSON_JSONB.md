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
1. [JSON vs JSONB](#1-json-vs-jsonb)
2. [JSON 데이터 저장](#2-json-데이터-저장)
3. [JSON 연산자](#3-json-연산자)
4. [JSON 함수](#4-json-함수)
5. [인덱싱과 성능](#5-인덱싱과-성능)
6. [실전 패턴](#6-실전-패턴)
7. [연습 문제](#7-연습-문제)

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

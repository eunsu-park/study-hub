# 14. PostgreSQL JSON/JSONB Features

**Previous**: [Backup and Operations](./13_Backup_and_Operations.md) | **Next**: [Query Optimization](./15_Query_Optimization.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between the JSON and JSONB data types and choose the appropriate one for a given use case
2. Store, modify, and delete JSON data using PostgreSQL's built-in functions and operators
3. Query nested JSON structures using arrow operators (`->`, `->>`, `#>`, `#>>`)
4. Apply containment (`@>`), existence (`?`), and merge (`||`) operators on JSONB data
5. Create GIN indexes on JSONB columns and verify their usage with EXPLAIN ANALYZE
6. Implement real-world patterns including schema-less event logging, EAV replacement, and document versioning

---

Modern applications frequently deal with semi-structured data -- user preferences, API responses, event payloads, and configuration blobs -- that do not fit neatly into rigid relational columns. PostgreSQL's native JSON/JSONB support lets you store and query this flexible data alongside your relational tables, with full indexing and transactional guarantees. This eliminates the need for a separate document database in many architectures, simplifying your stack while preserving the power of SQL.

## Table of Contents
1. [JSON vs JSONB](#1-json-vs-jsonb)
2. [Storing JSON Data](#2-storing-json-data)
3. [JSON Operators](#3-json-operators)
4. [JSON Functions](#4-json-functions)
5. [Indexing and Performance](#5-indexing-and-performance)
6. [Real-world Patterns](#6-real-world-patterns)
7. [Practice Problems](#7-practice-problems)

---

## 1. JSON vs JSONB

### 1.1 Type Comparison

```
┌─────────────────────────────────────────────────────────────┐
│                    JSON vs JSONB                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  JSON                          JSONB                        │
│  ────────────────────         ────────────────────         │
│  • Stored as text              • Stored as binary          │
│  • Preserves input             • Parsed before storage     │
│  • Preserves whitespace/order • Removes whitespace, sorts  │
│  • Allows duplicate keys       • Keeps last key value      │
│  • Faster storage              • Slightly slower storage   │
│  • Slower processing (re-parse)• Faster processing         │
│  • Limited indexing            • GIN index support         │
│                                                             │
│  Recommendation: Use JSONB in most cases                    │
│        Use JSON only when preserving original format needed │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Basic Usage

```sql
-- JSONB stores pre-parsed binary — supports GIN indexing and containment operators (@>, ?)
-- JSON stores raw text — reparsed on every access, no index support; use only for write-once/read-raw scenarios
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    attributes JSONB,    -- JSONB recommended
    raw_data JSON        -- For preserving original
);

-- Insert data
INSERT INTO products (name, attributes) VALUES
('Laptop', '{"brand": "Dell", "specs": {"cpu": "i7", "ram": 16}}'),
('Phone', '{"brand": "Apple", "specs": {"model": "iPhone 15", "storage": 256}}');

-- JSON format validation
SELECT '{"valid": true}'::jsonb;  -- Success
SELECT '{invalid}'::jsonb;        -- Error: invalid JSON
```

---

## 2. Storing JSON Data

### 2.1 JSON Creation Functions

```sql
-- json_build_object: Create object from key-value pairs
SELECT json_build_object(
    'name', 'John',
    'age', 30,
    'active', true
);
-- {"name": "John", "age": 30, "active": true}

-- jsonb_build_object (JSONB version)
SELECT jsonb_build_object(
    'product', 'Laptop',
    'price', 999.99
);

-- json_build_array: Create array
SELECT json_build_array(1, 2, 'three', true, null);
-- [1, 2, "three", true, null]

-- row_to_json: Row to JSON
SELECT row_to_json(t)
FROM (SELECT 1 AS id, 'test' AS name) t;
-- {"id": 1, "name": "test"}

-- to_jsonb: Convert value to JSONB
SELECT to_jsonb(ARRAY[1, 2, 3]);
-- [1, 2, 3]

-- json_agg: Aggregate rows into array
SELECT json_agg(name) FROM products;
-- ["Laptop", "Phone"]

-- jsonb_object_agg: Aggregate key-value pairs into object
SELECT jsonb_object_agg(name, id) FROM products;
-- {"Laptop": 1, "Phone": 2}
```

### 2.2 Modifying JSON Data

```sql
-- jsonb_set: Set/add value
UPDATE products
SET attributes = jsonb_set(attributes, '{specs,ram}', '32')
WHERE name = 'Laptop';

-- Add nested path (create_if_missing = true)
UPDATE products
SET attributes = jsonb_set(
    attributes,
    '{specs,gpu}',
    '"RTX 4090"',
    true  -- Create path if missing
)
WHERE name = 'Laptop';

-- Update multiple values at once
UPDATE products
SET attributes = attributes || '{"color": "silver", "weight": 2.1}'
WHERE name = 'Laptop';

-- Delete key
UPDATE products
SET attributes = attributes - 'color'
WHERE name = 'Laptop';

-- Delete nested key
UPDATE products
SET attributes = attributes #- '{specs,gpu}'
WHERE name = 'Laptop';

-- Add array element
UPDATE products
SET attributes = jsonb_set(
    attributes,
    '{tags}',
    COALESCE(attributes->'tags', '[]'::jsonb) || '"new_tag"'
);
```

---

## 3. JSON Operators

### 3.1 Access Operators

```sql
-- -> : JSON object/array element (returns JSON)
SELECT attributes->'brand' FROM products;
-- "Dell" (JSON with quotes)

-- ->> : Extract as text
SELECT attributes->>'brand' FROM products;
-- Dell (text)

-- #> : Access by path (returns JSON)
SELECT attributes#>'{specs,cpu}' FROM products;
-- "i7"

-- #>> : Access by path (returns text)
SELECT attributes#>>'{specs,cpu}' FROM products;
-- i7

-- Array access
SELECT '[1, 2, 3]'::jsonb->0;   -- 1
SELECT '[1, 2, 3]'::jsonb->-1;  -- 3 (last)
SELECT '[1, 2, 3]'::jsonb->10;  -- NULL (out of range)
```

### 3.2 Comparison Operators (JSONB only)

```sql
-- = : Equality comparison
SELECT * FROM products
WHERE attributes->'brand' = '"Dell"'::jsonb;

-- @> : Contains (left contains right)
-- Prefer @> over ->> = for filtering — @> leverages GIN indexes for O(1) lookups
-- instead of scanning and extracting every row's text value
SELECT * FROM products
WHERE attributes @> '{"brand": "Dell"}'::jsonb;

-- <@ : Contained by (right contains left)
SELECT * FROM products
WHERE '{"brand": "Dell", "specs": {}}'::jsonb <@ attributes;

-- ? : Key exists
SELECT * FROM products
WHERE attributes ? 'brand';

-- ?| : Any key exists (OR)
SELECT * FROM products
WHERE attributes ?| ARRAY['brand', 'manufacturer'];

-- ?& : All keys exist (AND)
SELECT * FROM products
WHERE attributes ?& ARRAY['brand', 'specs'];

-- || : Merge
SELECT '{"a": 1}'::jsonb || '{"b": 2}'::jsonb;
-- {"a": 1, "b": 2}

-- - : Remove key
SELECT '{"a": 1, "b": 2}'::jsonb - 'a';
-- {"b": 2}

-- - : Remove array element (by index)
SELECT '[1, 2, 3]'::jsonb - 1;
-- [1, 3]

-- #- : Remove by path
SELECT '{"a": {"b": 2}}'::jsonb #- '{a,b}';
-- {"a": {}}
```

### 3.3 Conditional Searches

```sql
-- Contains specific value
SELECT * FROM products
WHERE attributes @> '{"brand": "Dell"}';

-- Nested value search
SELECT * FROM products
WHERE attributes @> '{"specs": {"cpu": "i7"}}';

-- Search in array
-- Assuming: attributes = {"tags": ["laptop", "electronics"]}
SELECT * FROM products
WHERE attributes->'tags' ? 'laptop';

-- Numeric comparison
SELECT * FROM products
WHERE (attributes->>'price')::numeric > 500;

-- Check for missing key
SELECT * FROM products
WHERE NOT (attributes ? 'discontinued');

-- NULL value check
SELECT * FROM products
WHERE attributes->'stock' IS NULL;

-- Check if JSON value is null (JSON null vs SQL NULL differ)
SELECT * FROM products
WHERE attributes->'stock' = 'null'::jsonb;
```

---

## 4. JSON Functions

### 4.1 Extraction Functions

```sql
-- jsonb_extract_path: Extract value by path
SELECT jsonb_extract_path(attributes, 'specs', 'cpu') FROM products;

-- jsonb_extract_path_text: Extract as text
SELECT jsonb_extract_path_text(attributes, 'specs', 'cpu') FROM products;

-- jsonb_array_elements: Expand array to rows
SELECT jsonb_array_elements('[1, 2, 3]'::jsonb);
-- 1
-- 2
-- 3

-- jsonb_array_elements_text: Expand as text
SELECT jsonb_array_elements_text('["a", "b", "c"]'::jsonb);

-- jsonb_each: Object to key-value rows
SELECT * FROM jsonb_each('{"a": 1, "b": 2}'::jsonb);
-- key | value
-- a   | 1
-- b   | 2

-- jsonb_each_text: With text values
SELECT * FROM jsonb_each_text('{"a": 1, "b": "text"}'::jsonb);

-- jsonb_object_keys: List of keys
SELECT jsonb_object_keys('{"a": 1, "b": 2}'::jsonb);
-- a
-- b

-- jsonb_array_length: Array length
SELECT jsonb_array_length('[1, 2, 3]'::jsonb);
-- 3
```

### 4.2 Conversion Functions

```sql
-- jsonb_typeof: Check JSON type
SELECT jsonb_typeof('"string"'::jsonb);  -- string
SELECT jsonb_typeof('123'::jsonb);       -- number
SELECT jsonb_typeof('true'::jsonb);      -- boolean
SELECT jsonb_typeof('null'::jsonb);      -- null
SELECT jsonb_typeof('[]'::jsonb);        -- array
SELECT jsonb_typeof('{}'::jsonb);        -- object

-- jsonb_strip_nulls: Remove null values
SELECT jsonb_strip_nulls('{"a": 1, "b": null}'::jsonb);
-- {"a": 1}

-- jsonb_pretty: Pretty print
SELECT jsonb_pretty('{"a":1,"b":2}'::jsonb);
/*
{
    "a": 1,
    "b": 2
}
*/

-- Array to PostgreSQL array
SELECT ARRAY(SELECT jsonb_array_elements_text('["a", "b"]'::jsonb));
-- {a,b}

-- PostgreSQL array to JSON array
SELECT to_jsonb(ARRAY['a', 'b']);
-- ["a", "b"]
```

### 4.3 Aggregate Functions

```sql
-- Aggregate multiple rows into JSON array
SELECT jsonb_agg(attributes) FROM products;

-- Filter while aggregating
SELECT jsonb_agg(attributes) FILTER (WHERE name LIKE 'L%') FROM products;

-- Aggregate as object
SELECT jsonb_object_agg(id, attributes) FROM products;

-- Merge arrays
SELECT jsonb_agg(elem)
FROM products, jsonb_array_elements(attributes->'tags') AS elem;
```

---

## 5. Indexing and Performance

### 5.1 GIN Index

```sql
-- Default GIN: supports @>, ?, ?|, ?& — use when queries need multiple operator types
CREATE INDEX idx_products_attrs
ON products USING GIN (attributes);

-- jsonb_path_ops: 2-3x smaller index, faster @> lookups — choose when only containment queries are needed
CREATE INDEX idx_products_attrs_path
ON products USING GIN (attributes jsonb_path_ops);

-- Index specific key
CREATE INDEX idx_products_brand
ON products USING GIN ((attributes->'brand'));

-- B-tree index (for specific value comparison)
CREATE INDEX idx_products_brand_btree
ON products ((attributes->>'brand'));

-- Function-based index
CREATE INDEX idx_products_price
ON products (((attributes->>'price')::numeric));
```

### 5.2 Check Index Usage

```sql
-- Check execution plan
EXPLAIN ANALYZE
SELECT * FROM products
WHERE attributes @> '{"brand": "Dell"}';

-- If GIN index is used:
-- Bitmap Index Scan on idx_products_attrs

-- Check index size
SELECT pg_size_pretty(pg_indexes_size('products'));
```

### 5.3 Performance Optimization

```sql
-- Extract hot-path keys into dedicated columns — B-tree on a scalar column is faster
-- and more memory-efficient than GIN on the entire JSONB document
ALTER TABLE products ADD COLUMN brand VARCHAR(100);
UPDATE products SET brand = attributes->>'brand';
CREATE INDEX idx_products_brand_col ON products(brand);

-- Partial index
CREATE INDEX idx_active_products
ON products USING GIN (attributes)
WHERE (attributes->>'active')::boolean = true;

-- Composite index
CREATE INDEX idx_products_composite
ON products (name, (attributes->>'brand'));

-- Update statistics
ANALYZE products;
```

---

## 6. Real-world Patterns

### 6.1 Schema-less Table

```sql
-- JSONB shines for event logs: each event type has different fields, so a fixed schema
-- would require constant ALTER TABLE or wide nullable columns. JSONB accommodates any shape.
CREATE TABLE events (
    id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    occurred_at TIMESTAMPTZ DEFAULT NOW(),
    data JSONB NOT NULL
);

CREATE INDEX idx_events_type ON events(event_type);
CREATE INDEX idx_events_data ON events USING GIN (data);
CREATE INDEX idx_events_occurred ON events(occurred_at);

-- Insert events
INSERT INTO events (event_type, data) VALUES
('user_signup', '{"user_id": 123, "email": "user@example.com"}'),
('purchase', '{"user_id": 123, "product_id": 456, "amount": 99.99}'),
('page_view', '{"user_id": 123, "page": "/products", "referrer": "google"}');

-- Query events
SELECT * FROM events
WHERE event_type = 'purchase'
AND (data->>'amount')::numeric > 50
AND occurred_at > NOW() - INTERVAL '7 days';
```

### 6.2 Replacing EAV

```sql
-- EAV requires one row per attribute — querying "all attributes for product X" means
-- joining/pivoting many rows, and you lose type safety (everything is VARCHAR)
CREATE TABLE product_attributes_eav (
    product_id INT,
    attribute_name VARCHAR(100),
    attribute_value VARCHAR(255)
);

-- JSONB collapses all attributes into one indexed column — single-row reads,
-- native types (number, boolean, array), and GIN-accelerated containment queries
CREATE TABLE products_jsonb (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    base_price DECIMAL(10,2),
    attributes JSONB DEFAULT '{}'
);

-- Store various attributes
INSERT INTO products_jsonb (name, base_price, attributes) VALUES
('T-Shirt', 29.99, '{"size": "M", "color": "blue", "material": "cotton"}'),
('Laptop', 999.99, '{"cpu": "i7", "ram": 16, "storage": "512GB SSD"}'),
('Book', 15.99, '{"author": "John Doe", "pages": 300, "isbn": "123-456"}');

-- Dynamic filtering
SELECT * FROM products_jsonb
WHERE attributes @> '{"color": "blue"}'
OR attributes @> '{"ram": 16}';
```

### 6.3 Version Management

```sql
-- Document version management
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

-- Auto-save version with trigger
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

### 6.4 JSON Schema Validation

```sql
-- Simple validation with CHECK constraint
ALTER TABLE products ADD CONSTRAINT valid_attributes CHECK (
    attributes ? 'brand' AND
    jsonb_typeof(attributes->'brand') = 'string'
);

-- Complex validation with function
CREATE OR REPLACE FUNCTION validate_product_attributes(attrs JSONB)
RETURNS BOOLEAN AS $$
BEGIN
    -- Required field check
    IF NOT (attrs ? 'brand') THEN
        RETURN FALSE;
    END IF;

    -- Type check
    IF jsonb_typeof(attrs->'brand') != 'string' THEN
        RETURN FALSE;
    END IF;

    -- If specs exists, must be object
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

## 7. Practice Problems

### Practice 1: User Settings Storage
```sql
-- Requirements:
-- 1. Create table to store user settings in JSONB
-- 2. Write function to merge default settings
-- 3. Write functions to query/update specific settings

-- Write schema and functions:
```

### Practice 2: JSON Aggregate Report
```sql
-- Requirements:
-- Generate report from orders table in JSON format:
-- {
--   "total_orders": 100,
--   "total_revenue": 5000.00,
--   "by_status": {"pending": 20, "completed": 80},
--   "top_products": [{"id": 1, "count": 50}, ...]
-- }

-- Write query:
```

### Practice 3: JSON Search Optimization
```sql
-- Requirements:
-- 1. Generate 1 million rows of event data
-- 2. Compare different indexes
-- 3. Establish optimal indexing strategy

-- Test and analyze:
```

### Practice 4: Hierarchical JSON Processing
```sql
-- Requirements:
-- Process organizational structure JSON:
-- {"name": "CEO", "children": [{"name": "CTO", "children": [...]}]}
-- Flatten all nodes, extract paths, etc.

-- Use recursive CTE:
```

---

## References

- [PostgreSQL JSON Functions](https://www.postgresql.org/docs/current/functions-json.html)
- [PostgreSQL JSON Types](https://www.postgresql.org/docs/current/datatype-json.html)
- [GIN Index](https://www.postgresql.org/docs/current/gin.html)

---

**Previous**: [Backup and Operations](./13_Backup_and_Operations.md) | **Next**: [Query Optimization](./15_Query_Optimization.md)
